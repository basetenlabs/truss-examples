import json
from collections import OrderedDict

import tensorrt_llm
import torch
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from whisper_trt.apply_timestamps_rule_processor import ApplyTimestampsRule
from whisper_trt.language_detection_processor import LanguageDetectionRules


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config_path = engine_dir / "encoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.n_mels = config["pretrained_config"]["n_mels"]
        self.dtype = config["pretrained_config"]["dtype"]
        self.num_languages = config["pretrained_config"]["num_languages"]

    def get_session(self, engine_dir):
        serialize_path = engine_dir / "encoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel):
        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device,
        )

        inputs = OrderedDict()
        inputs["x"] = mel
        inputs["input_lengths"] = input_lengths

        output_list = [
            TensorInfo("x", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), input_lengths.shape),
        ]

        output_info = (self.session).infer_shapes(output_list)

        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        # TODO: fail a little nicer
        assert ok, "Engine execution failed"
        stream.synchronize()
        audio_features = outputs["output"]
        return audio_features


class WhisperDecoding:
    def __init__(
        self,
        engine_dir,
        runtime_mapping,
        lang_token_ids,
        eot_id,
        sot_id,
        no_speech_id,
        pad_id,
        notimestamps_id,
        debug_mode=False,
    ):
        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode
        )
        self.lang_token_ids = lang_token_ids
        self.eot_id = eot_id
        self.sot_id = sot_id
        self.no_speech_id = no_speech_id
        self.pad_id = pad_id
        self.notimestamps_id = notimestamps_id
        self.debug_mode = debug_mode

    def get_config(self, engine_dir):
        config_path = engine_dir / "decoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["pretrained_config"])
        decoder_config.update(config["build_config"])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / "decoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_attention_heads"],
            num_kv_heads=self.decoder_config["num_attention_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            cross_attention=True,
            num_layers=self.decoder_config["num_hidden_layers"],
            gpt_attention_plugin=self.decoder_config["plugin_config"][
                "gpt_attention_plugin"
            ],
            remove_input_padding=self.decoder_config["plugin_config"][
                "remove_input_padding"
            ],
            has_position_embedding=self.decoder_config["has_position_embedding"],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        # Override gather_tree to fix the logits_processor issue
        # the sequence_length_buffer needs to be cloned to avoid changing the original tensor.
        def gather_tree(*args, **kwargs):
            args = list(args)
            args[0] = args[0].clone()
            return torch.ops.tensorrt_llm.gather_tree(*tuple(args), **kwargs)

        decoder_generation_session.gather_tree = gather_tree

        return decoder_generation_session

    def detect_language(
        self,
        encoder_outputs,
        num_beams,
    ):
        batch_size = encoder_outputs.shape[0]
        decoder_input_ids = torch.tensor(self.sot_id).repeat([batch_size, 1])
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(batch_size)],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(batch_size)],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones([batch_size, 1, encoder_outputs.shape[1]]).int().cuda()
        )

        # generation config
        sampling_config = SamplingConfig(
            end_id=self.eot_id,
            pad_id=self.pad_id,
            num_beams=num_beams,
            use_beam_hyps=True,
            return_dict=True,
            output_sequence_lengths=True,
            max_new_tokens=1,
        )
        sampling_config.output_log_probs = False
        sampling_config.output_logits = True

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            1,  # max_new_tokens
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        language_logits_processor = LanguageDetectionRules(
            num_beams=num_beams,
            batch_size=batch_size,
            nospeech_id=self.no_speech_id,
            lang_token_ids=self.lang_token_ids,
        )

        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
            logits_processor=language_logits_processor,
            return_dict=False,
        )
        # Only return the first beam and drop the input token id
        return output_ids[:, 0, 1:]

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        eot_id,
        num_beams,
        max_new_tokens,
        notimestamps_id=None,
        remove_input_ids=False,
        no_speech_id=None,
        no_speech_probability_threshold=0.6,
        use_timstamps_processor=False,
        duration_secs=[],
    ) -> torch.Tensor:
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones([encoder_outputs.shape[0], 1, encoder_outputs.shape[1]])
            .int()
            .cuda()
        )

        # generation config
        sampling_config = SamplingConfig(
            # end_id=self.eot_id,
            # pad_id=self.pad_id,
            end_id=eot_id,
            pad_id=eot_id,
            num_beams=num_beams,
            use_beam_hyps=True,
            return_dict=True,
            output_sequence_lengths=True,
            max_new_tokens=max_new_tokens,
        )
        sampling_config.output_log_probs = True

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        apply_timestamp_rules_logits_processor = None
        if use_timstamps_processor:
            apply_timestamp_rules_logits_processor = ApplyTimestampsRule(
                num_beams=num_beams,
                batch_size=len(decoder_input_ids),
                duration_secs=duration_secs,
                notimestamps_id=notimestamps_id,
                timestamp_begin_id=notimestamps_id + 1,
                eot_id=eot_id,
                # We are essentially assuming that all the inputs to the decoder are the same length and have been
                # padded before being sent to this function.
                input_prompt_ids_offset=decoder_max_input_length,
            )

        result_dict = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
            logits_processor=apply_timestamp_rules_logits_processor,
            return_dict=True,
        )
        torch.cuda.synchronize()

        # TODO: update to use cum log probs instead
        output_ids, log_probs = result_dict["output_ids"], result_dict["log_probs"]
        return_output_ids = []
        return_avg_log_probs = []
        for output_id, log_prob in zip(output_ids, log_probs):
            results_mask = torch.any(output_id != 50257, dim=-1)
            output_id = output_id[results_mask]
            log_prob = log_prob[results_mask]

            sorted_indices = torch.argsort(
                torch.sum(log_prob, dim=-1), dim=0, descending=True
            )
            output_id = output_id[sorted_indices]

            if no_speech_id is not None and output_id[0][-1] == no_speech_id:
                no_speech_probability = torch.softmax(
                    log_prob[sorted_indices][0], dim=-1
                )[-1]
                if no_speech_probability < no_speech_probability_threshold:
                    # logging.warn("picking up next beam even though it might be worse")
                    output_id = output_id[1]
                    l = log_prob[1]
                else:
                    output_id = output_id[0]
                    l = log_prob[0]
            else:
                output_id = output_id[0]
                l = log_prob[0]

            # output_id = output_id[0]
            # l = log_prob[0]

            if remove_input_ids:
                output_id = output_id[decoder_max_input_length:]

            return_output_ids.append(output_id.cpu())
            return_avg_log_probs.append(torch.mean(l[l != 0]))

        return torch.stack(return_output_ids), torch.stack(return_avg_log_probs)
