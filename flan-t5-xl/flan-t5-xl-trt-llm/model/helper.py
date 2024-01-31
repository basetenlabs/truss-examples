import json
from pathlib import Path

import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm import logger
from tensorrt_llm._utils import torch_to_numpy, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return "{}_{}_tp{}_rank{}.engine".format(model, dtype, tp_size, rank)
    return "{}_{}_tp{}_pp{}_rank{}.engine".format(model, dtype, tp_size, pp_size, rank)


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)

    builder_config = config["builder_config"]
    plugin_config = config["plugin_config"]
    use_gpt_attention_plugin = plugin_config["gpt_attention_plugin"]
    remove_input_padding = plugin_config["remove_input_padding"]
    tp_size = builder_config["tensor_parallel"]
    pp_size = builder_config["pipeline_parallel"]
    gpus_per_node = builder_config["gpus_per_node"]
    world_size = tp_size * pp_size
    assert (
        world_size == tensorrt_llm.mpi_world_size()
    ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
    num_heads = builder_config["num_heads"]
    hidden_size = builder_config["hidden_size"]
    head_size = builder_config["head_size"]
    vocab_size = builder_config["vocab_size"]
    num_layers = builder_config["num_layers"]
    num_kv_heads = builder_config.get("num_kv_heads", num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = builder_config["cross_attention"]
    has_position_embedding = builder_config["has_position_embedding"]
    has_token_type_embedding = builder_config["has_token_type_embedding"]
    use_custom_all_reduce = config["plugin_config"].get("use_custom_all_reduce", False)
    dtype = builder_config["precision"]

    gather_context_logits = builder_config.get("gather_context_logits", False)
    gather_generation_logits = builder_config.get("gather_generation_logits", False)
    max_prompt_embedding_table_size = builder_config.get(
        "max_prompt_embedding_table_size", 0
    )

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )

    return model_config, tp_size, pp_size, gpus_per_node, dtype


class TRTLLMEncDecModel:
    def __init__(self, engine_name, engine_dir, debug_mode=False, skip_encoder=False):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()
        self.skip_encoder = skip_encoder

        engine_dir = Path(engine_dir)

        def engine_setup(component):
            # model config
            config_path = engine_dir / component / "config.json"
            logger.info(f"Using config path {config_path}")
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path
            )

            # MGMN config
            world_size = tp_size * pp_size
            runtime_rank = tensorrt_llm.mpi_rank()
            assert (
                runtime_rank < world_size
            ), "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            runtime_mapping = tensorrt_llm.Mapping(
                world_size,
                runtime_rank,
                tp_size=tp_size,
                pp_size=pp_size,
                gpus_per_node=gpus_per_node,
            )

            # load engine
            engine_fname = get_engine_name(
                engine_name, dtype, tp_size, pp_size, runtime_rank
            )
            with open(engine_dir / component / engine_fname, "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config

        if not skip_encoder:
            (
                self.encoder_model_config,
                self.encoder_runtime_mapping,
                encoder_engine_buffer,
            ) = engine_setup(component="encoder")

            # for Pipeline Parallelism in encoder
            self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
                self.encoder_runtime_mapping.tp_size,
                self.encoder_runtime_mapping.pp_size,
                self.encoder_runtime_mapping.rank,
            )

            # session setup
            self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
                encoder_engine_buffer
            )
        else:
            (
                self.encoder_model_config,
                self.encoder_runtime_mapping,
                encoder_engine_buffer,
            ) = (None, None, None)
            self.nccl_comm, self.encoder_session = None, None

        (
            self.decoder_model_config,
            self.decoder_runtime_mapping,
            decoder_engine_buffer,
        ) = engine_setup(component="decoder")
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
            debug_mode=debug_mode,
        )
        self.stream = torch.cuda.current_stream().cuda_stream

    @classmethod
    def from_engine(cls, engine_name, engine_dir, debug_mode=False, skip_encoder=False):
        return cls(
            engine_name, engine_dir, debug_mode=debug_mode, skip_encoder=skip_encoder
        )

    def process_input(
        self, input_ids, remove_input_padding=False, pad_token_id=0, prompt_tasks=None
    ):
        if remove_input_padding:
            # in remove padding mode --> flatten input, calculate actual length and max length
            # Note: 1st token should never be removed, even if it is pad_token_id
            first_ids = input_ids[:, 0]
            input_ids = input_ids[:, 1:]
            input_lengths = 1 + (input_ids != pad_token_id).sum(dim=1).type(
                torch.IntTensor
            ).to(
                self.device
            )  # [batch_size]
            new_ids = []
            for i in range(len(input_ids)):
                row = input_ids[i, :]
                row = row[row != pad_token_id]
                new_ids.append(
                    torch.cat((torch.IntTensor([first_ids[i]]).to(self.device), row))
                )
            input_ids = torch.cat(new_ids)  # [num_tokens]
            if prompt_tasks is not None:
                prompt_tasks = prompt_tasks[: input_ids.shape[0]]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1
                + (input_ids[:, 1:] != pad_token_id)
                .sum(dim=1)
                .type(torch.IntTensor)
                .to(self.device),
                dtype=torch.int32,
                device=self.device,
            )
        max_input_length = torch.max(input_lengths).item()
        return input_ids, input_lengths, max_input_length, prompt_tasks

    def encoder_run(
        self,
        input_ids,
        input_lengths,
        max_input_length,
        position_ids=None,
        token_type_ids=None,
        debug_mode=False,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        attention_mask=None,
    ):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = (
            self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        )
        if input_ids.dim() == 1:
            hidden_states_shape = (input_ids.shape[0], hidden_size)  # [num_tokens,D]
        else:
            hidden_states_shape = (
                input_ids.shape[0],
                input_ids.shape[1],
                hidden_size,
            )  # [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name)
        )

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {}
        if self.encoder_runtime_mapping.is_first_pp_rank():
            inputs["input_ids"] = input_ids.contiguous()
            if self.encoder_model_config.has_position_embedding:
                if position_ids is None:
                    if self.encoder_model_config.remove_input_padding:
                        position_ids = [
                            torch.arange(
                                sample_length,
                                dtype=torch.int32,
                                device=input_ids.device,
                            )
                            for sample_length in torch_to_numpy(input_lengths)
                        ]
                        position_ids = torch.cat(position_ids)
                    else:
                        bsz, seq_len = input_ids.shape[:2]
                        position_ids = torch.arange(
                            seq_len, dtype=torch.int32, device=input_ids.device
                        ).expand(bsz, -1)
                inputs["position_ids"] = position_ids.contiguous()
            if self.encoder_model_config.has_token_type_embedding:
                inputs["token_type_ids"] = token_type_ids.contiguous()

            if self.encoder_model_config.max_prompt_embedding_table_size > 0:
                inputs["prompt_embedding_table"] = prompt_embedding_table.contiguous()
                inputs["tasks"] = prompt_tasks.contiguous()
                inputs["prompt_vocab_size"] = prompt_vocab_size.contiguous()
        else:
            # just need a placeholder, engine will call NCCL to recv and fill data from previous rank
            inputs["hidden_states_input"] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype("hidden_states_input"),
                device=self.device,
            ).contiguous()
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.contiguous()

        inputs["input_lengths"] = input_lengths
        # use shape info to pass max length info in remove padding mode
        inputs["max_input_length"] = torch.empty(
            (max_input_length,),
            dtype=hidden_states_dtype("max_input_length"),
            device=self.device,
        ).contiguous()

        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)

        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {}
        if self.encoder_runtime_mapping.is_last_pp_rank():
            outputs["encoder_output"] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype("encoder_output"),
                device=self.device,
            ).contiguous()
        else:
            outputs["hidden_states_output"] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype("hidden_states_output"),
                device=self.device,
            ).contiguous()

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if (
                    engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                    and name not in outputs.keys()
                ):
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(
                        tuple(shape),
                        dtype=trt_dtype_to_torch(dtype),
                        device=self.device,
                    )
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        ok = self.encoder_session.run(inputs, outputs, self.stream)

        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

        # Tensor Parallelism is handled by model/engine definition
        # But we need to broadcast among PP group at the end of encoder's Pipeline Parallelism
        # After this, all ranks should recv the encoder output, and world might be re-configured using decoder's TP-PP config
        def pp_communicate_encoder_output(encoder_output):
            if self.encoder_runtime_mapping.is_last_pp_rank():
                for pp_rank in self.encoder_runtime_mapping.pp_group:
                    if pp_rank != self.encoder_runtime_mapping.rank:
                        self.nccl_comm.send(encoder_output, pp_rank)
                return encoder_output
            else:
                self.nccl_comm.recv(
                    encoder_output, self.encoder_runtime_mapping.pp_group[-1]
                )
                return encoder_output

        if self.encoder_runtime_mapping.has_pp():
            # use hidden_states output buffer to receive output as the shapes are same
            encoder_output_buf = (
                outputs["encoder_output"]
                if self.encoder_runtime_mapping.is_last_pp_rank()
                else outputs["hidden_states_output"]
            )
            encoder_output = pp_communicate_encoder_output(encoder_output_buf)
        else:
            encoder_output = outputs["encoder_output"]

        # -------------------------------------------
        if (
            debug_mode and self.encoder_runtime_mapping.tp_rank == 0
        ):  # only tp_rank 0 print encoder output
            torch.cuda.synchronize()
            # use print_tensor() to print the tensors registered in the encoder network
            print("--------------------------------------")
            print("Debug output for Encoder")
            print("--------------------------------------")
            print("Registered output tensors are: ", outputs.keys())
            for k, v in outputs.items():
                print_tensor(k, v, num_elements=30)
            print_tensor("encoder_output", encoder_output)
            print("--------------------------------------")
        # -------------------------------------------

        return encoder_output

    def generate(
        self,
        encoder_input_ids,
        decoder_input_ids,
        max_new_tokens,
        num_beams=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        debug_mode=False,
        return_dict=False,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        attention_mask=None,
    ):
        ## ensure all externally provided tensors are on the correct device.
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = torch.tensor(
                attention_mask, dtype=torch.int32, device=self.device
            )

        ## encoder run
        encoder_remove_input_padding = (
            self.encoder_model_config.remove_input_padding
            if self.encoder_model_config
            else self.decoder_model_config.remove_input_padding
        )

        (
            encoder_input_ids,
            encoder_input_lengths,
            encoder_max_input_length,
            prompt_tasks,
        ) = self.process_input(
            encoder_input_ids, encoder_remove_input_padding, pad_token_id, prompt_tasks
        )

        if not self.skip_encoder:
            logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
            encoder_output = self.encoder_run(
                encoder_input_ids,
                encoder_input_lengths,
                encoder_max_input_length,
                debug_mode=debug_mode,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
                attention_mask=attention_mask,
            )
        else:
            encoder_output = prompt_embedding_table
            if encoder_input_ids.dim() > 1:
                encoder_output = encoder_output.unsqueeze(0)

        ## decoder run
        logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        (
            decoder_input_ids,
            decoder_input_lengths,
            decoder_max_input_length,
            _,
        ) = self.process_input(
            decoder_input_ids,
            self.decoder_model_config.remove_input_padding,
            pad_token_id,
        )

        # `cross_attention_mask` in context phase [batch_size, query_len, encoder_input_len]
        # where query_len happens to be 1 in current cases, but not necessarily always, and
        # `cross_attention_mask` in generation phase [batch_size, 1, encoder_input_len] where
        # the query_len is always 1 since we have kv cache.
        cross_attention_mask = None
        if attention_mask is not None:
            cross_attention_mask = torch.tensor(
                attention_mask, dtype=torch.int32, device=self.device
            ).reshape(attention_mask.shape[0], 1, attention_mask.shape[1])

        # generation config
        sampling_config = SamplingConfig(
            end_id=eos_token_id, pad_id=pad_token_id, num_beams=num_beams, min_length=1
        )

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=encoder_max_input_length,
        )
        torch.cuda.synchronize()

        output_ids = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
            return_dict=return_dict,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        return output_ids
