import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_trt import WhisperModel

from whisper_trt.types import BatchWhisperItem

from async_batcher.batcher import AsyncBatcher
import torch


class WhisperBatchProcessor(AsyncBatcher[list[BatchWhisperItem], list[str]]):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model: "WhisperModel" = model
        self.lang_id_tensor = torch.tensor(self._model._sot_id).unsqueeze(0)

    async def process_batch(self, batch: list[BatchWhisperItem]) -> list[list[int]]:
        max_new_tokens = max(item.max_new_tokens for item in batch)
        batch_size = len(batch)

        logging.info(
            f"Processing batch with `{batch_size}` elements, "
            f"max_new_tokens={max_new_tokens}"
        )

        mel_batch = torch.cat([item.mel for item in batch], dim=0).type(torch.float16)

        encoder_output = self._model._encoder.get_audio_features(mel_batch)

        # lang_audio_output_ids = self._model._decoder.detect_language(
        #     encoder_outputs=encoder_output,
        #     num_beams=self._model._num_beams,
        # )

        # detected_language_tokens = self._model._tokenizer.decode_batch(
        #     lang_audio_output_ids.tolist()
        # )[: len(batch)]
        detected_language_tokens = ["<|en|>"] * len(batch)
        prompts = [
            self._model._get_text_prefix(
                lang,
                language=item.language,
                prompt=item.prompt,
                task=item.task,
                prefix=item.prefix,
            )
            for lang, item in zip(detected_language_tokens, batch)
        ]

        # From this point forward, we only want to process the prompts that
        # returned non-None values.
        valid_indices, valid_prompts = [], []
        valid_durations = []
        for i, prompt in enumerate(prompts):
            if prompt is not None:
                valid_indices.append(i)
                valid_prompts.append(prompt)
                valid_durations.append(batch[i].duration_secs)
        if len(valid_indices) > 0:
            valid_token_ids = self._model._tokenizer.encode_batch(
                valid_prompts, allowed_special=self._model._tokenizer.special_tokens_set
            )
            max_length = max(len(tokens) for tokens in valid_token_ids)

            padded_token_ids = [
                [50257] * (max_length - len(tokens)) + tokens
                for tokens in valid_token_ids
            ]

            # padded_token_ids = [
            #     [self._model._pad_id] * (max_length - len(tokens)) + tokens
            #     for tokens in valid_token_ids
            # ]
            valid_token_ids = torch.tensor(padded_token_ids)

            output_ids, avg_log_probs = self._model._decoder.generate(
                valid_token_ids,
                encoder_outputs=torch.index_select(
                    encoder_output, 0, torch.tensor(valid_indices).cuda()
                ),
                max_new_tokens=max_new_tokens,
                num_beams=self._model._num_beams,
                duration_secs=valid_durations,
                eot_id=self._model._eot_id,
                use_timstamps_processor=True,
                notimestamps_id=self._model._notimestamps_id,
            )
            decoded_tokens = self._model._tokenizer.decode_batch(output_ids.tolist())

        batch_result = [(None, None)] * batch_size
        for i, idx in enumerate(valid_indices):
            # Make sure to only return as many tokens as the original request asked for
            batch_result[idx] = decoded_tokens[i], avg_log_probs[i]

        return batch_result
