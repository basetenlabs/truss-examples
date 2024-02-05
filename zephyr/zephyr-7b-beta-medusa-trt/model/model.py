import copy
from threading import Thread

import torch
from huggingface_hub import snapshot_download
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer, GenerationConfig, TextIteratorStreamer, pipeline

MEDUSA_CHOICES = [
    (0,),
    (0, 0),
    (1,),
    (0, 1),
    (2,),
    (0, 0, 0),
    (1, 0),
    (0, 2),
    (3,),
    (0, 3),
    (4,),
    (2, 0),
    (0, 0, 1),
    (0, 4),
    (5,),
    (0, 5),
    (0, 1, 0),
    (1, 1),
    (6,),
    (0, 0, 2),
    (3, 0),
    (0, 6),
    (7,),
    (0, 7),
    (0, 8),
    (0, 0, 3),
    (1, 0, 0),
    (0, 9),
    (0, 2, 0),
    (1, 2),
    (4, 0),
    (8,),
    (9,),
    (2, 1),
    (0, 1, 1),
    (0, 0, 4),
    (0, 0, 0, 0),
    (5, 0),
    (0, 3, 0),
    (1, 3),
    (0, 0, 5),
    (0, 0, 6),
    (6, 0),
    (2, 0, 0),
    (1, 0, 1),
    (0, 1, 2),
    (0, 4, 0),
    (1, 4),
    (3, 1),
    (2, 2),
    (0, 0, 7),
    (7, 0),
    (0, 2, 1),
    (0, 0, 8),
    (0, 1, 3),
    (0, 5, 0),
    (1, 5),
    (0, 0, 9),
    (1, 1, 0),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (4, 1),
    (2, 3),
]


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._tokenizer = None
        self._model = None
        self._data_dir = kwargs["data_dir"]

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta", use_fast=True
        )
        snapshot_download(
            repo_id="baseten/zephyr-7b-beta-medusa-fp16", local_dir="/app/data"
        )
        self._model = ModelRunner.from_dir(
            str(self._data_dir), medusa_choices=MEDUSA_CHOICES
        )

    def predict(self, request: dict):
        prompt = request["prompt"]
        is_streaming = request.get("streaming", True)
        input_ids = torch.tensor(self._tokenizer.encode(prompt), dtype=torch.int32).to(
            "cuda"
        )
        outputs = self._model.generate(
            [input_ids],
            end_id=self._tokenizer.eos_token_id,
            pad_id=self._tokenizer.pad_token_id,
            temperature=0.0,
            max_new_tokens=request.get("max_new_tokens", 512),
            max_batch_size=1,
            max_input_len=len(input_ids),
            medusa_choices=MEDUSA_CHOICES,
            streaming=is_streaming,
            return_dict=True,
            output_sequence_lengths=True,
        )
        torch.cuda.synchronize()
        if is_streaming:
            prev_decoded = None
            for output, last in _with_last_flag(outputs):
                output_ids = output["output_ids"]
                seq_lens = output["sequence_lengths"]
                # HACK(pankaj) There's a bug in TensorRT-LLM python runtime where
                # sequence lengths are not returned correctly for medusa. This is
                # a temporary workaround. The workaround is pretty painful, as we
                # need to substract len(MEDUSA_CHOICES) for everything but the last
                # iteration.
                slen = seq_lens[0][0]
                if not last:
                    slen = slen - len(MEDUSA_CHOICES)
                decoded = self._tokenizer.decode(
                    output_ids[0][0][len(input_ids) : slen]
                )
                if prev_decoded is None:
                    ret = decoded
                else:
                    ret = decoded[len(prev_decoded) :]
                prev_decoded = decoded
                yield ret
        else:
            output_ids = outputs["output_ids"][0][0]
            seq_len = outputs["sequence_lengths"][0][0]
            decoded = self._tokenizer.decode(output_ids[len(input_ids) : seq_len])
            return decoded


def _with_last_flag(generator):
    """Yield items from a generator along with a flag indicating if it's the last item."""
    # copy is important here because the underlying tensors are overwritten by
    # the generation loop.
    prev_item = copy.deepcopy(next(generator))
    for item in generator:
        yield prev_item, False
        prev_item = copy.deepcopy(item)
    yield prev_item, True
