import argparse
import json
import os
from typing import List, Tuple

import tensorrt as trt
import torch
from transformers import AutoConfig, AutoTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ModelConfig, SamplingConfig, Session,
                                  TensorInfo)
import requests
from PIL import Image

def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class QwenModel:
    def __init__(
        self,
        tokenizer_dir: str,
        qwen_engine_dir: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            trust_remote_code=True,
        )
        
        with open(os.path.join(qwen_engine_dir, 'config.json'), 'r') as f:
            engine_config = json.load(f)
        with open(os.path.join(tokenizer_dir, 'generation_config.json'), 'r') as f:
            generation_config = json.load(f)
        
        top_k = generation_config['top_k']
        top_p = generation_config['top_p']
        chat_format = generation_config['chat_format']
        
        if chat_format == "raw":
            eos_token_id = engine_config['eos_token_id']
            pad_token_id = engine_config['pad_token_id']
        elif chat_format == "chatml":
            pad_token_id = eos_token_id = self.tokenizer.im_end_id
        else:
            raise Exception("unknown chat format ", chat_format)

        use_gpt_attention_plugin = engine_config['plugin_config'][
            'gpt_attention_plugin']
        remove_input_padding = engine_config['plugin_config']['remove_input_padding']
        dtype = engine_config['builder_config']['precision']
        tp_size = engine_config['builder_config']['tensor_parallel']
        pp_size = engine_config['builder_config']['pipeline_parallel']
        world_size = tp_size * pp_size
        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        num_heads = engine_config['builder_config']['num_heads'] // world_size
        max_batch_size = engine_config['builder_config']['max_batch_size']
        hidden_size = engine_config['builder_config']['hidden_size'] // world_size
        vocab_size = engine_config['builder_config']['vocab_size']
        num_layers = engine_config['builder_config']['num_layers']
        num_kv_heads = engine_config['builder_config'].get('num_kv_heads', num_heads)
        paged_kv_cache = engine_config['plugin_config']['paged_kv_cache']
        tokens_per_block = engine_config['plugin_config']['tokens_per_block']
        max_prompt_embedding_table_size = engine_config['builder_config'].get(
            'max_prompt_embedding_table_size', 0)
        quant_mode = QuantMode(engine_config['builder_config']['quant_mode'])
        if engine_config['builder_config'].get('multi_query_mode', False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1
        # num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
        use_custom_all_reduce = engine_config['plugin_config'].get(
            'use_custom_all_reduce', False)

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=runtime_rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        model_config = ModelConfig(
            max_batch_size=max_batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            remove_input_padding=remove_input_padding,
            dtype=dtype,
            quant_mode=quant_mode,
            use_custom_all_reduce=use_custom_all_reduce,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )
        sampling_config = SamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=1,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )
        
        engine_name = get_engine_name('qwen', dtype, tp_size, pp_size,
                                      runtime_rank)
        serialize_path = os.path.join(qwen_engine_dir, engine_name)
        print(f'Loading engine from {serialize_path}')

        self.model_config = model_config
        self.sampling_config = sampling_config
        self.runtime_mapping = runtime_mapping
        self.runtime_rank = runtime_rank
        self.serialize_path = serialize_path
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
    
    def load(self):
        with open(self.serialize_path, 'rb') as f:
            engine_buffer = f.read() 
        self.decoder = tensorrt_llm.runtime.GenerationSession(
            self.model_config,
            engine_buffer,
            self.runtime_mapping,
        )
        self.config, _ = AutoConfig.from_pretrained(
            self.tokenizer_dir,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )
    
    
    def ptuning_setup(self, prompt_table, dtype, hidden_size, tasks, input_ids):
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                           dtype=torch.int32,
                                           device="cuda")
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))
            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(',')],
                                 dtype=torch.int32,
                                 device="cuda")
            assert tasks.shape[0] == input_ids.shape[
                0], "Number of supplied tasks must match input batch size"
        else:
            tasks = torch.zeros([input_ids.size(0)], dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]


    def make_context(
        self,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "You are a helpful assistant.",
        max_window_size: int = 6144,
    ):
        if history is None:
            history = []

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [self.tokenizer.im_start_id]  # 151644
        im_end_tokens = [self.tokenizer.im_end_id]  # [151645]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role, allowed_special=set(self.tokenizer.IMAGE_ST)
            ) + nl_tokens + self.tokenizer.encode(
                content, allowed_special=set(self.tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (len(system_tokens) +
                                    len(next_context_tokens) +
                                    len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (nl_tokens + im_start_tokens +
                           _tokenize_str("user", query)[1] + im_end_tokens +
                           nl_tokens + im_start_tokens +
                           self.tokenizer.encode("assistant") + nl_tokens)
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text, context_tokens
    
    def predict(self, image_embeds, image_path, input_text, max_new_tokens, history = []):
        if image_path is None:
            content_list = []
        else:
            content_list = image_path
        
        if history is None:
            history = []
        content_list.append({'text': input_text})
        query = self.tokenizer.from_list_format(content_list)
        raw_text, context_tokens = self.make_context(query, history=history)
        # context_tokens = self.tokenizer.encode(query)
        input_ids = torch.tensor([context_tokens]).to('cuda')
        bos_pos = torch.where(input_ids == self.config.visual['image_start_id'])
        eos_pos = torch.where(
            input_ids == self.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        vocab_size = self.config.vocab_size
        fake_prompt_id = torch.arange(vocab_size,
                                      vocab_size +
                                      image_embeds.shape[0] * image_embeds.shape[1],
                                      device='cuda')
        fake_prompt_id = fake_prompt_id.reshape(image_embeds.shape[0],
                                                image_embeds.shape[1])
        for idx, (i, a, b) in enumerate(img_pos):
            input_ids[i][a + 1:b] = fake_prompt_id[idx]
        input_ids = input_ids.contiguous().to(torch.int32).cuda()
        input_lengths = torch.tensor(input_ids.size(1),
                                     dtype=torch.int32).cuda()
        dtype = self.model_config.dtype
        prompt_table, tasks, task_vocab_size = self.ptuning_setup(
            image_embeds, dtype, self.config.hidden_size, None, input_ids)
        input_tokens = input_ids
        input_ids = None
        input_lengths = None
        input_ids = torch.as_tensor(input_tokens,
                                    device="cuda",
                                    dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(max_new_tokens,
                             self.global_max_input_len - max_input_length)
        self.decoder.setup(batch_size=input_lengths.size(0),
                           max_context_length=max_input_length,
                           max_new_tokens=max_new_tokens)
        
        output_ids = self.decoder.decode(input_ids, input_lengths,
                                             self.sampling_config, prompt_table,
                                             tasks, task_vocab_size)
        torch.cuda.synchronize()
        runtime_rank = tensorrt_llm.mpi_rank()
        input_lengths = torch.tensor([input_tokens.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)

        input_len = input_tokens[0]
        outputs = output_ids[b][0, len(input_len):].tolist()
        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        return output_text

class VitPreprocess:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith(
                    "https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return images


class Vit:
    def __init__(self, vit_engine_dir: str):
        with open(os.path.join(vit_engine_dir, 'visual_encoder/visual_encoder_fp16.plan'), 'r') as f:
            engine_buffer = f.read()
        self.vit_engine = Session.from_serialized_engine(
            engine_buffer
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.preprocess = VitPreprocess(image_size=448)

    def predict(self, image_path, stream):
        image = self.preprocess.encode(image_path).to(self.device)        
        images = torch.cat([image])
        batch_size = images.size(0)
        images = images.expand(batch_size, -1, -1, -1).contiguous()
        visual_inputs = {'input': images.float()}
        visual_output_info = self.vit_engine.infer_shapes(
            [TensorInfo('input', trt.DataType.FLOAT, images.shape)])
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
            for t in visual_output_info
        }
        ok = self.session_vit.run(visual_inputs, visual_outputs, stream)
        return visual_outputs['output']
    
class Model:
    def __init__(self, tokenizer_dir: str, qwen_engine_dir: str, vit_engine_dir: str):
        self.qwen = QwenModel(tokenizer_dir, qwen_engine_dir)
        self.qwen.load()
        self.vit = Vit(vit_engine_dir)
    
    def predict(self, image_path, query, history):
        image_embeds = self.vit.predict(image_path)
        response = self.qwen.predict(image_embeds, image_path, query, 100, history)
        return response


if __name__ == '__main__':
    model = Model(tokenizer_dir='./Qwen-VL-Chat/', qwen_engine_dir='./trt_engines/Qwen-VL-7B-Chat-int4-gptq/', vit_engine_dir='./plan/')
    # data = {'image': ['./pics/demo.jpeg'], 'prompt' : 'what is in this photo?'}
    # output = model.predict(data)
    output = model.predict(image_path='./pics/demo.jpeg', query='what is in this photo?', history=[])
    print(output)