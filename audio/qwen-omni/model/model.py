from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import io
import soundfile as sf
import base64

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.processor = None

    def load(self):
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="bfloat16",
            device_map="auto",
            attn_implementation="eager",
        ).eval()

        print(f"loaded model to {self.model.device}")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    def predict(self, model_input):
        # Set whether to use audio in video
        USE_AUDIO_IN_VIDEO = True
        conversation = model_input["messages"]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Inference: Generation of the output text and audio
        text_ids, audio = self.model.generate(
            **inputs,
            speaker=model_input.get("speaker", "Chelsie"),  # Chelsie, Aiden, Ethan
            thinker_return_dict_in_generate=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        text = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        buf = io.BytesIO()
        sf.write(
            buf,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
            format="WAV",
        )
        wav_bytes = buf.getvalue()

        # Base64-encode
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return {"text": text, "audio": b64}
