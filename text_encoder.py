"""Native Qwen3-VL language conditioning with the Iris template and layer taps."""
import torch

from comfy import sd1_clip
import comfy.model_management
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer
from comfy.text_encoders.llama import BaseLlama, Llama2_, Qwen3VL_4BConfig

HIDDEN_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_LENGTH = 300


class IrisTokenizer(Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        tokenizer = self.qwen3vl_4b.tokenizer
        self.prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids = self.qwen3vl_4b.tokenizer.encode(text, add_special_tokens=False)
        budget = TEXT_LENGTH - len(self.suffix_ids)
        ids = self.prefix_ids + ids[:budget] + self.suffix_ids
        ids += [151643] * (len(self.prefix_ids) + TEXT_LENGTH - len(ids))
        pairs = [(token, 1.0, i) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]
        return {"qwen3vl_4b": [pairs]}


class IrisLanguageModel(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3VL_4BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype


class IrisQwenClip(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, model_options=model_options, layer="hidden", layer_idx=-1, textmodel_json_config={}, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=IrisLanguageModel, enable_attention_masks=True, return_attention_masks=True)

    def encode_token_weights(self, token_weight_pairs):
        self.set_clip_options({"layer": HIDDEN_LAYERS})
        # Native taps at block i's input equal HF hidden_states[i].
        states, _, extra = self.encode([[token[0] for token in row] for row in token_weight_pairs])
        states = states.transpose(1, 2)
        start = states.shape[1] - TEXT_LENGTH
        states = states[:, start:]
        mask = extra["attention_mask"][:, start:]
        states = states * mask[:, :, None, None]
        device = comfy.model_management.intermediate_device()
        return states.to(device), None, {"attention_mask": mask.to(device)}


class IrisTE(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=IrisQwenClip, model_options=model_options)

    def load_sd(self, sd):
        result = super().load_sd(sd)
        missing, unexpected = result
        missing = [key for key in missing if key.startswith("model.")]
        if missing or unexpected:
            raise ValueError(f"Invalid Qwen3-VL-4B text weights: missing={missing}, unexpected={unexpected}")
        return result
