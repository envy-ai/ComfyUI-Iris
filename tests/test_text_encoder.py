import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

if "comfyui_iris_tests" not in sys.modules:
    pack = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_iris_tests", pack / "__init__.py", submodule_search_locations=[str(pack)]
    )
    iris = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = iris
    spec.loader.exec_module(iris)

from comfyui_iris_tests import text_encoder


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == text_encoder.PREFIX:
            return [10, 11]
        if text == text_encoder.SUFFIX:
            return [90, 91]
        if text == "short":
            return [20]
        return list(range(400))


def test_tokenizer_preserves_template_suffix_and_pads_fixed_length(monkeypatch):
    def initialize_parent(self, embedding_directory=None, tokenizer_data=None, model_type=None):
        self.qwen3vl_4b = SimpleNamespace(tokenizer=TinyTokenizer())

    monkeypatch.setattr(text_encoder.Qwen3VLTokenizer, "__init__", initialize_parent)
    tokenizer = text_encoder.IrisTokenizer()

    short = tokenizer.tokenize_with_weights("short")["qwen3vl_4b"][0]
    short_ids = [token[0] for token in short]
    assert len(short_ids) == 302
    assert short_ids[:5] == [10, 11, 20, 90, 91]
    assert set(short_ids[5:]) == {151643}

    long = tokenizer.tokenize_with_weights("long", return_word_ids=True)["qwen3vl_4b"][0]
    long_ids = [token[0] for token in long]
    assert len(long_ids) == 302
    assert long_ids[:3] == [10, 11, 0]
    assert long_ids[-2:] == [90, 91]
    assert all(token[2] == index for index, token in enumerate(long))
