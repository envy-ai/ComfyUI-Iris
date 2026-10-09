import json
from pathlib import Path
import subprocess
import sys


def test_mixed_quantizer_only_selects_aligned_attention_weights():
    script = Path(__file__).parents[1] / "scripts" / "quantize.py"
    probe = r"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("iris_quantize", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = {
    "aligned_attention": ("blocks.0.attn.q_proj_x.weight", {"dtype": "F16", "shape": [3072, 3072]}),
    "aligned_single_stream_attention": ("blocks.0.q_proj.weight", {"dtype": "BF16", "shape": [4096, 2560]}),
    "unaligned_output_width": ("blocks.0.attn.q_proj_x.weight", {"dtype": "F16", "shape": [6826, 3072]}),
    "unaligned_input_width": ("blocks.0.attn.q_proj_x.weight", {"dtype": "F16", "shape": [3072, 3000]}),
    "mlp_w1": ("blocks.0.mlp_x.w1.weight", {"dtype": "F16", "shape": [6826, 3072]}),
    "mlp_w3": ("blocks.0.mlp_x.w3.weight", {"dtype": "F16", "shape": [6826, 3072]}),
    "non_float_attention": ("blocks.0.attn.q_proj_x.weight", {"dtype": "I8", "shape": [3072, 3072]}),
}
print(json.dumps({name: module.is_target(*case) for name, case in cases.items()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(result.stdout) == {
        "aligned_attention": True,
        "aligned_single_stream_attention": True,
        "unaligned_output_width": False,
        "unaligned_input_width": False,
        "mlp_w1": False,
        "mlp_w3": False,
        "non_float_attention": False,
    }
