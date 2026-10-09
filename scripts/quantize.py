#!/usr/bin/env python3
"""Stream the original Iris checkpoint to FP16 or mixed INT8 ConvRot on CPU."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from safetensors import safe_open

from comfy.cli_args import args as comfy_args
comfy_args.cpu = True
from comfy.quant_ops import QuantizedTensor


DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
               "I32": 4, "U32": 4, "I64": 8, "U64": 8,
               "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
QUANT_CONFIG = json.dumps({"format": "int8_tensorwise", "convrot": True,
                           "convrot_groupsize": 256}, separators=(",", ":")).encode()
POLICY = {
    "int8": "trunk attention Q/K/V/output projections, input width divisible by 256 and output width divisible by 4",
    "fp16": "all other floating tensors, including content gates, all MLPs, text adapter, modulation, timestep, norms and pixel head",
    "scale": "float32 per output channel",
    "convrot_groupsize": 256,
    "calibrated": False,
}


def read_header(path):
    with open(path, "rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        return json.loads(source.read(length))


def is_target(name, spec):
    if not name.startswith("blocks.") or not name.endswith(".weight"):
        return False
    if spec["dtype"] not in FLOAT_DTYPES or len(spec["shape"]) != 2:
        return False
    parts = name.split(".")
    projection = parts[-2]
    attention = ((parts[2] == "attn" and projection in {
        "q_proj_x", "k_proj_x", "v_proj_x", "q_proj_y", "k_proj_y", "v_proj_y", "proj_x", "proj_y"
    }) or (parts[2] in {"q_proj", "k_proj", "v_proj", "attn_proj"}))
    return attention and spec["shape"][0] % 4 == 0 and spec["shape"][1] % 256 == 0


def tensor_bytes(spec):
    return DTYPE_BYTES[spec["dtype"]] * math.prod(spec["shape"])


def build_header(source, mode):
    tensors = {}
    targets = []
    fp16_parameters = 0
    int8_parameters = 0
    for name, spec in source.items():
        if name == "__metadata__":
            continue
        shape = spec["shape"]
        if mode == "int8-convrot" and is_target(name, spec):
            targets.append(name)
            int8_parameters += math.prod(shape)
            tensors[name] = {"dtype": "I8", "shape": shape}
            tensors[name + "_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
            tensors[name[:-len("weight")] + "comfy_quant"] = {"dtype": "U8", "shape": [len(QUANT_CONFIG)]}
        else:
            dtype = "F16" if spec["dtype"] in FLOAT_DTYPES else spec["dtype"]
            tensors[name] = {"dtype": dtype, "shape": shape}
            if dtype == "F16":
                fp16_parameters += math.prod(shape)
    # Larger dtypes first keep every numerical tensor's payload aligned.
    result = {}
    offset = 0
    for name in sorted(tensors, key=lambda key: -DTYPE_BYTES[tensors[key]["dtype"]]):
        spec = tensors[name]
        length = tensor_bytes(spec)
        result[name] = {**spec, "data_offsets": [offset, offset + length]}
        offset += length
    summary = {"format": mode, "quantized_tensors": len(targets),
               "int8_parameters": int8_parameters, "fp16_parameters": fp16_parameters,
               "payload_bytes": offset, "quantized_weights": targets,
               "policy": POLICY if mode == "int8-convrot" else "all floating tensors stored as FP16"}
    return result, summary


def write_tensor(output, data_start, spec, tensor, name):
    if list(tensor.shape) != spec["shape"] or tensor.element_size() * tensor.numel() != tensor_bytes(spec):
        raise ValueError(f"Serialization shape or size mismatch for {name}")
    output.seek(data_start + spec["data_offsets"][0])
    output.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=["fp16", "int8-convrot"], required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output_partial = args.output.with_name(args.output.name + ".partial")
    report_path = args.output.with_suffix(".json")
    report_partial = report_path.with_name(report_path.name + ".partial")
    for path in (args.output, output_partial, report_path, report_partial):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    source_header = read_header(args.input)
    output_header, summary = build_header(source_header, args.format)
    summary["source"] = str(args.input.resolve())
    summary["output"] = str(args.output.resolve())
    summary["threads"] = args.threads
    metadata = dict(source_header.get("__metadata__", {}))
    metadata["iris_conversion"] = json.dumps({key: value for key, value in summary.items() if key != "quantized_weights"}, separators=(",", ":"))
    output_header["__metadata__"] = metadata
    header_bytes = json.dumps(output_header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    data_start = 8 + len(header_bytes)
    summary["file_bytes"] = data_start + summary["payload_bytes"]
    print(f"{args.format}: {summary['quantized_tensors']} INT8 tensors, {summary['file_bytes'] / 2**30:.3f} GiB; CPU threads={args.threads}", flush=True)
    targets = set(summary["quantized_weights"])
    with safe_open(args.input, framework="pt", device="cpu") as source, open(output_partial, "xb") as output:
        output.write(struct.pack("<Q", len(header_bytes)))
        output.write(header_bytes)
        output.truncate(summary["file_bytes"])
        for index, name in enumerate(source.keys(), 1):
            weight = source.get_tensor(name)
            if weight.is_floating_point() and not torch.isfinite(weight).all():
                raise ValueError(f"Source tensor contains non-finite values: {name}")
            if name in targets:
                quantized = QuantizedTensor.from_float(weight, "TensorWiseINT8Layout", scale="recalculate", per_channel=True, convrot=True, convrot_groupsize=256)
                state = quantized.state_dict(name)
                write_tensor(output, data_start, output_header[name], state[name], name)
                scales = state[name + "_scale"]
                if not torch.isfinite(scales).all():
                    raise ValueError(f"INT8 scales contain non-finite values: {name}")
                write_tensor(output, data_start, output_header[name + "_scale"], scales, name + "_scale")
                quant_name = name[:-len("weight")] + "comfy_quant"
                output.seek(data_start + output_header[quant_name]["data_offsets"][0])
                output.write(QUANT_CONFIG)
                del state, scales, quantized
            else:
                converted = weight.to(torch.float16) if weight.is_floating_point() else weight
                if converted.is_floating_point() and not torch.isfinite(converted).all():
                    raise ValueError(f"FP16 conversion produced non-finite values: {name}")
                write_tensor(output, data_start, output_header[name], converted, name)
                del converted
            del weight
            gc.collect()
            print(f"[{index}/{len(source.keys())}] {name}", flush=True)
        output.flush()
        os.fsync(output.fileno())
    with open(report_partial, "x") as report:
        json.dump(summary, report, indent=2)
        report.write("\n")
        report.flush()
        os.fsync(report.fileno())
    # Hard links commit completed files atomically without replacing any output.
    os.link(output_partial, args.output)
    output_partial.unlink()
    os.link(report_partial, report_path)
    report_partial.unlink()
    print(f"Completed: {args.output}; report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
