"""The frozen architecture whitelist behind "evaluation only trusts state_dict".

The evaluator may not execute candidate code, and `torch.load` of a full
pickled model executes code at unpickle time. The interface between training
and evaluation is therefore two files:

* `weights.pt` — a plain state_dict, loadable with `torch.load(weights_only=True)`;
* `architecture.json` — a serialized architecture produced HERE, by tracing
  the trained model with `torch.fx` and refusing anything outside a frozen
  vocabulary of module classes, function targets and tensor methods.

`rebuild_model` reconstructs a callable model from that JSON with frozen code
only (torch's own whitelisted classes plus a `torch.fx.GraphModule` compiled
from the recorded graph), then the caller loads the state_dict strictly.
No byte of the candidate's file is imported or unpickled on the evaluation
side: a candidate's `forward` runs as recorded torch ops, its `__init__` and
module-level code never run outside the training subprocess at all.

Serialization is deliberately lossy about code and strict about semantics:
an architecture that cannot be expressed in the frozen vocabulary refuses to
export (`ArchitectureExportRefused`), which fails the training round honestly
instead of widening the evaluation side.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import operator
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from core.batch import ModelOutput

ARCHITECTURE_FILE = "architecture.json"
EVAL_WEIGHTS_FILE = "eval-weights.pt"
ARCH_VERSION = "research-arch-v1"
TASK_CLASS_COUNTS = {"constitution": 9, "syndrome": 148, "toy": 3}
ELEMENTS_PER_CLASS_LIMIT = 65_536
TASK_ELEMENT_LIMITS = {
    task: classes * ELEMENTS_PER_CLASS_LIMIT
    for task, classes in TASK_CLASS_COUNTS.items()
}
MAX_ARCHITECTURE_BYTES = 1024 * 1024
MAX_WEIGHTS_BYTES = 64 * 1024 * 1024
MAX_ARCH_MODULES = 256
MAX_GRAPH_NODES = 2048
MAX_ARCH_DEPTH = 32
MAX_ARCH_DIM = 65_536
MAX_TENSOR_RANK = 8


class ArchitectureExportRefused(ValueError):
    """The model uses something outside the frozen evaluation vocabulary."""


def _fn_key(target) -> str:
    keys = {
        torch.cat: "torch.cat",
        torch.stack: "torch.stack",
        torch.sigmoid: "torch.sigmoid",
        torch.relu: "torch.relu",
        torch.softmax: "torch.softmax",
        torch.tanh: "torch.tanh",
        torch.flatten: "torch.flatten",
        F.relu: "F.relu",
        F.sigmoid: "F.sigmoid",
        F.softmax: "F.softmax",
        F.tanh: "F.tanh",
        F.gelu: "F.gelu",
        getattr: "getattr",
        len: "len",
        operator.getitem: "operator.getitem",
        operator.add: "operator.add",
        operator.sub: "operator.sub",
        operator.mul: "operator.mul",
        operator.truediv: "operator.truediv",
        operator.neg: "operator.neg",
        operator.lt: "operator.lt",
        operator.le: "operator.le",
        operator.gt: "operator.gt",
        operator.ge: "operator.ge",
        operator.eq: "operator.eq",
        operator.ne: "operator.ne",
        ModelOutput: "core.batch.ModelOutput",
    }
    return keys.get(target, "")


FUNCTION_TARGETS = {
    "torch.cat": torch.cat,
    "torch.stack": torch.stack,
    "torch.sigmoid": torch.sigmoid,
    "torch.relu": torch.relu,
    "torch.softmax": torch.softmax,
    "torch.tanh": torch.tanh,
    "torch.flatten": torch.flatten,
    "F.relu": F.relu,
    "F.sigmoid": F.sigmoid,
    "F.softmax": F.softmax,
    "F.tanh": F.tanh,
    "F.gelu": F.gelu,
    "getattr": getattr,
    "len": len,
    "operator.getitem": operator.getitem,
    "operator.add": operator.add,
    "operator.sub": operator.sub,
    "operator.mul": operator.mul,
    "operator.truediv": operator.truediv,
    "operator.neg": operator.neg,
    "operator.lt": operator.lt,
    "operator.le": operator.le,
    "operator.gt": operator.gt,
    "operator.ge": operator.ge,
    "operator.eq": operator.eq,
    "operator.ne": operator.ne,
    "core.batch.ModelOutput": ModelOutput,
}

#: Tensor methods a recorded graph may call. Methods are looked up on tensors
#: the model itself produced, so the risk is misuse, not arbitrary code — but
#: the vocabulary is still closed rather than open.
METHOD_WHITELIST = frozenset({
    "float", "double", "half", "unsqueeze", "squeeze", "view", "reshape",
    "transpose", "permute", "contiguous", "flatten", "to", "softmax",
    "sigmoid", "tanh", "relu", "gelu", "clamp", "abs", "sum", "mean",
    "max", "min", "pow", "sqrt", "log", "exp", "chunk", "split",
    "masked_fill", "round", "size", "dim", "numel", "t", "neg",
})

#: Module classes the evaluator will rebuild, with the ctor kwargs extracted
#: from a live module. Containers take no ctor args: children are re-attached
#: by path in topological order (nn.Module.named_modules is preorder).
MODULE_EXPORT: dict[type, tuple[str, object]] = {
    nn.Linear: ("nn.Linear", lambda m: {
        "in_features": m.in_features, "out_features": m.out_features,
        "bias": m.bias is not None,
    }),
    nn.GELU: ("nn.GELU", lambda m: {}),
    nn.ReLU: ("nn.ReLU", lambda m: {"inplace": m.inplace}),
    nn.Sigmoid: ("nn.Sigmoid", lambda m: {}),
    nn.Tanh: ("nn.Tanh", lambda m: {}),
    nn.SiLU: ("nn.SiLU", lambda m: {}),
    nn.LeakyReLU: ("nn.LeakyReLU", lambda m: {"negative_slope": m.negative_slope}),
    nn.Dropout: ("nn.Dropout", lambda m: {"p": m.p}),
    nn.LayerNorm: ("nn.LayerNorm", lambda m: {
        "normalized_shape": list(m.normalized_shape), "eps": m.eps,
        "elementwise_affine": m.elementwise_affine,
    }),
    nn.BatchNorm1d: ("nn.BatchNorm1d", lambda m: {
        "num_features": m.num_features, "eps": m.eps, "momentum": m.momentum,
        "affine": m.affine, "track_running_stats": m.track_running_stats,
    }),
    nn.Softmax: ("nn.Softmax", lambda m: {"dim": m.dim}),
    nn.Flatten: ("nn.Flatten", lambda m: {
        "start_dim": m.start_dim, "end_dim": m.end_dim,
    }),
    nn.Identity: ("nn.Identity", lambda m: {}),
    nn.Embedding: ("nn.Embedding", lambda m: {
        "num_embeddings": m.num_embeddings, "embedding_dim": m.embedding_dim,
    }),
    nn.Sequential: ("nn.Sequential", lambda m: {}),
    nn.ModuleList: ("nn.ModuleList", lambda m: {}),
    nn.ModuleDict: ("nn.ModuleDict", lambda m: {}),
}

MODULE_BUILDERS: dict[str, object] = {
    "nn.Linear": nn.Linear,
    "nn.GELU": nn.GELU,
    "nn.ReLU": nn.ReLU,
    "nn.Sigmoid": nn.Sigmoid,
    "nn.Tanh": nn.Tanh,
    "nn.SiLU": nn.SiLU,
    "nn.LeakyReLU": nn.LeakyReLU,
    "nn.Dropout": nn.Dropout,
    "nn.LayerNorm": nn.LayerNorm,
    "nn.BatchNorm1d": nn.BatchNorm1d,
    "nn.Softmax": nn.Softmax,
    "nn.Flatten": nn.Flatten,
    "nn.Identity": nn.Identity,
    "nn.Embedding": nn.Embedding,
    "nn.Sequential": nn.Sequential,
    "nn.ModuleList": nn.ModuleList,
    "nn.ModuleDict": nn.ModuleDict,
}


def _encode_constant(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"const": value}
    if isinstance(value, torch.dtype):
        return {"const": {"__dtype__": str(value).removeprefix("torch.")}}
    if isinstance(value, torch.device):
        return {"const": {"__device__": str(value)}}
    raise ArchitectureExportRefused(
        f"non-serializable constant in the traced graph: {type(value).__name__}"
    )


def _encode_arg(arg):
    if isinstance(arg, torch.fx.Node):
        return {"node": arg.name}
    if isinstance(arg, (tuple, list)):
        return {"seq": [_encode_arg(a) for a in arg]}
    if isinstance(arg, dict):
        return {"dict": {k: _encode_arg(v) for k, v in arg.items()}}
    return _encode_constant(arg)


def _decode_arg(arg, env):
    if isinstance(arg, dict):
        if "node" in arg:
            return env[arg["node"]]
        if "const" in arg:
            value = arg["const"]
            if isinstance(value, dict) and "__dtype__" in value:
                return getattr(torch, value["__dtype__"])
            if isinstance(value, dict) and "__device__" in value:
                return torch.device(value["__device__"])
            return value
        if "seq" in arg:
            return tuple(_decode_arg(a, env) for a in arg["seq"])
        if "dict" in arg:
            return {k: _decode_arg(v, env) for k, v in arg["dict"].items()}
    raise ArchitectureExportRefused(f"malformed serialized argument: {arg!r}")


def export_architecture(
    model: nn.Module,
    *,
    task: str = "constitution",
    expected_n_classes: int | None = None,
) -> dict:
    """Trace and bound a model in the frozen JSON vocabulary, or refuse."""
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001 — any tracing failure is a refusal
        raise ArchitectureExportRefused(
            f"model does not trace into the frozen evaluation vocabulary: {exc}"
        ) from exc

    modules: dict[str, dict] = {}
    for name, mod in model.named_modules():
        if not name:
            continue
        if len(modules) >= MAX_ARCH_MODULES:
            raise ArchitectureExportRefused("model exceeds the frozen module count limit")
        entry = MODULE_EXPORT.get(type(mod))
        if entry is None:
            raise ArchitectureExportRefused(
                f"module {name!r} of type {type(mod).__name__} is not in the "
                "frozen architecture whitelist; evaluation only trusts "
                "state_dict + a whitelisted architecture"
            )
        key, extract = entry
        modules[name] = {"class": key, "args": extract(mod)}

    # get_attr targets are attribute NAME strings; the check used to compare
    # them against the tensor objects from named_parameters/named_buffers and
    # refused every get_attr node (H-04)
    own_attrs = {name for name, _ in model.named_parameters()}
    own_attrs |= {name for name, _ in model.named_buffers()}
    graph: list[dict] = []
    for node in traced.graph.nodes:
        spec: dict = {"op": node.op, "name": node.name}
        if node.op == "placeholder":
            spec["target"] = None
        elif node.op == "call_function":
            key = _fn_key(node.target)
            if not key:
                raise ArchitectureExportRefused(
                    f"call target {getattr(node.target, '__name__', node.target)!r} "
                    "is not in the frozen function whitelist"
                )
            spec["target"] = key
        elif node.op == "call_method":
            if node.target not in METHOD_WHITELIST:
                raise ArchitectureExportRefused(
                    f"method .{node.target}() is not in the frozen method whitelist"
                )
            spec["target"] = node.target
        elif node.op == "call_module":
            if node.target not in modules:
                raise ArchitectureExportRefused(
                    f"call_module target {node.target!r} is not a known module"
                )
            spec["target"] = node.target
        elif node.op == "get_attr":
            if node.target not in own_attrs:
                raise ArchitectureExportRefused(
                    f"get_attr target {node.target!r} is not a parameter or buffer"
                )
            spec["target"] = node.target
            # the rebuild side must be able to register a placeholder for
            # this attribute (H-04): record its kind, shape and dtype so the
            # strict state_dict load has a key to fill
            value = model
            for part in node.target.split("."):
                value = getattr(value, part)
            parameters = {name for name, _ in model.named_parameters()}
            spec["attr"] = {
                "kind": "parameter" if node.target in parameters else "buffer",
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        elif node.op == "output":
            spec["target"] = None
        else:  # pragma: no cover — fx has no other node kinds today
            raise ArchitectureExportRefused(f"unsupported graph node kind {node.op!r}")
        spec["args"] = _encode_arg(node.args)
        spec["kwargs"] = _encode_arg(dict(node.kwargs))
        graph.append(spec)

    return {
        "version": ARCH_VERSION,
        "model_type": getattr(model, "model_type", type(model).__name__),
        "input_spec": model.input_spec.to_dict(),
        "modules": modules,
        "graph": graph,
    }


def _bounded_regular_bytes(path: str | Path, *, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened as a regular file") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if info.st_size > maximum:
            raise ValueError(f"{label} exceeds the {maximum}-byte limit")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(fd, min(65_536, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum:
            raise ValueError(f"{label} exceeds the {maximum}-byte limit")
        if len(payload) != info.st_size or os.read(fd, 1):
            raise ValueError(f"{label} changed while it was being read")
        return bytes(payload)
    finally:
        os.close(fd)

def _positive_int(value, label: str, *, maximum: int = MAX_ARCH_DIM) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer from 1 to {maximum}")
    return value

def _task_element_limit(task: str, expected_n_classes: int | None) -> int:
    classes = TASK_CLASS_COUNTS.get(task)
    if classes is None:
        raise ValueError(f"unknown model task {task!r}")
    if expected_n_classes is not None and expected_n_classes != classes:
        raise ValueError(
            f"{task} requires {classes} output classes, got {expected_n_classes}"
        )
    return TASK_ELEMENT_LIMITS[task]

def _shape_elements(shape, label: str, *, allow_scalar: bool = False) -> int:
    if not isinstance(shape, list) or len(shape) > MAX_TENSOR_RANK:
        raise ValueError(f"{label} tensor shape has invalid rank")
    if not shape and allow_scalar:
        return 1
    if not shape:
        raise ValueError(f"{label} tensor shape must not be scalar")
    total = 1
    for index, dim in enumerate(shape):
        total *= _positive_int(dim, f"{label} shape dimension {index}")
    return total

def _validate_input_spec(value: object):
    if not isinstance(value, dict) or set(value) != {"version", "modalities"}:
        raise ValueError("architecture input_spec has an invalid schema")
    version = value["version"]
    modalities = value["modalities"]
    if not isinstance(version, str) or not 1 <= len(version) <= 128:
        raise ValueError("architecture input_spec version is invalid")
    if not isinstance(modalities, list) or not 1 <= len(modalities) <= 16:
        raise ValueError("architecture input_spec modality count is invalid")
    for item in modalities:
        if not isinstance(item, dict) or set(item) != {
            "name", "kind", "dim", "units", "description",
        }:
            raise ValueError("architecture modality has an invalid schema")
        if not isinstance(item["name"], str) or not 1 <= len(item["name"]) <= 128:
            raise ValueError("architecture modality name is invalid")
        if item["kind"] not in ("tabular", "waveform"):
            raise ValueError("architecture modality kind is invalid")
        _positive_int(item["dim"], "architecture modality dimension")
        for key, limit in (("units", 256), ("description", 512)):
            if not isinstance(item[key], str) or len(item[key]) > limit:
                raise ValueError(f"architecture modality {key} is invalid")
    from core.specs import InputSpecSet
    try:
        return InputSpecSet.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"architecture input_spec is invalid: {exc}") from exc

def _module_path(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError(f"{label} path is invalid")
    parts = value.split(".")
    reserved = {
        "_modules", "_parameters", "_buffers", "training",
        "_backward_hooks", "_forward_hooks", "_forward_pre_hooks",
        "_state_dict_hooks", "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
    if any(
        not re.fullmatch(r"[A-Za-z0-9_]{1,64}", part) or part in reserved
        for part in parts
    ):
        raise ValueError(f"{label} path is invalid")
    return value

def _finite_number(value, label: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} is below its allowed range")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} is above its allowed range")
    return number

def _module_element_count(class_name: str, args: object) -> int:
    if not isinstance(args, dict):
        raise ValueError("architecture module args must be an object")
    schemas = {
        "nn.Linear": {"in_features", "out_features", "bias"},
        "nn.GELU": set(),
        "nn.ReLU": {"inplace"},
        "nn.Sigmoid": set(),
        "nn.Tanh": set(),
        "nn.SiLU": set(),
        "nn.LeakyReLU": {"negative_slope"},
        "nn.Dropout": {"p"},
        "nn.LayerNorm": {"normalized_shape", "eps", "elementwise_affine"},
        "nn.BatchNorm1d": {
            "num_features", "eps", "momentum", "affine", "track_running_stats",
        },
        "nn.Softmax": {"dim"},
        "nn.Flatten": {"start_dim", "end_dim"},
        "nn.Identity": set(),
        "nn.Embedding": {"num_embeddings", "embedding_dim"},
        "nn.Sequential": set(),
        "nn.ModuleList": set(),
        "nn.ModuleDict": set(),
    }
    if class_name not in schemas or set(args) != schemas[class_name]:
        raise ValueError(f"architecture module {class_name!r} has invalid args")
    if class_name == "nn.Linear":
        # Hashed-text modalities expose 65,536 buckets plus one availability bit.
        in_features = _positive_int(
            args["in_features"], "Linear in_features", maximum=MAX_ARCH_DIM + 1,
        )
        out_features = _positive_int(args["out_features"], "Linear out_features")
        if type(args["bias"]) is not bool:
            raise ValueError("Linear bias must be boolean")
        return in_features * out_features + (out_features if args["bias"] else 0)
    if class_name == "nn.LayerNorm":
        elements = _shape_elements(args["normalized_shape"], "LayerNorm")
        _finite_number(args["eps"], "LayerNorm eps", minimum=1e-12, maximum=1.0)
        if type(args["elementwise_affine"]) is not bool:
            raise ValueError("LayerNorm elementwise_affine must be boolean")
        return elements * (2 if args["elementwise_affine"] else 0)
    if class_name == "nn.BatchNorm1d":
        features = _positive_int(args["num_features"], "BatchNorm num_features")
        _finite_number(args["eps"], "BatchNorm eps", minimum=1e-12, maximum=1.0)
        if type(args["affine"]) is not bool or type(args["track_running_stats"]) is not bool:
            raise ValueError("BatchNorm flags must be boolean")
        momentum = args["momentum"]
        if momentum is not None:
            _finite_number(momentum, "BatchNorm momentum", minimum=0.0, maximum=1.0)
        return features * (
            (2 if args["affine"] else 0)
            + (2 if args["track_running_stats"] else 0)
        ) + (1 if args["track_running_stats"] else 0)
    if class_name == "nn.Embedding":
        return (
            _positive_int(args["num_embeddings"], "Embedding num_embeddings")
            * _positive_int(args["embedding_dim"], "Embedding embedding_dim")
        )
    if class_name == "nn.ReLU":
        if type(args["inplace"]) is not bool:
            raise ValueError("ReLU inplace must be boolean")
    elif class_name == "nn.LeakyReLU":
        _finite_number(args["negative_slope"], "LeakyReLU negative_slope",
                       minimum=0.0, maximum=100.0)
    elif class_name == "nn.Dropout":
        _finite_number(args["p"], "Dropout p", minimum=0.0, maximum=0.999999)
    elif class_name == "nn.Softmax":
        if type(args["dim"]) is not int or not -MAX_TENSOR_RANK <= args["dim"] <= MAX_TENSOR_RANK:
            raise ValueError("Softmax dim is outside the allowed range")
    elif class_name == "nn.Flatten":
        for key in ("start_dim", "end_dim"):
            if type(args[key]) is not int or not -MAX_TENSOR_RANK <= args[key] <= MAX_TENSOR_RANK:
                raise ValueError(f"Flatten {key} is outside the allowed range")
    return 0

def _validate_encoded_arg(value: object, names: set[str], depth: int = 0) -> None:
    if depth > MAX_ARCH_DEPTH:
        raise ValueError("architecture graph arguments exceed the nesting limit")
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("architecture graph argument is malformed")
    key, item = next(iter(value.items()))
    if key == "node":
        if not isinstance(item, str) or item not in names:
            raise ValueError("architecture graph references an unknown node")
    elif key == "const":
        if isinstance(item, dict):
            if set(item) == {"__dtype__"}:
                allowed = {
                    "float16", "float32", "float64", "bfloat16",
                    "int8", "uint8", "int16", "int32", "int64", "bool",
                }
                if item["__dtype__"] not in allowed:
                    raise ValueError("architecture graph dtype is not whitelisted")
            elif set(item) == {"__device__"}:
                if item["__device__"] != "cpu":
                    raise ValueError("architecture graph device must be cpu")
            else:
                raise ValueError("architecture graph constant is malformed")
        elif item is not None and not isinstance(item, (bool, int, float, str)):
            raise ValueError("architecture graph constant has an invalid type")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("architecture graph constant must be finite")
        elif isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**63 - 1:
            raise ValueError("architecture graph integer is out of range")
        elif isinstance(item, str) and len(item) > 4096:
            raise ValueError("architecture graph string is too long")
    elif key == "seq":
        if not isinstance(item, list) or len(item) > 256:
            raise ValueError("architecture graph sequence is invalid")
        for child in item:
            _validate_encoded_arg(child, names, depth + 1)
    elif key == "dict":
        if not isinstance(item, dict) or len(item) > 256:
            raise ValueError("architecture graph dictionary is invalid")
        for entry_key, child in item.items():
            if not isinstance(entry_key, str) or len(entry_key) > 128:
                raise ValueError("architecture graph dictionary key is invalid")
            _validate_encoded_arg(child, names, depth + 1)
    else:
        raise ValueError("architecture graph argument kind is not whitelisted")

def _attribute_spec(
    spec: object,
    label: str,
    element_limit: int,
) -> tuple[tuple[int, ...], torch.dtype, int]:
    if not isinstance(spec, dict) or set(spec) != {"kind", "shape", "dtype"}:
        raise ValueError(f"{label} metadata is malformed")
    if spec["kind"] not in ("parameter", "buffer"):
        raise ValueError(f"{label} kind is invalid")
    elements = _shape_elements(spec["shape"], label)
    if elements > element_limit:
        raise ValueError(f"{label} exceeds the task tensor element budget")
    dtype_name = spec["dtype"]
    allowed = {
        "float16", "float32", "float64", "bfloat16",
        "int8", "uint8", "int16", "int32", "int64", "bool",
    }
    if not isinstance(dtype_name, str) or dtype_name not in allowed:
        raise ValueError(f"{label} dtype is not whitelisted")
    return tuple(spec["shape"]), getattr(torch, dtype_name), elements

def _validate_architecture(
    arch: object, task: str, expected_n_classes: int | None,
):
    element_limit = _task_element_limit(task, expected_n_classes)
    if not isinstance(arch, dict) or set(arch) != {
        "version", "model_type", "input_spec", "modules", "graph",
    }:
        raise ValueError("architecture has an invalid top-level schema")
    if arch.get("version") != ARCH_VERSION:
        raise ValueError(
            f"architecture version {arch.get('version')!r} is not {ARCH_VERSION!r}"
        )
    if not isinstance(arch["model_type"], str) or not 1 <= len(arch["model_type"]) <= 128:
        raise ValueError("architecture model_type is invalid")
    input_spec = _validate_input_spec(arch["input_spec"])
    modules = arch["modules"]
    graph = arch["graph"]
    if not isinstance(modules, dict) or len(modules) > MAX_ARCH_MODULES:
        raise ValueError("architecture module count exceeds the limit")
    if not isinstance(graph, list) or not 1 <= len(graph) <= MAX_GRAPH_NODES:
        raise ValueError("architecture graph node count exceeds the limit")
    total_elements = 0
    paths = set()
    parameter_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for raw_path, spec in modules.items():
        path = _module_path(raw_path, "module")
        paths.add(path)
        if not isinstance(spec, dict) or set(spec) != {"class", "args"}:
            raise ValueError("architecture module spec is malformed")
        class_name = spec["class"]
        if not isinstance(class_name, str) or class_name not in MODULE_BUILDERS:
            raise ValueError(f"architecture names non-whitelisted module class {class_name!r}")
        total_elements += _module_element_count(class_name, spec["args"])
        args = spec["args"]
        prefix = path + "."
        if class_name == "nn.Linear":
            in_features = args["in_features"]
            out_features = args["out_features"]
            parameter_specs[prefix + "weight"] = ((out_features, in_features), torch.float32)
            if args["bias"]:
                parameter_specs[prefix + "bias"] = ((out_features,), torch.float32)
        elif class_name == "nn.LayerNorm" and args["elementwise_affine"]:
            shape = tuple(args["normalized_shape"])
            parameter_specs[prefix + "weight"] = (shape, torch.float32)
            parameter_specs[prefix + "bias"] = (shape, torch.float32)
        elif class_name == "nn.BatchNorm1d":
            features = args["num_features"]
            if args["affine"]:
                parameter_specs[prefix + "weight"] = ((features,), torch.float32)
                parameter_specs[prefix + "bias"] = ((features,), torch.float32)
            if args["track_running_stats"]:
                parameter_specs[prefix + "running_mean"] = ((features,), torch.float32)
                parameter_specs[prefix + "running_var"] = ((features,), torch.float32)
                parameter_specs[prefix + "num_batches_tracked"] = ((), torch.int64)
        elif class_name == "nn.Embedding":
            parameter_specs[prefix + "weight"] = (
                (args["num_embeddings"], args["embedding_dim"]), torch.float32,
            )
    for path in paths:
        parent = path.rpartition(".")[0]
        if parent and parent not in paths:
            raise ValueError(f"architecture module parent {parent!r} is missing")
    defined: set[str] = set()
    output_seen = False
    attr_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for index, spec in enumerate(graph):
        if not isinstance(spec, dict):
            raise ValueError("architecture graph node is malformed")
        op = spec.get("op")
        name = spec.get("name")
        expected_keys = {"op", "name", "target", "args", "kwargs"}
        if op == "get_attr":
            expected_keys.add("attr")
        if set(spec) != expected_keys:
            raise ValueError("architecture graph node has an invalid schema")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) is None:
            raise ValueError("architecture graph node name is invalid")
        if name in defined or output_seen:
            raise ValueError("architecture graph node order is invalid")
        target = spec["target"]
        if op == "placeholder":
            if target is not None or index != 0:
                raise ValueError("architecture placeholder is malformed")
        elif op == "call_function":
            if target not in FUNCTION_TARGETS:
                raise ValueError(f"architecture function target {target!r} is not whitelisted")
        elif op == "call_method":
            if target not in METHOD_WHITELIST:
                raise ValueError(f"architecture method target {target!r} is not whitelisted")
        elif op == "call_module":
            if target not in paths:
                raise ValueError(f"architecture call_module target {target!r} is unknown")
        elif op == "get_attr":
            target = _module_path(target, "get_attr")
            shape, dtype, elements = _attribute_spec(
                spec["attr"], f"get_attr {target}", element_limit,
            )
            recorded = attr_specs.get(target)
            if recorded is not None and recorded != (shape, dtype):
                raise ValueError(f"architecture get_attr {target!r} has conflicting metadata")
            if recorded is None:
                attr_specs[target] = (shape, dtype)
                total_elements += elements
            existing = parameter_specs.get(target)
            if existing is not None and existing != (shape, dtype):
                raise ValueError(f"architecture get_attr {target!r} mismatches its module")
            if existing is None:
                parameter_specs[target] = (shape, dtype)
        elif op == "output":
            if target is not None or index != len(graph) - 1:
                raise ValueError("architecture output node must be final")
            output_seen = True
        else:
            raise ValueError(f"architecture graph op {op!r} is not whitelisted")
        _validate_encoded_arg(spec["args"], defined)
        _validate_encoded_arg(spec["kwargs"], defined)
        defined.add(name)
    if not output_seen:
        raise ValueError("architecture graph has no output")
    if total_elements > element_limit:
        raise ValueError(
            f"architecture exceeds the {task} task element budget of {element_limit}"
        )
    return input_spec, element_limit, parameter_specs

def _validate_checkpoint_state(
    state: object,
    expected: dict[str, tuple[tuple[int, ...], torch.dtype]],
    element_limit: int,
) -> None:
    if not isinstance(state, Mapping) or len(state) > MAX_ARCH_MODULES * 8:
        raise ValueError("weights state_dict is malformed or too large")
    if set(state) != set(expected):
        raise ValueError("weights keys do not match the whitelisted architecture")
    total_elements = 0
    total_bytes = 0
    for key, value in state.items():
        if not isinstance(key, str) or len(key) > 512 or not torch.is_tensor(value):
            raise ValueError("weights state_dict contains an unsupported entry")
        shape, dtype = expected[key]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"weights tensor {key!r} shape or dtype does not match architecture")
        if (
            value.layout != torch.strided
            or not value.is_contiguous()
            or value.storage_offset() != 0
        ):
            raise ValueError(f"weights tensor {key!r} must be contiguous dense storage")
        elements = value.numel()
        total_elements += elements
        total_bytes += elements * value.element_size()
        if total_elements > element_limit or total_bytes > MAX_WEIGHTS_BYTES:
            raise ValueError("weights exceed the task element or byte budget")

def rebuild_model(
    arch: dict,
    *,
    task: str = "constitution",
    expected_n_classes: int | None = None,
) -> nn.Module:
    """Validate bounds, then rebuild a callable model from frozen code only."""
    input_spec, _element_limit, _expected = _validate_architecture(
        arch, task, expected_n_classes,
    )
    root = nn.Module()
    for path, spec in sorted(
        arch["modules"].items(), key=lambda item: (item[0].count("."), item[0])
    ):
        builder = MODULE_BUILDERS.get(spec["class"])
        if builder is None:
            raise ValueError(
                f"architecture names non-whitelisted module class {spec['class']!r}"
            )
        module = builder(**spec["args"])
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    graph = torch.fx.Graph()
    env: dict[str, torch.fx.Node] = {}
    for spec in arch["graph"]:
        op, name = spec["op"], spec["name"]
        args = _decode_arg(spec["args"], env)
        if op == "placeholder":
            node = graph.placeholder(name)
        elif op == "call_function":
            target = FUNCTION_TARGETS.get(spec["target"])
            if target is None:
                raise ValueError(
                    f"architecture names non-whitelisted function {spec['target']!r}"
                )
            node = graph.call_function(target, args, _decode_arg(spec["kwargs"], env))
        elif op == "call_method":
            if spec["target"] not in METHOD_WHITELIST:
                raise ValueError(
                    f"architecture names non-whitelisted method {spec['target']!r}"
                )
            node = graph.call_method(spec["target"], args, _decode_arg(spec["kwargs"], env))
        elif op == "call_module":
            node = graph.call_module(spec["target"], args, _decode_arg(spec["kwargs"], env))
        elif op == "get_attr":
            # register the placeholder the graph copies from at GraphModule
            # build time; the strict load fills it with the recorded tensors
            attr = spec.get("attr")
            if attr is None:
                raise ValueError(
                    f"get_attr node {spec['name']!r} carries no attr record"
                )
            tensor = torch.empty(
                attr["shape"], dtype=getattr(torch, attr["dtype"])
            )
            parent = root
            parts = spec["target"].split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            if attr["kind"] == "parameter":
                parent.register_parameter(parts[-1], nn.Parameter(tensor))
            else:
                parent.register_buffer(parts[-1], tensor)
            node = graph.get_attr(spec["target"])
        elif op == "output":
            graph.output(args[0] if isinstance(args, tuple) else args)
            break
        else:
            raise ValueError(f"malformed graph node kind {op!r}")
        env[name] = node

    model = torch.fx.GraphModule(root, graph)
    model.input_spec = input_spec
    model.model_type = arch["model_type"]
    model.eval()
    return model


def save_eval_model(
    model: nn.Module,
    out_dir: str | Path,
    stem: str,
    *,
    task: str = "constitution",
    expected_n_classes: int | None = None,
) -> dict:
    """Write only a task-bounded state_dict and frozen architecture."""
    arch = export_architecture(
        model, task=task, expected_n_classes=expected_n_classes,
    )
    _input_spec, element_limit, expected = _validate_architecture(
        arch, task, expected_n_classes,
    )
    state = model.state_dict()
    _validate_checkpoint_state(state, expected, element_limit)
    architecture_bytes = json.dumps(
        arch, indent=2, ensure_ascii=False,
    ).encode("utf-8")
    if len(architecture_bytes) > MAX_ARCHITECTURE_BYTES:
        raise ArchitectureExportRefused(
            f"architecture JSON exceeds the {MAX_ARCHITECTURE_BYTES}-byte limit"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / f"{stem}.pt"
    arch_path = out / f"{stem}-{ARCHITECTURE_FILE}"
    torch.save(state, weights_path)
    if weights_path.stat().st_size > MAX_WEIGHTS_BYTES:
        weights_path.unlink(missing_ok=True)
        raise ArchitectureExportRefused(
            f"weights file exceeds the {MAX_WEIGHTS_BYTES}-byte limit"
        )
    arch_path.write_bytes(architecture_bytes)
    return {
        "weights_path": str(weights_path),
        "architecture_path": str(arch_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "architecture_sha256": hashlib.sha256(arch_path.read_bytes()).hexdigest(),
    }


def load_eval_model(
    weights_path: str | Path,
    arch_path: str | Path,
    *,
    task: str = "constitution",
    expected_n_classes: int | None = None,
) -> nn.Module:
    """Load bounded state_dict tensors into a fully prevalidated architecture."""
    arch_payload = _bounded_regular_bytes(
        arch_path, maximum=MAX_ARCHITECTURE_BYTES, label="architecture JSON",
    )
    try:
        arch = json.loads(arch_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("architecture JSON is invalid") from exc
    _input_spec, element_limit, expected = _validate_architecture(
        arch, task, expected_n_classes,
    )
    weight_payload = _bounded_regular_bytes(
        weights_path, maximum=MAX_WEIGHTS_BYTES, label="weights file",
    )
    try:
        state = torch.load(
            io.BytesIO(weight_payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"weights file could not be safely loaded: {exc}") from exc
    _validate_checkpoint_state(state, expected, element_limit)
    model = rebuild_model(
        arch, task=task, expected_n_classes=expected_n_classes,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"weights do not match the recorded architecture: {exc}"
        ) from exc
    model.eval()
    return model
