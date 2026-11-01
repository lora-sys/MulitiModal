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
import json
import operator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from core.batch import ModelOutput

ARCHITECTURE_FILE = "architecture.json"
EVAL_WEIGHTS_FILE = "eval-weights.pt"
ARCH_VERSION = "research-arch-v1"


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


def export_architecture(model: nn.Module) -> dict:
    """Trace `model` into the frozen JSON vocabulary, or refuse."""
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
        entry = MODULE_EXPORT.get(type(mod))
        if entry is None:
            raise ArchitectureExportRefused(
                f"module {name!r} of type {type(mod).__name__} is not in the "
                "frozen architecture whitelist; evaluation only trusts "
                "state_dict + a whitelisted architecture"
            )
        key, extract = entry
        modules[name] = {"class": key, "args": extract(mod)}

    own_attrs = {p for _, p in model.named_parameters()}
    own_attrs |= {b for _, b in model.named_buffers()}
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


def rebuild_model(arch: dict) -> nn.Module:
    """Rebuild a callable model from serialized architecture — frozen code only."""
    if arch.get("version") != ARCH_VERSION:
        raise ValueError(
            f"architecture version {arch.get('version')!r} is not {ARCH_VERSION!r}"
        )
    root = nn.Module()
    for path, spec in arch["modules"].items():
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
            node = graph.get_attr(spec["target"])
        elif op == "output":
            graph.output(args[0] if isinstance(args, tuple) else args)
            break
        else:
            raise ValueError(f"malformed graph node kind {op!r}")
        env[name] = node

    from core.specs import InputSpecSet

    model = torch.fx.GraphModule(root, graph)
    model.input_spec = InputSpecSet.from_dict(arch["input_spec"])
    model.model_type = arch["model_type"]
    model.eval()
    return model


def save_eval_model(model: nn.Module, out_dir: str | Path, stem: str) -> dict:
    """Write the state_dict-only weights + frozen architecture description."""
    arch = export_architecture(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / f"{stem}.pt"
    arch_path = out / f"{stem}-{ARCHITECTURE_FILE}"
    torch.save(model.state_dict(), weights_path)
    arch_path.write_text(json.dumps(arch, indent=2, ensure_ascii=False))
    return {
        "weights_path": str(weights_path),
        "architecture_path": str(arch_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "architecture_sha256": hashlib.sha256(arch_path.read_bytes()).hexdigest(),
    }


def load_eval_model(weights_path: str | Path, arch_path: str | Path) -> nn.Module:
    """Load state_dict (weights_only) into the frozen rebuild, strictly."""
    arch = json.loads(Path(arch_path).read_text())
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = rebuild_model(arch)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"weights do not match the recorded architecture: {exc}"
        ) from exc
    model.eval()
    return model
