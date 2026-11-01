"""SD v1 campaign driver — source-level AutoResearch on TCM-SD (issue #53).

Same mechanics as the v10/v11 drivers: every patch is generated against the
CURRENT committed state of the worktree's candidate_sd.py with asserted
anchors (a silently non-matching replacement would produce an unchanged file
and a zero diff, which is how a patch list quietly turns into no-ops). The
engine adjudicates KEEP/REVERT; this driver only proposes.
"""
import json
import pathlib
import subprocess
import sys

WT = pathlib.Path("/home/lora/workspace/ar-sd")
REPO = pathlib.Path("/home/lora/workspace/projects/MulitiModal")
TARGET = WT / "src/research/candidate_sd.py"
OUT = REPO / "runs/sd-campaign"
MANIFEST = REPO / "runs/sd-campaign/manifest.json"
DATASET = REPO / "runs/sd-campaign/records.json"
PROPOSER = "main:zai-glm-driver"
FOLDS, REPEATS = 3, 2
BUDGET = "90"

TRAIN_ANCHOR = '''    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    deadline = time.monotonic() + budget_seconds
    model.train()
    while time.monotonic() < deadline:
        epoch_done = False
        for item in fold.train:
            batch = item.batch
            out = model(batch)'''
TRAIN_CUDA = '''    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    deadline = time.monotonic() + budget_seconds
    model.train()
    while time.monotonic() < deadline:
        epoch_done = False
        for item in fold.train:
            batch = item.batch
            if device.type != "cpu":
                batch = ModelBatch(
                    features={k: v.to(device) for k, v in batch.features.items()},
                    availability={
                        k: v.to(device) for k, v in batch.availability.items()
                    },
                    targets={k: v.to(device) for k, v in batch.targets.items()},
                    target_mask={
                        k: v.to(device) for k, v in batch.target_mask.items()
                    },
                )
            out = model(batch)'''


def replace(text, anchor, new, label):
    assert anchor in text, f"{label}: anchor not found"
    return text.replace(anchor, new, 1)


def t_cuda(text):
    text = replace(text, "from core.batch import ModelOutput",
                   "from core.batch import ModelBatch, ModelOutput", "cuda import")
    return replace(text, TRAIN_ANCHOR, TRAIN_CUDA, "cuda train")


def t_lr(text):
    return replace(text, "LEARNING_RATE = 3e-3", "LEARNING_RATE = 1e-2", "lr")


def t_class_weights(text):
    """Inverse-frequency class weights over the fold's labeled train rows."""
    old = '''    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)'''
    new = '''    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_classes = model.n_targets
    counts = torch.zeros(n_classes)
    for item in fold.train:
        b = item.batch
        m = b.target_mask["syndrome_labels"].squeeze(-1).bool()
        labels = b.targets["syndrome_labels"].squeeze(-1)[m].long()
        counts += torch.bincount(labels, minlength=n_classes)
    present = counts[counts > 0]
    weights = torch.zeros(n_classes)
    weights[counts > 0] = present.sum() / (len(present) * present)
    weights = weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)'''
    text = replace(text, old, new, "class-weight counts")
    old_loss = "            loss = F.cross_entropy(logits[mask], labels)"
    new_loss = "            loss = F.cross_entropy(logits[mask], labels, weight=weights)"
    return replace(text, old_loss, new_loss, "class-weight loss")


def patch_for(transform, name):
    original = TARGET.read_text()
    TARGET.write_text(transform(original))
    diff = subprocess.run(["git", "diff"], cwd=WT, text=True,
                          capture_output=True).stdout
    TARGET.write_text(original)
    assert diff.strip(), f"{name}: empty diff — the transform matched nothing"
    path = pathlib.Path(f"/tmp/sdv1-{name}.patch")
    path.write_text(diff)
    return path


def run_round(name, patch_path, intent):
    cmd = [
        sys.executable, "-m", "runtime.cli", "research-src-round",
        "--worktree", str(WT), "--branch", "autoresearch/sd-v1",
        "--data-manifest", str(MANIFEST), "--dataset", str(DATASET),
        "--out", str(OUT), "--patch-file", str(patch_path),
        "--intent", intent, "--proposer", PROPOSER,
        "--task", "syndrome", "--editable", "src/research/candidate_sd.py",
        "--budget-s", BUDGET, "--folds", str(FOLDS), "--repeats", str(REPEATS),
    ]
    env = {"PATH": "/usr/bin:/bin", "HOME": "/home/lora",
           "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, env=env)
    try:
        entry = json.loads(proc.stdout[proc.stdout.index("{"):])
    except (ValueError, json.JSONDecodeError):
        print(f"{name}: NO JSON (rc={proc.returncode}) {proc.stderr[-400:]}",
              flush=True)
        return None
    value = (entry.get("metrics") or {}).get("primary_value")
    print(
        f"{name:18} {entry['decision']:6} {entry['status']:9} "
        f"{'-' if value is None else format(value, '.4f')} "
        f"growth={(entry.get('complexity') or {}).get('growth')} "
        f"| {entry['reason'][:90]}",
        flush=True,
    )
    return entry


def main() -> int:
    rounds = [
        # round 0 (cuda) already landed as the KEEP baseline; the transforms
        # below anchor on the CURRENT committed file (which has the cuda
        # changes), never re-applying an already-landed patch
        ("cuda_lr1e2", t_lr,
         "lr 3e-3 -> 1e-2 on the CUDA baseline: 148-way softmax over a wide "
         "input usually wants larger steps than the small-skeleton default"),
        ("cuda_classw", t_class_weights,
         "inverse-frequency class weights: the train distribution spans 7,566 "
         "to 9 cases per class, so unweighted CE under-ranks the 100+ tail "
         "classes that macro-F1 averages over"),
    ]
    plateau = 0
    for name, transform, intent in rounds:
        patch = patch_for(transform, name)
        entry = run_round(name, patch, intent)
        if entry is None:
            continue
        plateau = 0 if entry["decision"] == "KEEP" else plateau + 1
        if plateau >= 5:
            print(f"plateau: {plateau} consecutive REVERTs, stopping", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
