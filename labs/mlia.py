"""Small helpers shared by every ML-in-Action lab notebook (lives next to the notebooks in labs/).

    from mlia import load_checkpoint, save_checkpoint, show_check

Checkpoints let a learner who fell behind jump to the start of any section:
    df, X_train = load_checkpoint("lab1-3")   # loads checkpoints/lab1-3.pkl (made from the solution notebook)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
RANDOM_STATE = 42


def save_checkpoint(name: str, **objects) -> None:
    """Used by the instructor's build (MLIA_SAVE_CHECKPOINTS=1) — learners never need to call it."""
    if os.environ.get("MLIA_SAVE_CHECKPOINTS") != "1":
        return
    CKPT.mkdir(exist_ok=True)
    with open(CKPT / f"{name}.pkl", "wb") as f:
        pickle.dump(objects, f)


def load_checkpoint(name: str):
    """Return the saved objects of checkpoint `name` in the order they were saved (1 object → that object)."""
    path = CKPT / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบ {path.name} — ตรวจว่าเปิด notebook จากโฟลเดอร์ labs/ และ git pull ล่าสุดแล้ว")
    with open(path, "rb") as f:
        objs = pickle.load(f)
    print("โหลด checkpoint", name, "→", ", ".join(objs))
    vals = list(objs.values())
    return vals[0] if len(vals) == 1 else tuple(vals)


def show_check(ok: bool, good: str, bad: str) -> None:
    """Print a ✅ / ❌ line so learners can self-check a TODO."""
    print(("✅ " + good) if ok else ("❌ " + bad))
