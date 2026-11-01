"""E5: equal-budget learning-rate tuning for XLM-R large and IndicBERT v2 on emotion (pre-registered).

  python scripts/e5_tune.py --device cuda --save /content/drive/MyDrive/mhrisk_e5

Stage 1 runs each grid cell of configs/E5_1_* and configs/E5_2_* with seed 13 and no test evaluation.
Selection uses validation only: the encoder LR with the highest calibrated validation emotion macro-F1
(ties go to the lower LR). Stage 2 copies that LR into configs/E5_3_* / E5_4_* (whose encoder_lr is null
until then) and runs seeds 13, 21, 42 with test evaluation. A finished run is skipped, so the script can be
restarted after a disconnect; with --save, finished runs are copied there and restored from there.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.experiments.runner import expand, run_one, runs_dir  # noqa: E402

PAIRS = (("E5_1_xlmr_large_emotion_grid.json", "E5_3_xlmr_large_emotion_tuned.json"),
         ("E5_2_indicbertv2_emotion_grid.json", "E5_4_indicbertv2_emotion_tuned.json"))


def sync(src: Path, dst: Path):
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("resume*", "checkpoint"))


def select(grid_cfg: dict) -> dict:
    """Validation-only choice of encoder_lr from the seed-13 grid runs."""
    rows = []
    for c in expand(grid_cfg):
        m = json.loads((runs_dir() / c["name"] / "seed_13" / "metrics.json").read_text())
        assert m["test"] is None, f"{c['name']} has test metrics; grid runs must not see test"
        rows.append({"run": c["name"], "encoder_lr": c["train"]["encoder_lr"], "val_macro_f1": m["val"]["emotion"]["macro_f1"],
                     "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"]})
    best = max(rows, key=lambda r: (round(r["val_macro_f1"], 6), -r["encoder_lr"]))
    return {"grid": grid_cfg["name"], "rule": "max seed-13 validation emotion macro-F1; ties to the lower LR",
            "candidates": rows, "chosen_encoder_lr": best["encoder_lr"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="folder that keeps finished runs (e.g. on Google Drive)")
    a = ap.parse_args()
    save = Path(a.save) if a.save else None

    def run(cfg, seed):
        if save:
            sync(save / cfg["name"] / f"seed_{seed}", runs_dir() / cfg["name"] / f"seed_{seed}")
        out = run_one(cfg, seed, a.device)
        m = json.loads((out / "metrics.json").read_text())
        print(f"  {cfg['name']} seed {seed}: val {m['val']['emotion']['macro_f1']:.4f}, best epoch {m['best_epoch']}/{m['epochs_run']}", flush=True)
        if save:
            sync(out, save / cfg["name"] / f"seed_{seed}")

    for grid_file, final_file in PAIRS:
        grid = json.loads((ROOT / "configs" / grid_file).read_text())
        print(f"stage 1: {grid['name']}", flush=True)
        for c in expand(grid):
            run(c, 13)
        sel = select(grid)
        print(f"  chosen encoder_lr {sel['chosen_encoder_lr']}", flush=True)
        final = json.loads((ROOT / "configs" / final_file).read_text())
        assert final["train"]["encoder_lr"] is None, f"{final_file}: encoder_lr must stay null until selection"
        final["train"]["encoder_lr"] = sel["chosen_encoder_lr"]
        final["selection"] = sel
        (runs_dir() / final["name"]).mkdir(parents=True, exist_ok=True)
        (runs_dir() / final["name"] / "selection.json").write_text(json.dumps(sel, indent=2))
        if save:
            (save / final["name"]).mkdir(parents=True, exist_ok=True)
            (save / final["name"] / "selection.json").write_text(json.dumps(sel, indent=2))
        print(f"stage 2: {final['name']}", flush=True)
        for seed in final["seeds"]:
            run(final, seed)
