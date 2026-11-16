#!/usr/bin/env python3
"""Train a DSAD segmentation model on train surgeries, select on validation.

Arm A1 (specialist): ``--structures ureter`` trains one binary model on the
frames that carry a ureter mask. Arm A2+ (single model): ``--structures all``.
Only the train and val splits are ever read here.

A run that would exceed ``--time-budget-min`` saves ``last.pt`` and exits with
code 75; starting it again with the same arguments resumes from ``last.pt``.
Resumption restores model, optimiser, scheduler, scaler and RNG states, so it is
equivalent in distribution, not bitwise, to an uninterrupted run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from davinci.data.dsad import STRUCTURES
from davinci.data.dsad_torch import DSADDataset
from davinci.metrics.dsad_eval import evaluate
from davinci.models.seg import build_model, masked_bce_dice

EXIT_REQUEUE = 75


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bce_only(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def rng_state(g: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader": g.get_state()}


def set_rng_state(state: dict, g: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    g.set_state(state["loader"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--structures", default="all")
    ap.add_argument("--model", default="deeplabv3_r50_coco")
    ap.add_argument("--loss", choices=("bce", "bce_dice"), default="bce_dice")
    ap.add_argument("--dice-on", choices=("weighted", "mask"), default="weighted",
                    help="channels entering soft Dice: any weighted channel (v1.2) or mask channels only (amendment 3)")
    ap.add_argument("--scheduler", choices=("step09", "cosine"), default="cosine")
    ap.add_argument("--use-absent", action="store_true")
    ap.add_argument("--overlap-pairs", type=Path, default=None,
                    help="JSON from p2_overlap_pairs.py; exempt pairs keep no other-mask negatives")
    ap.add_argument("--presence-loss-weight", type=float, default=0.5)
    ap.add_argument("--soft-targets", action="store_true", help="mean of three annotators (A6)")
    ap.add_argument("--multilabel", action="store_true", help="add dense multilabel/ masks (ablation)")
    ap.add_argument("--train-surgeries", type=int, default=0,
                    help="label budget: use this many training surgeries (seeded subset, seed 2027)")
    ap.add_argument("--negative-ratio", type=float, default=0.0,
                    help="specialists: absent frames added per mask frame (A1N)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="micro-batches per optimiser step (effective batch = batch-size x accum-steps)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (args.output / "result.json").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    structures = STRUCTURES if args.structures == "all" else tuple(args.structures.split(","))
    specialist = len(structures) == 1
    manifest = json.loads(args.manifest.read_text())
    kwargs = dict(structures=structures, use_absent=args.use_absent,
                  require_mask=structures[0] if specialist else None,
                  other_mask_negatives=not specialist, cache_root=args.cache_root)
    if args.negative_ratio:
        assert specialist and args.use_absent, "--negative-ratio needs one structure and --use-absent"
    if args.overlap_pairs:
        kwargs["overlap_pairs"] = [tuple(p) for p in json.loads(args.overlap_pairs.read_text())["overlap_pairs"]]
    if args.model.startswith("dinov2"):
        kwargs["size_hw"] = (518, 644)  # multiples of the 14-pixel patch
    surgeries = None
    if args.train_surgeries:
        pool = sorted({r["surgery"] for r in manifest["records"] if r["split"] == "train"})
        surgeries = sorted(np.random.default_rng(2027).permutation(pool)[:args.train_surgeries].tolist())
    train_ds = DSADDataset(manifest, "train", train=True, negative_ratio=args.negative_ratio,
                           negative_seed=args.seed, soft_targets=args.soft_targets,
                           use_multilabel=args.multilabel, surgeries=surgeries, **kwargs)
    val_ds = DSADDataset(manifest, "val", train=False, **kwargs)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, generator=g, persistent_workers=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    model = build_model(args.model, len(structures)).to(device, memory_format=torch.channels_last)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and (".a." in n or ".b." in n)]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not (".a." in n or ".b." in n)]
    groups = [{"params": rest, "lr": args.lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lr * 0.5})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    trainable = sum(p.numel() for p in rest + lora)
    if args.scheduler == "step09":  # Kolbinger et al.: x0.9 every 10 epochs
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    else:
        warm = 3
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else
                                                  0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, args.epochs - warm))))
    scaler = torch.amp.GradScaler()
    loss_fn = bce_only if args.loss == "bce" else masked_bce_dice
    history, best, best_epoch, first_epoch = [], -1.0, -1, 0
    select_key = "critical_macro_iou" if not specialist else "macro_iou"
    run_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_args["overlap_pairs_list"] = kwargs.get("overlap_pairs", [])
    run_args["train_surgery_ids"] = surgeries
    last_path = args.output / "last.pt"
    resumes = 0
    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"], g)
        history, best, best_epoch = ck["history"], ck["best"], ck["best_epoch"]
        first_epoch, resumes = ck["epoch"] + 1, ck["resumes"] + 1
        print(json.dumps({"resumed_from_epoch": ck["epoch"], "resumes": resumes}), flush=True)
    t0 = time.time() - (history[-1]["elapsed_s"] if history else 0.0)
    for epoch in range(first_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        losses, skipped, data_s = [], 0, 0.0
        opt.zero_grad(set_to_none=True)
        tick = time.time()
        for step, batch in enumerate(train_dl):
            x = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            t = batch["target"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True)
            data_s += time.time() - tick
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            logits, presence = out if isinstance(out, tuple) else (out, None)
            if loss_fn is masked_bce_dice and args.dice_on == "mask":
                loss = loss_fn(logits, t, w, dice_channels=batch["has_mask"].to(device))
            else:
                loss = loss_fn(logits, t, w)
            if presence is not None:
                pw = batch["presence_weight"].to(device)
                pl = F.binary_cross_entropy_with_logits(
                    presence.float(), batch["presence"].to(device), reduction="none")
                loss = loss + args.presence_loss_weight * (pl * pw).sum() / pw.sum().clamp_min(1.0)
            scaler.scale(loss / args.accum_steps).backward()
            losses.append(loss.item())
            if (step + 1) % args.accum_steps == 0:
                scale = scaler.get_scale()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                skipped += int(scaler.get_scale() < scale)
            tick = time.time()
        sched.step()
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "amp_skipped": skipped,
                 "lr": opt.param_groups[0]["lr"], "data_wait_s": data_s,
                 "elapsed_s": time.time() - t0}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val = evaluate(model, val_dl, structures, device)
            entry["val"] = val
            score = val[select_key] if val[select_key] is not None else -1.0
            if score > best:
                best, best_epoch = score, epoch
                torch.save({"model": model.state_dict(), "args": run_args, "epoch": epoch,
                            "structures": structures}, args.output / "best.pt")
        history.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "val"} |
                         ({"val_" + select_key: entry["val"][select_key]} if "val" in entry else {})),
              flush=True)
        epoch_s = time.time() - epoch_start
        out_of_time = (args.time_budget_min > 0 and epoch < args.epochs - 1 and
                       time.time() - start + 1.5 * epoch_s > args.time_budget_min * 60)
        if out_of_time or (epoch + 1) % 5 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "rng": rng_state(g), "history": history, "best": best,
                        "best_epoch": best_epoch, "epoch": epoch, "resumes": resumes},
                       last_path.with_suffix(".tmp"))
            last_path.with_suffix(".tmp").replace(last_path)
        if out_of_time:
            print(json.dumps({"requeue_after_epoch": epoch}), flush=True)
            sys.exit(EXIT_REQUEUE)
    selection = "best_validation"
    if best_epoch < 0:
        # The fold's validation surgeries have no mask of a selection structure
        # (amendment 8): keep the final checkpoint.
        selection = "final_epoch_undefined_validation_metric"
        best_epoch = history[-1]["epoch"]
        torch.save({"model": model.state_dict(), "args": run_args, "epoch": best_epoch,
                    "structures": structures}, args.output / "best.pt")
        history[-1].setdefault("val", evaluate(model, val_dl, structures, device))
    result = {"args": run_args, "selection": selection,
              "structures": list(structures), "selection_metric": select_key,
              "best_epoch": best_epoch, "best_val": best,
              "best_val_metrics": next(h["val"] for h in history if h["epoch"] == best_epoch),
              "train_frames": len(train_ds), "trainable_parameters": trainable, "val_frames": len(val_ds),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "resumes": resumes, "test_accessed": False, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "history": history}
    (args.output / "result.json").write_text(json.dumps(result, indent=1))
    last_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
