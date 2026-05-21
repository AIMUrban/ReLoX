# main.py
import os
import argparse
import copy
import datetime
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# import project deps
from framework import ReLoX
from tools import (
    append_city_report,
    evaluate,
    format_metrics,
    format_short_metrics,
    infer_class_counts_from_maps,
    move_batch_to_device,
    prefix_metrics,
    resolve_grid_wh,
    resolve_out_dir,
)
from traj_dataset import build_datasets, make_collator


def parse_args():
    ap = argparse.ArgumentParser("Next-location training + zero-shot testing entry")

    # --- data ---
    ap.add_argument("--data_root", default="dataset/nyc250m",
                    help="training data root (e.g., dataset/kumamoto)")
    ap.add_argument("--test_data_root", type=str, default="dataset/kumamoto",
                    help="[optional] zero-shot target data root; empty to skip")
    ap.add_argument("--test_data_roots", nargs="*", default=[],
                    help="[optional] zero-shot target data roots (repeatable). "
                         "If set, overrides --test_data_root.")
    ap.add_argument("--zeroshot_only", action="store_true", default=True,
                    help="Skip training; run zero-shot testing only (requires trained weights under --data_root).")
    ap.add_argument("--test_only", action="store_true", default=False,
                    help="Skip training; run in-domain test only (requires trained weights under --data_root).")
    ap.add_argument("--zeroshot_ckpt", type=str, default="best",
                    help="Which checkpoint to load for evaluation (test/zero-shot). "
                         "Options: 'best' (alias for model_best.pt), 'model.pt', "
                         "or a filename under <out_dir>/model_ckpt/, or an explicit path. "
                         "Use --out_dir to change the run output folder.")

    # --- training ---
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_every", type=int, default=0,
                    help="test every N epochs (0 means only after training)")

    # --- loss weights ---
    ap.add_argument("--lambda_rel", type=float, default=1)
    ap.add_argument("--lambda_abs", type=float, default=1)

    # --- model / vision ---
    ap.add_argument("--grid_w", type=int, default=200)
    ap.add_argument("--grid_h", type=int, default=200)
    ap.add_argument("--rel_size", type=int, default=8)
    ap.add_argument("--match_dim", type=int, default=128)
    ap.add_argument("--drop_r_p", type=float, default=0.5,
                    help="drop prob for local R channel in training")

    # --- misc ---
    ap.add_argument("--time_size", type=int, default=24, help="time dimension")
    ap.add_argument("--out_dir", type=str, default="",
                    help="Optional output folder for this run. "
                         "If relative, it is created under --data_root. "
                         "Default: <data_root>/outputs (shared, not safe for parallel sweeps).")
    ap.add_argument("--desc", type=str, default="", help="experiment note")

    return ap.parse_args()


def run_training(args):
    """
    training + in-domain test
    """
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] start training...")
    print(f"Train Data Root: {args.data_root}")

    # Datasets
    train_ds, eval_ds, test_ds, _ = build_datasets(args)
    n_train = len(train_ds) if train_ds is not None else 0
    n_val = len(eval_ds) if eval_ds is not None else 0
    n_test = len(test_ds) if test_ds is not None else 0

    # Model
    model = ReLoX(args=args)

    # DataLoader
    data_collator = make_collator()
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=data_collator,
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=data_collator,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-6)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[6, 8],
        gamma=0.5,
    )

    if n_train > 0:
        print(f"[Dataset sizes] train={n_train:,} | val={n_val:,} | test={n_test:,}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable params: {trainable:,}")

    # checkpoint dir (save every epoch)
    out_dir = resolve_out_dir(args)
    save_dir = os.path.join(out_dir, "model_ckpt")
    os.makedirs(save_dir, exist_ok=True)

    best_metric = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 30 + f" Epoch {epoch}/{args.epochs} " + "=" * 30)
        model.train()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"), start=1):
            batch = move_batch_to_device(batch, device)
            out = model(**batch)
            loss = out["loss"]
            loss.backward()

            bs = batch["target_loc"].size(0)
            running_loss += float(out["loss"].item()) * bs
            seen += bs

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = running_loss / max(seen, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"[Train] loss={avg_train_loss:.4f} | lr={lr:.6g}")

        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, args, device, desc=f"Val {epoch}/{args.epochs}")
            print(f"[Val] {format_short_metrics(format_metrics(eval_metrics), 'eval')}")

            metric_key = "eval_final_mrr" if "eval_final_mrr" in eval_metrics else "eval_mrr"
            current_metric = eval_metrics.get(metric_key, -eval_metrics.get("eval_loss", 0.0))
            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(save_dir, "model_best.pt"))

        lr_scheduler.step()

        if test_ds is not None and args.test_every and (epoch % args.test_every == 0):
            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=data_collator,
            )
            test_metrics = prefix_metrics(
                evaluate(model, test_loader, args, device, desc=f"Test {epoch}/{args.epochs}"),
                "test",
            )
            print(f"[Test] {format_short_metrics(format_metrics(test_metrics), 'test')}")

        # save epoch checkpoint
        epoch_ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Saved epoch checkpoint to: {epoch_ckpt_path}")

    # Restore best checkpoint if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # save model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Saved model to: {save_dir}")
    print("Training Completed:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # in-domain test
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        test_metrics = prefix_metrics(
            evaluate(model, test_loader, args, device, desc="Test"),
            "test",
        )

        print("\n" + "=" * 50)
        print("Source Test Results")
        print("=" * 50)
        print(f"Dataset: {args.data_root}")

        formatted_metrics = format_metrics(test_metrics)
        print(f"[Test] {format_short_metrics(formatted_metrics, 'test')}")

        if args.desc:
            print(f"DESC: {args.desc}")
        print("=" * 50)

        report = "\n".join([
            "=" * 50,
            f"Source Test Results ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            f"Dataset: {args.data_root}",
            f"[Test] {format_short_metrics(formatted_metrics, 'test')}",
            f"DESC: {args.desc}" if args.desc else "",
            "=" * 50,
        ])
        report_path = append_city_report(out_dir, "test_results.txt", report)
        print(f"Saved source test report to: {report_path}")


def run_zeroshot(args):
    """
    zero-shot test
    """
    zs_args = copy.deepcopy(args)
    zs_args.data_root = args.test_data_root
    zs_args.zero_shot = True
    # For cls heads, abs head shape may mismatch across cities (different grid sizes), so we disable it by default.
    zs_args.lambda_abs = 0
    zs_args.grid_w, zs_args.grid_h = resolve_grid_wh(zs_args.data_root, zs_args.grid_w, zs_args.grid_h)

    print("\n" + "#" * 50)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] start zero-shot testing...")
    print(f"Source (Train): {args.data_root}")
    print(f"Target (Test) : {args.test_data_root}")
    print("#" * 50)

    # 1. data
    _, _, test_ds, _ = build_datasets(zs_args)
    if test_ds is None:
        print(f"Error: no test set found at '{args.test_data_root}'")
        return

    # 2. infer class counts
    try:
        cls_info = infer_class_counts_from_maps(args.test_data_root)
        n_loc = cls_info["n_loc"]
        print(f"[Location info] n_loc={n_loc:,} (target domain)")
    except Exception as e:
        print(f"[Warning] failed to infer target class counts: {e}")

    # 3. init model
    model = ReLoX(args=zs_args)

    # 4. load weights (shape-aware)
    train_out_dir = resolve_out_dir(args)
    ckpt_arg = str(getattr(args, "zeroshot_ckpt", "best") or "best").strip()
    if ckpt_arg.lower() in ("best", "bset"):
        ckpt_arg = "model_best.pt"

    if os.path.exists(ckpt_arg):
        ckpt_path = ckpt_arg
    else:
        ckpt_path = os.path.join(train_out_dir, "model_ckpt", ckpt_arg)

    if not os.path.exists(ckpt_path):
        print(f"Error: trained weights not found at {ckpt_path}")
        return

    try:
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts = "unknown"
    print(f"[ZeroShot] Loading checkpoint: {ckpt_path} (mtime={ts})")

    custom_state_dict = torch.load(ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()

    filtered_state_dict = {}
    matched = 0
    shape_mismatch = 0
    mismatch_items = []
    for k, v in custom_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
            matched += 1
        elif k in model_state_dict:
            shape_mismatch += 1
            mismatch_items.append((k, tuple(v.shape), tuple(model_state_dict[k].shape)))

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    mismatch_keys = [x[0] for x in mismatch_items]
    suffix = f" mismatch_keys={mismatch_keys}" if mismatch_keys else ""
    print(
        f"[ZeroShot] Loaded params: matched={matched} shape_mismatch={shape_mismatch} "
        f"total_in_ckpt={len(custom_state_dict)}{suffix}"
    )
    if mismatch_items:
        for k, ckpt_shape, model_shape in mismatch_items:
            print(f"  - {k}: ckpt{ckpt_shape} != model{model_shape}")

    # 5. evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_collator = make_collator()
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    test_metrics = prefix_metrics(
        evaluate(model, test_loader, zs_args, device, desc="ZeroShot"),
        "zeroshot",
    )

    # 6. report
    print("\n" + "=" * 50)
    print("Zero-Shot Results")
    print("=" * 50)
    print(f"Source Domain: {args.data_root}")
    print(f"Target Domain: {args.test_data_root}")

    formatted_metrics = format_metrics(test_metrics)
    print(f"[ZeroShot] {format_short_metrics(formatted_metrics, 'zeroshot')}")

    if args.desc:
        print(f"DESC: {args.desc}")
    print("=" * 50)

    report = "\n".join([
        "=" * 50,
        f"Zero-Shot Results ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"Source Domain: {args.data_root}",
        f"Target Domain: {args.test_data_root}",
        f"[ZeroShot] {format_short_metrics(formatted_metrics, 'zeroshot')}",
        f"DESC: {args.desc}" if args.desc else "",
        "=" * 50,
    ])
    # Write train-city -> target-city reports under the train city's output directory.
    report_path = append_city_report(train_out_dir, "zeroshot_results.txt", report)
    print(f"Saved zero-shot report to (source): {report_path}")


def run_test_only(args):
    print("\n" + "#" * 50)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] start in-domain testing only...")
    print(f"Dataset (Test): {args.data_root}")
    print("#" * 50)

    # 1. data
    _, _, test_ds, _ = build_datasets(args)
    if test_ds is None:
        print(f"Error: no test set found at '{args.data_root}'")
        return

    # 2. init model
    model = ReLoX(args=args)

    # 3. load weights (shape-aware)
    train_out_dir = resolve_out_dir(args)
    ckpt_arg = str(getattr(args, "zeroshot_ckpt", "best") or "best").strip()
    if ckpt_arg.lower() in ("best", "bset"):
        ckpt_arg = "model_best.pt"

    if os.path.exists(ckpt_arg):
        ckpt_path = ckpt_arg
    else:
        ckpt_path = os.path.join(train_out_dir, "model_ckpt", ckpt_arg)

    if not os.path.exists(ckpt_path):
        print(f"Error: trained weights not found at {ckpt_path}")
        return

    try:
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts = "unknown"
    print(f"[TestOnly] Loading checkpoint: {ckpt_path} (mtime={ts})")

    custom_state_dict = torch.load(ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()

    filtered_state_dict = {}
    matched = 0
    shape_mismatch = 0
    for k, v in custom_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
            matched += 1
        elif k in model_state_dict:
            shape_mismatch += 1

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    print(f"[TestOnly] Loaded params: matched={matched} shape_mismatch={shape_mismatch} total_in_ckpt={len(custom_state_dict)}")

    # 4. evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_collator = make_collator()
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    test_metrics = prefix_metrics(
        evaluate(model, test_loader, args, device, desc="TestOnly"),
        "test",
    )

    print("\n" + "=" * 50)
    print("Source Test Results (TestOnly)")
    print("=" * 50)
    print(f"Dataset: {args.data_root}")

    formatted_metrics = format_metrics(test_metrics)
    print(f"[Test] {format_short_metrics(formatted_metrics, 'test')}")

    if args.desc:
        print(f"DESC: {args.desc}")
    print("=" * 50)

    report = "\n".join([
        "=" * 50,
        f"Source Test Results (TestOnly) ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"Dataset: {args.data_root}",
        f"Checkpoint: {ckpt_path} (mtime={ts})",
        f"[Test] {format_short_metrics(formatted_metrics, 'test')}",
        f"DESC: {args.desc}" if args.desc else "",
        "=" * 50,
    ])
    report_path = append_city_report(train_out_dir, "test_results.txt", report)
    print(f"Saved source test report to: {report_path}")


def main():
    args = parse_args()
    args.grid_w, args.grid_h = resolve_grid_wh(args.data_root, args.grid_w, args.grid_h)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.test_only:
        run_test_only(args)
    elif not args.zeroshot_only:
        run_training(args)

    # zero-shot test on one or multiple target cities
    targets = []
    if getattr(args, "test_data_roots", None):
        targets = list(args.test_data_roots)
    elif getattr(args, "test_data_root", None):
        targets = [args.test_data_root]

    src_norm = os.path.normpath(args.data_root)
    targets = [t for t in targets if t and os.path.normpath(t) != src_norm]

    for troot in targets:
        if not troot:
            continue
        torch.cuda.empty_cache()
        args_one = copy.deepcopy(args)
        args_one.test_data_root = troot
        run_zeroshot(args_one)


if __name__ == "__main__":
    main()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
