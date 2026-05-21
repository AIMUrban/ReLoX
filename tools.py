import os
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def resize_img(patch_chw, out):
    """
    patch_chw: [C,H,W], float tensor, usually in [0,1]
    """
    x = patch_chw.unsqueeze(0)        # [1,C,H,W]
    x = F.interpolate(
        x,
        size=(out, out),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )
    return x.squeeze(0)               # [C,out,out]


def format_metrics(metrics):
    """Round float values in metrics to 4 decimals."""
    out = {}
    for k, v in metrics.items():
        # Convert tensors/scalars to Python values first.
        val = v.item() if hasattr(v, "item") else v

        # Keep only 4 decimals for floats.
        if isinstance(val, float):
            out[k] = round(val, 4)
        else:
            out[k] = val
    return out


def read_len(json_path: Path) -> int:
    """Read mapping/list length from a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return int(len(obj))
    elif isinstance(obj, list):
        return int(len(obj))
    else:
        raise ValueError(f"Unsupported json format in {json_path}")


def infer_class_counts_from_maps(root_dir: str):
    root = Path(root_dir)
    uid_map = root / "uid_map.json"
    loc_map = root / "loc_map.json"
    n_user = read_len(uid_map)
    n_loc = read_len(loc_map)

    return {
        "n_user": n_user,
        "n_loc": n_loc,
    }


_GRID_WH_RE = re.compile(r"(\\d+)x(\\d+)", re.IGNORECASE)
_GRID_WH_RE = re.compile(r"(\\d+)x(\\d+)", re.IGNORECASE)


def _read_json_any_encoding(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def infer_grid_wh(data_root: str):
    root = Path(data_root)
    stats_path = root / "stats.json"
    if stats_path.exists():
        try:
            meta = _read_json_any_encoding(stats_path)
            if isinstance(meta, dict):
                gw = meta.get("grid_w", None)
                gh = meta.get("grid_h", None)
                if gw is not None and gh is not None:
                    try:
                        gw_i = int(gw)
                        gh_i = int(gh)
                        if gw_i > 0 and gh_i > 0:
                            return gw_i, gh_i
                    except Exception:
                        pass

                for k in ("csv", "city"):
                    s = str(meta.get(k, "") or "")
                    m = _GRID_WH_RE.search(s)
                    if m:
                        try:
                            gw_i = int(m.group(1))
                            gh_i = int(m.group(2))
                            if gw_i > 0 and gh_i > 0:
                                return gw_i, gh_i
                        except Exception:
                            pass
        except Exception:
            pass

    m = _GRID_WH_RE.search(str(root.name))
    if m:
        try:
            gw_i = int(m.group(1))
            gh_i = int(m.group(2))
            if gw_i > 0 and gh_i > 0:
                return gw_i, gh_i
        except Exception:
            pass

    poi_map = root / "poi_map.npy"
    if poi_map.exists():
        try:
            arr = np.load(poi_map)
            if getattr(arr, "ndim", 0) == 2:
                gh_i, gw_i = int(arr.shape[0]), int(arr.shape[1])
                if gw_i > 0 and gh_i > 0:
                    return gw_i, gh_i
        except Exception:
            pass

    for sp in ("train", "val", "test"):
        fp = root / f"{sp}_traj_frames.npz"
        if not fp.exists():
            continue
        try:
            with np.load(fp, allow_pickle=True) as z:
                if "frames" not in getattr(z, "files", ()):
                    continue
                frames = z["frames"]
                if getattr(frames, "ndim", 0) == 3:
                    gh_i, gw_i = int(frames.shape[1]), int(frames.shape[2])
                    if gw_i > 0 and gh_i > 0:
                        return gw_i, gh_i
        except Exception:
            pass

    return None


def resolve_grid_wh(data_root: str, grid_w: int, grid_h: int):
    inferred = infer_grid_wh(data_root)
    if inferred is None:
        return int(grid_w), int(grid_h)
    gw_i, gh_i = inferred
    if int(grid_w) != int(gw_i) or int(grid_h) != int(gh_i):
        print(f"[Grid] Override grid_w/h: {grid_w}x{grid_h} -> {gw_i}x{gh_i} (from {data_root})")
    return int(gw_i), int(gh_i)


def resolve_out_dir(args) -> str:
    out_dir = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir:
        if os.path.isabs(out_dir):
            return out_dir
        return os.path.join(args.data_root, out_dir)
    return os.path.join(args.data_root, "outputs")


def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def prefix_metrics(metrics, prefix):
    out = {}
    for k, v in metrics.items():
        if k.startswith("eval_"):
            out[f"{prefix}_{k[5:]}"] = v
        else:
            out[f"{prefix}_{k}"] = v
    return out


def append_city_report(out_dir: str, filename: str, text: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    return out_path


def format_short_metrics(metrics, prefix):
    def is_percent_metric(name: str) -> bool:
        if not name:
            return False
        return (
            ("acc@" in name)
            or name.endswith("_mrr")
            or name.endswith("mrr")
            or ("oow_rate" in name)
            or ("oow_acc" in name)
            or ("p_oow_" in name)
        )

    def fmt_metric(name, v):
        if isinstance(v, (float, int, np.floating, np.integer)):
            vv = float(v)
            if is_percent_metric(name):
                return f"{vv * 100:.2f}"
            return f"{vv:.4f}"
        return str(v)

    def pick_named(*names):
        for name in names:
            if name and name in metrics:
                return name, metrics[name]
        return None, None

    def build_head_line(head, tag, include_loss=False):
        mrr_name, mrr = pick_named(f"{prefix}_{head}_mrr", f"{prefix}_mrr" if head == "final" else None)
        acc1_name, acc1 = pick_named(f"{prefix}_{head}_acc@1", f"{prefix}_acc@1" if head == "final" else None)
        acc5_name, acc5 = pick_named(f"{prefix}_{head}_acc@5")
        acc10_name, acc10 = pick_named(f"{prefix}_{head}_acc@10")
        mde_name, mde = pick_named(f"{prefix}_{head}_mde")

        if mrr is None and acc1 is None and mde is None:
            return None

        parts = []
        if include_loss:
            loss_name, loss_val = pick_named(f"{prefix}_loss")
            if loss_val is not None:
                parts.append(f"loss={fmt_metric(loss_name, loss_val)}")

        if mrr is not None:
            parts.append(f"{tag}mrr={fmt_metric(mrr_name, mrr)}")
        if acc1 is not None:
            parts.append(f"{tag}acc1={fmt_metric(acc1_name, acc1)}")
        if acc5 is not None:
            parts.append(f"{tag}acc5={fmt_metric(acc5_name, acc5)}")
        if acc10 is not None:
            parts.append(f"{tag}acc10={fmt_metric(acc10_name, acc10)}")
        if mde is not None:
            parts.append(f"{tag}mde={fmt_metric(mde_name, mde)}")

        return " ".join(parts) if parts else None

    def has_head_metrics(head):
        keys = [
            f"{prefix}_{head}_mrr",
            f"{prefix}_{head}_acc@1",
            f"{prefix}_{head}_acc@5",
            f"{prefix}_{head}_acc@10",
            f"{prefix}_{head}_mde",
        ]
        return any(k in metrics for k in keys)

    has_abs = has_head_metrics("abs")
    has_rel = has_head_metrics("rel")

    lines = []
    use_final = not (has_abs ^ has_rel)

    oow_name, oow = pick_named(f"{prefix}_oow_rate")
    oow_acc_name, oow_acc = pick_named(f"{prefix}_oow_acc")

    if use_final:
        final_line = build_head_line("final", "", include_loss=True)
        if final_line:
            if oow is not None:
                final_line = f"{final_line} oow={fmt_metric(oow_name, oow)}"
            if oow_acc is not None:
                final_line = f"{final_line} oow_acc={fmt_metric(oow_acc_name, oow_acc)}"
            lines.append(final_line)

    rel_line = build_head_line("rel", "rel_", include_loss=not use_final and has_rel and not has_abs)
    if rel_line and (not use_final) and oow is not None:
        rel_line = f"{rel_line} oow={fmt_metric(oow_name, oow)}"
    if rel_line and (not use_final) and oow_acc is not None:
        rel_line = f"{rel_line} oow_acc={fmt_metric(oow_acc_name, oow_acc)}"
    if rel_line:
        lines.append(rel_line)

    abs_line = build_head_line("abs", "abs_", include_loss=not use_final and has_abs and not has_rel)
    if abs_line and (not use_final) and oow is not None:
        abs_line = f"{abs_line} oow={fmt_metric(oow_name, oow)}"
    if abs_line and (not use_final) and oow_acc is not None:
        abs_line = f"{abs_line} oow_acc={fmt_metric(oow_acc_name, oow_acc)}"
    if abs_line:
        lines.append(abs_line)

    return "\n".join(lines)


def evaluate(model, dataloader, args, device, desc):
    model.eval()

    topk = (1, 5, 10)

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = torch.zeros((), device=device, dtype=torch.long)
    reg_seen = False
    reg_acc1 = torch.zeros((), device=device, dtype=torch.long)
    reg_mde = torch.zeros((), device=device, dtype=torch.float64)
    reg_n = torch.zeros((), device=device, dtype=torch.long)

    def new_acc():
        return {
            "n": torch.zeros((), device=device, dtype=torch.long),
            "acc1": torch.zeros((), device=device, dtype=torch.long),
            "acc5": torch.zeros((), device=device, dtype=torch.long),
            "acc10": torch.zeros((), device=device, dtype=torch.long),
            "mrr": torch.zeros((), device=device, dtype=torch.float64),
            "mde": torch.zeros((), device=device, dtype=torch.float64),
        }

    acc = {
        "final": new_acc(),
        "abs": new_acc(),
        "rel": new_acc(),
        "final_oow": new_acc(),
        "final_in": new_acc(),
        "abs_oow": new_acc(),
        "abs_in": new_acc(),
        "rel_oow": new_acc(),
        "rel_in": new_acc(),
    }

    oow_n = torch.zeros((), device=device, dtype=torch.long)
    p_oow_sum = torch.zeros((), device=device, dtype=torch.float64)
    p_oow_oow_sum = torch.zeros((), device=device, dtype=torch.float64)
    p_oow_in_sum = torch.zeros((), device=device, dtype=torch.float64)
    p_oow_oow_n = torch.zeros((), device=device, dtype=torch.long)
    p_oow_in_n = torch.zeros((), device=device, dtype=torch.long)
    oow_acc_correct = torch.zeros((), device=device, dtype=torch.long)
    saw_p_oow = False

    def update_logits_metrics(bucket, logits, labels, mask=None):
        if logits is None:
            return
        if not isinstance(logits, torch.Tensor):
            return
        if logits.ndim != 2:
            return

        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            logits = logits[mask]
            labels = labels[mask]

        if logits.numel() == 0:
            return

        if logits.dtype not in (torch.float32, torch.float64):
            logits = logits.float()

        B, C = logits.shape
        bucket["n"] += B

        pred_ids = logits.argmax(dim=1)
        bucket["acc1"] += (pred_ids == labels).sum()

        kk10 = min(10, C)
        top10 = logits.topk(kk10, dim=1).indices
        if 5 in topk:
            top5 = top10[:, : min(5, kk10)]
            bucket["acc5"] += (top5 == labels[:, None]).any(dim=1).sum()
        if 10 in topk:
            bucket["acc10"] += (top10 == labels[:, None]).any(dim=1).sum()

        true_scores = logits.gather(1, labels[:, None]).squeeze(1)
        ranks = 1 + (logits > true_scores[:, None]).sum(dim=1)
        bucket["mrr"] += (1.0 / ranks.to(dtype=torch.float64)).sum()

        pred_y = pred_ids // args.grid_w
        pred_x = pred_ids % args.grid_w
        true_y = labels // args.grid_w
        true_x = labels % args.grid_w
        dist = torch.sqrt((pred_x - true_x).float().pow(2) + (pred_y - true_y).float().pow(2))
        bucket["mde"] += dist.to(dtype=torch.float64).sum()

    def finalize(bucket_prefix, bucket):
        n = bucket["n"]
        if n.item() == 0:
            return {}
        n_f = n.to(dtype=torch.float64)
        out = {f"eval_{bucket_prefix}_acc@1": (bucket["acc1"].to(dtype=torch.float64) / n_f).item(),
               f"eval_{bucket_prefix}_mrr": (bucket["mrr"] / n_f).item(),
               f"eval_{bucket_prefix}_mde": (bucket["mde"] / n_f).item(),
               f"eval_{bucket_prefix}_acc@5": (bucket["acc5"].to(dtype=torch.float64) / n_f).item(),
               f"eval_{bucket_prefix}_acc@10": (bucket["acc10"].to(dtype=torch.float64) / n_f).item()}
        return out

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, device)
            out = model(**batch)
            loss = out["loss"]
            bs = batch["target_loc"].size(0)
            total_loss += loss.detach().to(dtype=torch.float64) * bs
            total_samples += bs

            pred_xy = out.get("pred_xy", None)
            if pred_xy is not None:
                reg_seen = True
                pred_xy = pred_xy.to(device=device, dtype=torch.float32)
                true_x = batch["target_x"].to(device=device, dtype=torch.float32)
                true_y = batch["target_y"].to(device=device, dtype=torch.float32)

                pred_x_id = pred_xy[:, 0].round().clamp(0, int(args.grid_w) - 1).to(dtype=torch.long)
                pred_y_id = pred_xy[:, 1].round().clamp(0, int(args.grid_h) - 1).to(dtype=torch.long)
                pred_id = pred_y_id * int(args.grid_w) + pred_x_id
                reg_acc1 += (pred_id == batch["target_loc"]).sum()

                dist = torch.sqrt((pred_xy[:, 0] - true_x).pow(2) + (pred_xy[:, 1] - true_y).pow(2))
                reg_mde += dist.to(dtype=torch.float64).sum()
                reg_n += bs
                continue

            labels = batch["target_loc"]
            logits_final = out.get("logits")
            logits_abs = out.get("logits_abs")
            logits_rel = out.get("logits_rel")

            tloc_local = out.get("target_loc_local")
            mask_oow = None
            mask_in = None
            if tloc_local is not None:
                tloc_local = tloc_local.reshape(-1)
                mask_oow = tloc_local < 0
                mask_in = ~mask_oow
                oow_n += mask_oow.sum()

            update_logits_metrics(acc["final"], logits_final, labels)
            update_logits_metrics(acc["abs"], logits_abs, labels)
            update_logits_metrics(acc["rel"], logits_rel, labels)

            if mask_oow is not None:
                update_logits_metrics(acc["final_oow"], logits_final, labels, mask=mask_oow)
                update_logits_metrics(acc["final_in"], logits_final, labels, mask=mask_in)
                update_logits_metrics(acc["abs_oow"], logits_abs, labels, mask=mask_oow)
                update_logits_metrics(acc["abs_in"], logits_abs, labels, mask=mask_in)
                update_logits_metrics(acc["rel_oow"], logits_rel, labels, mask=mask_oow)
                update_logits_metrics(acc["rel_in"], logits_rel, labels, mask=mask_in)

            p_oow = out.get("p_oow")
            if p_oow is not None:
                saw_p_oow = True
                p_oow = p_oow.reshape(-1).to(dtype=torch.float64)
                p_oow_sum += p_oow.sum()
                if mask_oow is not None:
                    mask_oow_flat = mask_oow.reshape(-1)
                    mask_in_flat = mask_in.reshape(-1)

                    if mask_oow_flat.any().item():
                        p_oow_oow_sum += p_oow[mask_oow_flat].sum()
                        p_oow_oow_n += mask_oow_flat.sum()
                    if mask_in_flat.any().item():
                        p_oow_in_sum += p_oow[mask_in_flat].sum()
                        p_oow_in_n += mask_in_flat.sum()

                    pred_oow = (p_oow >= 0.5)
                    oow_acc_correct += (pred_oow.to(torch.bool) == mask_oow_flat).sum()

    total_samples_item = total_samples.item()
    avg_loss = (total_loss / max(total_samples_item, 1)).item()
    metrics = {"eval_loss": float(avg_loss)}

    if reg_seen:
        n = reg_n.item()
        if n > 0:
            metrics["eval_final_acc@1"] = (reg_acc1.to(dtype=torch.float64) / reg_n.to(dtype=torch.float64)).item()
            metrics["eval_final_mde"] = (reg_mde / reg_n.to(dtype=torch.float64)).item()
        return metrics

    metrics.update(finalize("final", acc["final"]))
    metrics.update(finalize("abs", acc["abs"]))
    metrics.update(finalize("rel", acc["rel"]))

    if (acc["final_oow"]["n"].item() + acc["final_in"]["n"].item()) > 0:
        metrics.update(finalize("final_oow", acc["final_oow"]))
        metrics.update(finalize("final_in", acc["final_in"]))
        metrics.update(finalize("abs_oow", acc["abs_oow"]))
        metrics.update(finalize("abs_in", acc["abs_in"]))
        metrics.update(finalize("rel_oow", acc["rel_oow"]))
        metrics.update(finalize("rel_in", acc["rel_in"]))
        metrics["eval_oow_rate"] = (oow_n.to(dtype=torch.float64) / max(total_samples_item, 1)).item()

    if saw_p_oow:
        metrics["eval_p_oow_mean"] = (p_oow_sum / max(total_samples_item, 1)).item()
        if p_oow_oow_n.item() > 0:
            metrics["eval_p_oow_oow_mean"] = (p_oow_oow_sum / p_oow_oow_n.to(dtype=torch.float64)).item()
        if p_oow_in_n.item() > 0:
            metrics["eval_p_oow_in_mean"] = (p_oow_in_sum / p_oow_in_n.to(dtype=torch.float64)).item()
        if total_samples_item > 0 and (acc["final_oow"]["n"].item() + acc["final_in"]["n"].item()) > 0:
            metrics["eval_oow_acc"] = (oow_acc_correct.to(dtype=torch.float64) / max(total_samples_item, 1)).item()

    return metrics

