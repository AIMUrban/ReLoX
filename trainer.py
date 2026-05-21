import numpy as np


def make_metrics_fn(grid_h, grid_w, topk=(1, 5, 10)):
    def _safe_topk_part(logits, k):
        C = logits.shape[1]
        k = int(min(k, C))
        if k <= 0:
            return None
        return np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]

    def _metrics_from_logits(prefix, logits, labels, mask=None):
        out = {}
        if logits is None:
            return out

        logits = np.asarray(logits)
        labels = np.asarray(labels)

        # Ensure logits are [B, C].
        if logits.ndim == 1:
            # Usually indicates non-logit values were passed by mistake.
            return out
        if logits.ndim != 2:
            raise ValueError(f"{prefix} logits must be 2D [B,C], got shape={logits.shape}")

        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            if mask.sum() == 0:
                return out
            logits = logits[mask]
            labels = labels[mask]

        B, C = logits.shape
        if B == 0:
            return out

        pred_ids = logits.argmax(axis=1)
        out[f"eval_{prefix}_acc@1"] = float((pred_ids == labels).mean())

        max_k = int(max(topk)) if len(topk) else 1
        part = _safe_topk_part(logits, max_k)
        if part is not None:
            for k in topk:
                k = int(k)
                if k <= 1:
                    continue
                kk = min(k, C)
                topk_idx = part[:, :kk]
                hit = (topk_idx == labels[:, None]).any(axis=1)
                out[f"eval_{prefix}_acc@{k}"] = float(hit.mean())

        true_scores = logits[np.arange(B), labels]
        ranks = 1 + (logits > true_scores[:, None]).sum(axis=1)
        out[f"eval_{prefix}_mrr"] = float((1.0 / ranks).mean())

        pred_y = pred_ids // grid_w
        pred_x = pred_ids % grid_w
        true_y = labels // grid_w
        true_x = labels % grid_w
        dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        out[f"eval_{prefix}_mde"] = float(dist.mean())

        return out

    def compute_metrics(p):
        preds = p.predictions  # dict from preprocess_logits_for_metrics
        labels = np.asarray(p.label_ids)

        final_logits = preds.get("final", None)
        abs_logits = preds.get("abs", None)
        rel_logits = preds.get("rel", None)
        p_oow = preds.get("p_oow", None)
        tloc_local = preds.get("tloc_local", None)

        out = {}
        out.update(_metrics_from_logits("final", final_logits, labels))
        out.update(_metrics_from_logits("abs", abs_logits, labels))
        out.update(_metrics_from_logits("rel", rel_logits, labels))

        # Subsets based on local target availability.
        if tloc_local is not None:
            tloc_local = np.asarray(tloc_local).reshape(-1)
            mask_oow = (tloc_local < 0)
            mask_in = ~mask_oow
            out["eval_oow_rate"] = float(mask_oow.mean())

            out.update(_metrics_from_logits("final_oow", final_logits, labels, mask=mask_oow))
            out.update(_metrics_from_logits("final_in", final_logits, labels, mask=mask_in))

            out.update(_metrics_from_logits("abs_oow", abs_logits, labels, mask=mask_oow))
            out.update(_metrics_from_logits("abs_in", abs_logits, labels, mask=mask_in))

            out.update(_metrics_from_logits("rel_oow", rel_logits, labels, mask=mask_oow))
            out.update(_metrics_from_logits("rel_in", rel_logits, labels, mask=mask_in))

        # p_oow diagnostics.
        if p_oow is not None:
            p_oow = np.asarray(p_oow).reshape(-1)
            out["eval_p_oow_mean"] = float(p_oow.mean())
            if tloc_local is not None:
                mask_oow = (np.asarray(tloc_local).reshape(-1) < 0)
                mask_in = ~mask_oow
                pred_oow = (p_oow >= 0.5)
                out["eval_oow_acc"] = float((pred_oow == mask_oow).mean())
                if mask_oow.sum() > 0:
                    out["eval_p_oow_oow_mean"] = float(p_oow[mask_oow].mean())
                if mask_in.sum() > 0:
                    out["eval_p_oow_in_mean"] = float(p_oow[mask_in].mean())

        return out

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # Convert tensors to numpy-friendly objects.
    out = {"final": logits["final"].detach().float().cpu().numpy()}

    if logits.get("abs", None) is not None:
        out["abs"] = logits["abs"].detach().float().cpu().numpy()
    if logits.get("rel", None) is not None:
        out["rel"] = logits["rel"].detach().float().cpu().numpy()

    # p_oow / tloc_local
    if logits.get("p_oow", None) is not None:
        out["p_oow"] = logits["p_oow"].detach().float().cpu().numpy().reshape(-1)
    if logits.get("tloc_local", None) is not None:
        out["tloc_local"] = logits["tloc_local"].detach().long().cpu().numpy().reshape(-1)

    return out
