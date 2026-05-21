import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tools import resize_img


def crop_egocentric_numpy(img_stack, center_x, center_y, crop_size=64):
    """
    img_stack: [C, H, W] numpy
    return: [C, crop_size, crop_size]
    """
    C, H, W = img_stack.shape
    pad_h = crop_size // 2
    pad_w = crop_size // 2

    padded_img = np.pad(
        img_stack,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    new_cx = int(center_x) + pad_w
    new_cy = int(center_y) + pad_h

    x1 = new_cx - crop_size // 2
    y1 = new_cy - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    cropped_img = padded_img[:, y1:y2, x1:x2]

    if cropped_img.shape[1] != crop_size or cropped_img.shape[2] != crop_size:
        out = np.zeros((C, crop_size, crop_size), dtype=img_stack.dtype)
        h_c = min(crop_size, cropped_img.shape[1])
        w_c = min(crop_size, cropped_img.shape[2])
        out[:, :h_c, :w_c] = cropped_img[:, :h_c, :w_c]
        return out

    return cropped_img


class TrajDataset(Dataset):
    def __init__(
        self,
        seq_npz_path,
        traj_frame_path,
        user_frame_path,
        poi_map_path,
        grid_w=200,
        grid_h=200,
        rel_size=0,
        pixel_size=128,
        use_image_cache=True,
    ):
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.rel_size = int(rel_size)
        self.crop_size = int(self.rel_size)
        self.pixel_size = int(pixel_size)
        self.use_image_cache = bool(use_image_cache)

        # Enable local crop/label only when rel_size > 0
        self.crop = self.rel_size > 0

        # 1) check files
        for p in [seq_npz_path, traj_frame_path, user_frame_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing file: {p}")

        # 2) meta
        self.data = np.load(seq_npz_path, allow_pickle=True)
        self.uid = self.data["uid"]
        self.seq_time = self.data["seq_time"]
        self.seq_x = self.data["seq_x"]
        self.seq_y = self.data["seq_y"]
        self.seq_loc = self.data["seq_loc"]
        self.target_x = self.data["target_x"]
        self.target_y = self.data["target_y"]
        self.valid_len = self.data["valid_len"]
        if "target_loc" in self.data:
            self.target_loc = self.data["target_loc"]
        else:
            self.target_loc = self.target_y.astype(np.int64) * self.grid_w + self.target_x.astype(np.int64)

        # 3) frames
        self.traj_frames = np.load(traj_frame_path, allow_pickle=True)["frames"]  # [N,H,W] G
        self.user_frames = np.load(user_frame_path, allow_pickle=True)["frames"]  # [N_user,H,W] R

        if Path(poi_map_path).exists():
            self.poi_map = np.load(poi_map_path)  # [H,W] B
        else:
            print(f"[Warning] POI map not found at {poi_map_path}, using zeros.")
            self.poi_map = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        assert len(self.traj_frames) == len(self.uid), \
            f"Mismatch: traj_frames({len(self.traj_frames)}) != meta({len(self.uid)})"

        # Optional: load precomputed image caches (local crop + resized global), to avoid per-epoch recompute.
        self._cache_local = None
        self._cache_abs = None
        if self.use_image_cache:
            self._try_load_image_cache(
                seq_npz_path=Path(seq_npz_path),
                traj_frame_path=Path(traj_frame_path),
                user_frame_path=Path(user_frame_path),
                poi_map_path=Path(poi_map_path),
            )

    def __len__(self):
        return int(self.uid.shape[0])

    def _cache_key(self, split: str) -> str:
        return (
            f"{split}_gw{self.grid_w}_gh{self.grid_h}_"
            f"rel{self.rel_size}_pix{self.pixel_size}_v1"
        )

    def _cache_paths(self, split: str, cache_dir: Path):
        key = self._cache_key(split)
        return {
            "meta": cache_dir / f"img_cache_{key}.json",
            "local": cache_dir / f"img_cache_local_{key}.npy",
            "abs": cache_dir / f"img_cache_abs_{key}.npy",
        }

    def _try_load_image_cache(self, seq_npz_path: Path, traj_frame_path: Path, user_frame_path: Path, poi_map_path: Path):
        split = seq_npz_path.stem
        cache_dir = seq_npz_path.parent
        paths = self._cache_paths(split, cache_dir)
        if not (paths["meta"].exists() and paths["local"].exists() and paths["abs"].exists()):
            return

        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        except Exception:
            return

        try:
            if int(meta.get("version", -1)) != 1:
                return
            if int(meta.get("n_samples", -1)) != int(len(self)):
                return
            if int(meta.get("pixel_size", -1)) != int(self.pixel_size):
                return
            if int(meta.get("rel_size", -1)) != int(self.rel_size):
                return
            if int(meta.get("grid_w", -1)) != int(self.grid_w) or int(meta.get("grid_h", -1)) != int(self.grid_h):
                return

            cache_mtime = float(meta.get("cache_mtime", 0.0) or 0.0)
            src_mtimes = [
                seq_npz_path.stat().st_mtime if seq_npz_path.exists() else 0.0,
                traj_frame_path.stat().st_mtime if traj_frame_path.exists() else 0.0,
                user_frame_path.stat().st_mtime if user_frame_path.exists() else 0.0,
                poi_map_path.stat().st_mtime if poi_map_path.exists() else 0.0,
            ]
            if max(src_mtimes) > cache_mtime + 1e-6:
                return

            self._cache_local = np.load(paths["local"], mmap_mode="r")
            self._cache_abs = np.load(paths["abs"], mmap_mode="r")

            exp_shape = (len(self), 3, self.pixel_size, self.pixel_size)
            if tuple(getattr(self._cache_local, "shape", ())) != exp_shape:
                self._cache_local = None
            if tuple(getattr(self._cache_abs, "shape", ())) != exp_shape:
                self._cache_abs = None
        except Exception:
            self._cache_local = None
            self._cache_abs = None

    def __getitem__(self, i):
        uid_i = int(self.uid[i])
        seq_time_i = self.seq_time[i].astype(np.int64, copy=False)
        seq_x_i = self.seq_x[i].astype(np.int64, copy=False)
        seq_y_i = self.seq_y[i].astype(np.int64, copy=False)
        seq_loc_i = self.seq_loc[i].astype(np.int64, copy=False) if self.seq_loc is not None else None
        valid_len_i = int(self.valid_len[i])

        # target (global)
        target_x_i = int(self.target_x[i])
        target_y_i = int(self.target_y[i])
        target_loc_i = int(self.target_loc[i]) if np.ndim(self.target_loc) > 0 else int(target_y_i * self.grid_w + target_x_i)

        # current position (global)
        cur_x = int(seq_x_i[valid_len_i - 1])
        cur_y = int(seq_y_i[valid_len_i - 1])

        cache_ok = (self._cache_local is not None) and (self._cache_abs is not None)
        if cache_ok:
            # np.load(..., mmap_mode="r") returns read-only arrays; PyTorch warns on non-writable numpy views.
            # Copy is cheap vs crop/resize and avoids undefined behavior if any later in-place op happens.
            traj_img_local = torch.from_numpy(np.array(self._cache_local[i], copy=True)).float().div_(255.0)
            abs_img_global = torch.from_numpy(np.array(self._cache_abs[i], copy=True)).float().div_(255.0)
        else:
            img_g = self.traj_frames[i]               # [H,W]
            img_r = self.user_frames[uid_i]           # [H,W]
            img_b = self.poi_map                      # [H,W]

            g_stack_global = np.stack([img_r, img_g, img_b], axis=0)

            abs_img_global = np.stack([img_r, img_g, np.zeros_like(img_b)], axis=0)  # [3,H,W]
            abs_img_global = torch.from_numpy(abs_img_global).float() / 255.0  # [3,H,W]
            abs_img_global = resize_img(abs_img_global, self.pixel_size)

            if self.crop:
                traj_img_local_np = crop_egocentric_numpy(
                    g_stack_global, cur_x, cur_y, crop_size=self.crop_size
                )  # [3,K,K]
                traj_img_local = torch.from_numpy(traj_img_local_np).float() / 255.0  # [3,K,K]
                traj_img_local = resize_img(traj_img_local, out=self.pixel_size)  # [3,pix,pix]
            else:
                traj_img_local = torch.from_numpy(g_stack_global).float() / 255.0  # [3,H,W]
                traj_img_local = resize_img(traj_img_local, out=self.pixel_size)   # [3,pix,pix]

        # ---- build local (crop-coords) target label ----
        if self.crop:
            dx_t = target_x_i - cur_x
            dy_t = target_y_i - cur_y

            K = self.rel_size
            c = K // 2

            target_x_local = int(dx_t + c)
            target_y_local = int(dy_t + c)

            target_local_valid = (0 <= target_x_local < K) and (0 <= target_y_local < K)
            if target_local_valid:
                target_loc_local = target_y_local * K + target_x_local
            else:
                target_loc_local = -1  # ignore_index
        else:
            target_loc_local = -1
            target_local_valid = False

        out = {
            "uid": uid_i,
            "seq_time": seq_time_i,
            "seq_x": seq_x_i,
            "seq_y": seq_y_i,
            "seq_loc": seq_loc_i,
            "valid_len": valid_len_i,

            "target_x": target_x_i,
            "target_y": target_y_i,
            "target_loc": target_loc_i,

            "traj_img_local": traj_img_local,
            "abs_img_global": abs_img_global,

            "target_loc_local": int(target_loc_local),
            "target_local_valid": bool(target_local_valid),
        }

        return out


def auto_paths(root, split):
    seq_npz = root / f"{split}.npz"
    traj_frame_npz = root / f"{split}_traj_frames.npz"
    user_frame_npz = root / "user_frames.npz"
    poi_map_npy = root / "poi_map.npy"

    if seq_npz.exists() and traj_frame_npz.exists() and user_frame_npz.exists():
        return {
            "seq": seq_npz,
            "traj_frame": traj_frame_npz,
            "user_frame": user_frame_npz,
            "poi_map": poi_map_npy,
        }
    return None


def build_datasets(args):
    root = Path(args.data_root)
    paths = {sp: auto_paths(root, sp) for sp in ("train", "val", "test")}
    if paths["train"] is None:
        raise FileNotFoundError(f"Missing required training files under {root}")

    rel_sz = int(getattr(args, "rel_size", 8))
    pix = int(getattr(args, "pixel_size", 128))

    ds_train = TrajDataset(
        paths["train"]["seq"],
        paths["train"]["traj_frame"],
        paths["train"]["user_frame"],
        paths["train"]["poi_map"],
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        rel_size=rel_sz,
        pixel_size=pix,
        use_image_cache=True,
    )

    ds_val = TrajDataset(
        paths["val"]["seq"],
        paths["val"]["traj_frame"],
        paths["val"]["user_frame"],
        paths["val"]["poi_map"],
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        rel_size=rel_sz,
        pixel_size=pix,
        use_image_cache=True,
    ) if paths["val"] else None

    ds_test = TrajDataset(
        paths["test"]["seq"],
        paths["test"]["traj_frame"],
        paths["test"]["user_frame"],
        paths["test"]["poi_map"],
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        rel_size=rel_sz,
        pixel_size=pix,
        use_image_cache=True,
    ) if paths["test"] else None

    return ds_train, ds_val, ds_test, paths


def make_collator():
    def to_tensor(arr):
        return torch.from_numpy(arr).long()

    def collate(samples):
        lengths = [int(b["valid_len"]) for b in samples]

        seq_time_list = [to_tensor(b["seq_time"]) for b in samples]
        seq_x_list = [to_tensor(b["seq_x"]) for b in samples]
        seq_y_list = [to_tensor(b["seq_y"]) for b in samples]
        seq_loc_list = [to_tensor(b["seq_loc"]) for b in samples]

        seq_time = pad_sequence(seq_time_list, batch_first=True, padding_value=-1)
        seq_x = pad_sequence(seq_x_list, batch_first=True, padding_value=-1)
        seq_y = pad_sequence(seq_y_list, batch_first=True, padding_value=-1)
        seq_loc = pad_sequence(seq_loc_list, batch_first=True, padding_value=-1)

        B, T_max = seq_time.size(0), seq_time.size(1)
        seq_mask = torch.zeros((B, T_max), dtype=torch.bool)
        for i, L in enumerate(lengths):
            seq_mask[i, :L] = True

        traj_img_local = torch.stack([b["traj_img_local"] for b in samples], dim=0)    # [B,3,pix,pix]
        abs_img_global = torch.stack([b["abs_img_global"] for b in samples], dim=0)   # [B,3,pix,pix]

        uid = torch.tensor([b["uid"] for b in samples], dtype=torch.long)
        valid_len = torch.tensor(lengths, dtype=torch.long)
        target_x = torch.tensor([b["target_x"] for b in samples], dtype=torch.long)
        target_y = torch.tensor([b["target_y"] for b in samples], dtype=torch.long)
        target_loc = torch.tensor([b["target_loc"] for b in samples], dtype=torch.long)

        target_loc_local = torch.tensor([b["target_loc_local"] for b in samples], dtype=torch.long)
        target_local_valid = torch.tensor([b["target_local_valid"] for b in samples], dtype=torch.bool)

        batch = {
            "uid": uid,
            "seq_time": seq_time,
            "seq_x": seq_x,
            "seq_y": seq_y,
            "seq_loc": seq_loc,
            "seq_mask": seq_mask,
            "valid_len": valid_len,

            "target_x": target_x,
            "target_y": target_y,
            "target_loc": target_loc,

            "traj_img_local": traj_img_local,
            "abs_img_global": abs_img_global,

            "target_loc_local": target_loc_local,
            "target_local_valid": target_local_valid,
        }

        return batch

    return collate


def precompute_image_cache(
    data_root: Path,
    split: str,
    grid_w: int,
    grid_h: int,
    rel_size: int,
    pixel_size: int = 128,
    overwrite: bool = False,
):
    paths = auto_paths(data_root, split)
    if paths is None:
        raise FileNotFoundError(f"Missing required files under {data_root} for split='{split}'")

    ds = TrajDataset(
        paths["seq"],
        paths["traj_frame"],
        paths["user_frame"],
        paths["poi_map"],
        grid_w=grid_w,
        grid_h=grid_h,
        rel_size=rel_size,
        pixel_size=pixel_size,
        use_image_cache=False,
    )

    out_paths = ds._cache_paths(split, data_root)
    if (not overwrite) and out_paths["meta"].exists() and out_paths["local"].exists() and out_paths["abs"].exists():
        print(f"[Skip] cache exists: {out_paths['meta']}")
        return out_paths

    N = len(ds)
    shape = (N, 3, pixel_size, pixel_size)
    local_mm = np.lib.format.open_memmap(out_paths["local"], mode="w+", dtype=np.uint8, shape=shape)
    abs_mm = np.lib.format.open_memmap(out_paths["abs"], mode="w+", dtype=np.uint8, shape=shape)

    for i in range(N):
        sample = ds[i]
        local_u8 = sample["traj_img_local"].clamp(0, 1).mul(255).byte().numpy()
        abs_u8 = sample["abs_img_global"].clamp(0, 1).mul(255).byte().numpy()
        local_mm[i] = local_u8
        abs_mm[i] = abs_u8
        if (i + 1) % 2000 == 0:
            print(f"[Cache] {split}: {i+1}/{N}")

    src_mtimes = [
        Path(paths["seq"]).stat().st_mtime,
        Path(paths["traj_frame"]).stat().st_mtime,
        Path(paths["user_frame"]).stat().st_mtime,
    ]
    if Path(paths["poi_map"]).exists():
        src_mtimes.append(Path(paths["poi_map"]).stat().st_mtime)

    meta = {
        "version": 1,
        "split": split,
        "n_samples": int(N),
        "grid_w": int(grid_w),
        "grid_h": int(grid_h),
        "rel_size": int(rel_size),
        "pixel_size": int(pixel_size),
        "cache_mtime": float(max(src_mtimes)),
    }
    out_paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved cache meta: {out_paths['meta']}")
    return out_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Precompute image caches (local crop + resized global) to avoid recomputing crop/resize each epoch."
    )
    ap.add_argument("--data_root", type=str, required=True, help="dataset root containing train/val/test npz + frames")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--grid_w", type=int, default=200)
    ap.add_argument("--grid_h", type=int, default=200)
    ap.add_argument("--rel_size", type=int, default=8)
    ap.add_argument("--pixel_size", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    root = Path(args.data_root)
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        precompute_image_cache(
            data_root=root,
            split=sp,
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            rel_size=args.rel_size,
            pixel_size=args.pixel_size,
            overwrite=args.overwrite,
        )


