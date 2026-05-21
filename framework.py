# framework.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_modules import TransEncoder, TokenAttn
from custom_resnet import CNNEncoder


class ReLoX(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.grid_h = int(args.grid_h)
        self.grid_w = int(args.grid_w)
        self.HW = self.grid_h * self.grid_w

        self.lambda_abs = float(getattr(args, "lambda_abs", 1.0))
        self.lambda_rel = float(getattr(args, "lambda_rel", 1.0))
        self.drop_r_p = float(getattr(args, "drop_r_p", 0.0))
        self.zero_shot = bool(getattr(args, "zero_shot", False))

        self.match_dim = int(getattr(args, "match_dim", 256))
        self.rel_size = int(getattr(args, "rel_size", 8))

        self.vision_local = CNNEncoder(out_dim=self.match_dim, c=8)
        self.vision_global = CNNEncoder(out_dim=self.match_dim, c=8)

        lt_dim = self.match_dim // 2
        lt_hid_dim = lt_dim // 2
        self.spatial_mlp_rel = nn.Sequential(
            nn.Linear(5, lt_hid_dim),
            nn.LayerNorm(lt_hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(lt_hid_dim, lt_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(3, lt_hid_dim),
            nn.LayerNorm(lt_hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(lt_hid_dim, lt_dim),
        )
        self.seq_encoder = TransEncoder(input_dim=self.match_dim)

        head_in_dim_ego = self.match_dim * 2
        head_in_dim_global = self.match_dim
        self.global_head = nn.Sequential(
            nn.Linear(head_in_dim_global, head_in_dim_global),
            nn.LayerNorm(head_in_dim_global),
            nn.LeakyReLU(0.2),
            nn.Linear(head_in_dim_global, self.HW),
        )
        self.local_head = nn.Sequential(
            nn.Linear(head_in_dim_ego, head_in_dim_ego),
            nn.LayerNorm(head_in_dim_ego),
            nn.LeakyReLU(0.2),
            nn.Linear(head_in_dim_ego, self.rel_size * self.rel_size + 1),
        )

        self.global_from_seq = TokenAttn(dim=self.match_dim, num_heads=1, attn_drop=0, proj_drop=0, pre_ln=False)
        self.local_from_seq = TokenAttn(dim=self.match_dim, num_heads=1, attn_drop=0, proj_drop=0, pre_ln=False)
        self.seq_from_local = TokenAttn(dim=self.match_dim, num_heads=1, attn_drop=0, proj_drop=0, pre_ln=False)

        gate_hid = self.match_dim // 2
        self.global_local_gate = nn.Sequential(
            nn.Linear(self.match_dim * 2, gate_hid),
            nn.Linear(gate_hid, 1),
            nn.Sigmoid(),
        )
        self.local_seq_gate = nn.Sequential(
            nn.Linear(self.match_dim * 2, gate_hid),
            nn.Linear(gate_hid, 1),
            nn.Sigmoid(),
        )
        self.seq_local_gate = nn.Sequential(
            nn.Linear(self.match_dim * 2, gate_hid),
            nn.Linear(gate_hid, 1),
            nn.Sigmoid(),
        )

    def encode_sequence(self, seq_x, seq_y, seq_time, seq_mask, valid_len):
        B, _ = seq_x.shape

        curr_x = torch.gather(seq_x, 1, (valid_len - 1).unsqueeze(1)).squeeze(1)
        curr_y = torch.gather(seq_y, 1, (valid_len - 1).unsqueeze(1)).squeeze(1)
        cur_t = torch.gather(seq_time, 1, (valid_len - 1).unsqueeze(1)).float()

        delta_t = (cur_t - seq_time.float()).clamp(min=0.0)
        period = 24.0
        dt_norm = torch.log1p(delta_t) / math.log(period)

        hour = seq_time.float()
        omega = 2.0 * math.pi / period
        sin_t = torch.sin(omega * hour)
        cos_t = torch.cos(omega * hour)
        time_feat = torch.stack([dt_norm, sin_t, cos_t], dim=-1)
        time_emb = self.time_mlp(time_feat)

        dx_hist = seq_x.float() - curr_x.float().view(B, 1)
        dy_hist = seq_y.float() - curr_y.float().view(B, 1)
        dist = torch.sqrt(dx_hist ** 2 + dy_hist ** 2)
        angle = torch.atan2(dy_hist, dx_hist)
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        spatial_feat_rel = torch.stack([dx_hist, dy_hist, dist, sin_a, cos_a], dim=-1)

        spatial_feat_rel = spatial_feat_rel.masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        loc_emb = self.spatial_mlp_rel(spatial_feat_rel)
        time_emb = time_emb.masked_fill(~seq_mask.unsqueeze(-1), 0.0)
        seq_emb = torch.cat([loc_emb, time_emb], dim=-1)

        seq_out = self.seq_encoder(seq_emb, src_key_padding_mask=~seq_mask, causal_mask=None)
        return seq_out, curr_x, curr_y

    def forward(self, **batch):
        device = batch["seq_x"].device
        seq_x = batch["seq_x"]
        seq_y = batch["seq_y"]
        seq_time = batch["seq_time"]
        seq_mask = batch["seq_mask"]
        valid_len = batch["valid_len"]

        traj_img_local = batch["traj_img_local"]
        abs_img_global = batch["abs_img_global"]
        B = seq_x.size(0)

        target_loc = batch["target_loc"]
        target_loc_local = batch["target_loc_local"].long()

        seq_states, curr_x, curr_y = self.encode_sequence(seq_x, seq_y, seq_time, seq_mask, valid_len)

        vis_global = self.vision_global(abs_img_global)
        vis_global_patches = vis_global["tokens"]

        if self.zero_shot:
            traj_img_local = traj_img_local.clone()
            traj_img_local[:, 0] = 0.0
        elif self.training and self.drop_r_p > 0.0:
            keep_p = max(0.0, min(1.0, 1.0 - self.drop_r_p))
            gate = torch.bernoulli(torch.full((B, 1, 1), keep_p, device=traj_img_local.device))
            if gate.min().item() < 1.0:
                traj_img_local = traj_img_local.clone()
                traj_img_local[:, 0] = traj_img_local[:, 0] * gate

        vis_local = self.vision_local(traj_img_local)
        vis_local_patches = vis_local["tokens"]

        local_update = self.local_from_seq(vis_local_patches, seq_states, key_padding_mask=~seq_mask)
        gate_local = self.local_seq_gate(torch.cat([vis_local_patches, local_update], dim=-1))
        local_tokens = vis_local_patches + gate_local * local_update

        seq_update = self.seq_from_local(seq_states, vis_local_patches)
        gate_seq = self.seq_local_gate(torch.cat([seq_states, seq_update], dim=-1))
        gate_seq = gate_seq * seq_mask.unsqueeze(-1)
        seq_tokens = seq_states + gate_seq * seq_update

        local_pooled = local_tokens.mean(dim=1)
        seq_mask_f = seq_mask.float()
        denom = seq_mask_f.sum(dim=1, keepdim=True)
        seq_pooled = (seq_tokens * seq_mask_f.unsqueeze(-1)).sum(dim=1) / denom

        global_update_seq = self.global_from_seq(vis_global_patches, seq_tokens, key_padding_mask=~seq_mask)
        gate_global = self.global_local_gate(torch.cat([vis_global_patches, global_update_seq], dim=-1))
        global_tokens_seq = vis_global_patches + gate_global * global_update_seq
        global_pooled = global_tokens_seq.mean(dim=1)

        loss = torch.tensor(0.0, device=device)
        logits_rel = None
        logits_abs = None
        logits_rel_global = None

        ego_in = torch.cat([local_pooled, seq_pooled], dim=-1)
        global_in = global_pooled

        if self.lambda_rel > 0:
            logits_rel = self.local_head(ego_in)
            K = self.rel_size
            c = K // 2
            rel_logits = logits_rel[:, : K * K]
            oow_logits = logits_rel[:, K * K :]

            grid_y, grid_x = torch.meshgrid(
                torch.arange(K, device=device),
                torch.arange(K, device=device),
                indexing="ij",
            )
            gx = (grid_x - c).unsqueeze(0) + curr_x.view(B, 1, 1)
            gy = (grid_y - c).unsqueeze(0) + curr_y.view(B, 1, 1)
            valid = (gx >= 0) & (gx < self.grid_w) & (gy >= 0) & (gy < self.grid_h)
            valid_flat = valid.view(B, -1)
            neg_inf = torch.finfo(rel_logits.dtype).min
            rel_logits = rel_logits.masked_fill(~valid_flat, neg_inf)
            logits_rel = torch.cat([rel_logits, oow_logits], dim=1)
            p_oow = torch.softmax(logits_rel, dim=-1)[:, -1]
            tgt_rel = torch.where(
                target_loc_local >= 0,
                target_loc_local,
                torch.full_like(target_loc_local, self.rel_size * self.rel_size),
            )

        if self.lambda_abs > 0:
            logits_abs = self.global_head(global_in)

        if logits_rel is not None:
            loss_rel = F.cross_entropy(logits_rel, tgt_rel)
            loss = loss + self.lambda_rel * loss_rel

        if logits_abs is not None:
            loss_abs = F.cross_entropy(logits_abs, target_loc)
            loss = loss + self.lambda_abs * loss_abs

        if self.training:
            return {"loss": loss}

        if self.lambda_rel > 0:
            rel_flat = logits_rel[:, : self.rel_size * self.rel_size]
            logits_rel_global = self._restore_rel_to_global_translate_only(rel_flat, curr_x, curr_y, K=self.rel_size)

        use_abs = self.lambda_abs > 0
        use_rel = self.lambda_rel > 0

        if use_abs and (not use_rel):
            final_logits = logits_abs
        elif use_rel and (not use_abs):
            final_logits = logits_rel_global
        else:
            eps = 1e-6
            log_poow = torch.log(p_oow.clamp_min(eps)).unsqueeze(1)
            log_pinx = torch.log((1.0 - p_oow).clamp_min(eps)).unsqueeze(1)
            final_logits = torch.logsumexp(
                torch.stack([logits_abs + log_poow, logits_rel_global + log_pinx], dim=0),
                dim=0,
            )

        return {
            "loss": loss,
            "logits": final_logits,
            "logits_abs": logits_abs if use_abs else None,
            "logits_rel": logits_rel_global if use_rel else None,
            "p_oow": p_oow if use_rel else None,
            "target_loc_local": target_loc_local,
        }

    def _restore_rel_to_global_translate_only(self, rel_logits_flat, curr_x, curr_y, K=None):
        if curr_x.dim() == 2:
            curr_x = curr_x.squeeze(1)
        if curr_y.dim() == 2:
            curr_y = curr_y.squeeze(1)

        B = rel_logits_flat.size(0)
        device = rel_logits_flat.device
        H, W = self.grid_h, self.grid_w
        if K is None:
            K = self.rel_size
        c = K // 2

        global_logits = torch.full((B, H * W), float("-inf"), device=device, dtype=rel_logits_flat.dtype)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(K, device=device),
            torch.arange(K, device=device),
            indexing="ij",
        )
        gx = (grid_x - c).unsqueeze(0) + curr_x.view(B, 1, 1)
        gy = (grid_y - c).unsqueeze(0) + curr_y.view(B, 1, 1)

        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)

        batch_off = torch.arange(B, device=device).view(B, 1, 1) * (H * W)
        flat_idx = batch_off + gy * W + gx

        valid_idx = flat_idx[valid]
        valid_val = rel_logits_flat.view(B, K, K)[valid]
        global_logits.view(-1).index_put_((valid_idx,), valid_val)
        return global_logits
