# models_xai.py - BreastVision-XAI V3: SCSE encoder/decoder, BI-RADS posterior zone, evidence coupling toàn diện & phân rã logit
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from bus_core import build_effnet_b3, UNetFamily
from losses_xai import lesion_ring, posterior_zone, EPS


# ============================ Khối chú ý nhẹ & Encoder/Decoder ============================
class SCSEBlock(nn.Module):
    # Concurrent Spatial and Channel Squeeze & Excitation (nhẹ, tăng chất lượng biên tổn thương nhỏ)
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(4, ch // r)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class EffNetB3Encoder(nn.Module):
    TAPS = (1, 2, 3, 5, 8)
    CH = {1: 24, 2: 32, 3: 48, 5: 136, 8: 1536}

    def __init__(self, pretrained=True):
        super().__init__()
        net, self.pretrained_loaded = build_effnet_b3(2, pretrained)
        self.features = net.features

    def forward(self, x):
        out = {}
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in self.TAPS:
                out[i] = x
        return out


def conv_block(cin, cout, use_scse=False):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if use_scse:
        layers.append(SCSEBlock(cout))
    return nn.Sequential(*layers)


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class SegDecoder(nn.Module):
    def __init__(self, ch=(256, 128, 64, 32), use_scse=False):
        super().__init__()
        C = EffNetB3Encoder.CH
        self.b4 = conv_block(C[8] + C[5], ch[0], use_scse=use_scse)
        self.b3 = conv_block(ch[0] + C[3], ch[1], use_scse=use_scse)
        self.b2 = conv_block(ch[1] + C[2], ch[2], use_scse=use_scse)
        self.b1 = conv_block(ch[2] + C[1], ch[3], use_scse=use_scse)
        self.out = nn.Conv2d(ch[3], 1, 1)

    def forward(self, f, out_size):
        d = self.b4(torch.cat([_up(f[8], f[5]), f[5]], 1))
        d = self.b3(torch.cat([_up(d, f[3]), f[3]], 1))
        d = self.b2(torch.cat([_up(d, f[2]), f[2]], 1))
        d = self.b1(torch.cat([_up(d, f[1]), f[1]], 1))
        return F.interpolate(self.out(d), size=out_size, mode='bilinear', align_corners=False)


class Neck(nn.Module):
    def __init__(self, dim, use_scse=False):
        super().__init__()
        C = EffNetB3Encoder.CH
        self.l3, self.l5, self.l8 = nn.Conv2d(C[3], dim, 1), nn.Conv2d(C[5], dim, 1), nn.Conv2d(C[8], dim, 1)
        fuse_layers = [nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)]
        if use_scse:
            fuse_layers.append(SCSEBlock(dim))
        self.fuse = nn.Sequential(*fuse_layers)

    def forward(self, f):
        z = self.l3(f[3]) + _up(self.l5(f[5]), f[3]) + _up(self.l8(f[8]), f[3])
        return self.fuse(z)


# ============================ Gộp đa vùng ============================
def masked_avg_pool(Z, m, eps=1e-3):
    return (Z * m).sum((2, 3)) / (m.sum((2, 3)) + eps)


def evidence_zone_weights(Eu, zone_mask, eps=1e-3):
    # V3: trọng số không gian trong vùng giải phẫu dựa trên evidence Eu (B,1,h,w).
    # Nếu zone_mask rỗng -> w = 0 (hữu hạn); nếu khác rỗng -> chuẩn hóa tổng theo không gian (có +eps ổn định).
    raw = Eu * zone_mask
    return raw / (raw.sum((2, 3), keepdim=True) + eps)


class RegionFusion(nn.Module):
    def __init__(self, mode, n_regions, dim):
        super().__init__()
        self.mode, self.n, self.dim = (mode if n_regions > 1 else 'concat'), n_regions, dim
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_regions)])
        if self.mode == 'weighted':
            self.a = nn.Parameter(torch.zeros(n_regions))
        if self.mode == 'gated':
            self.gate = nn.Linear(n_regions * dim, n_regions * dim)
        self.out_dim = n_regions * dim if self.mode == 'concat' else dim

    def forward(self, xs):
        v = [p(x) for p, x in zip(self.proj, xs)]
        if self.mode == 'concat':
            return torch.cat(v, 1)
        if self.mode == 'weighted':
            w = F.softmax(self.a, 0)
            return sum(wi * vi for wi, vi in zip(w, v))
        g = F.softmax(self.gate(torch.cat(xs, 1)).view(-1, self.n, self.dim), 1)
        return (g * torch.stack(v, 1)).sum(1)


# ============================ Mô hình chính ============================
@dataclass
class XAIConfig:
    num_classes: int = 2
    dim: int = 128
    fusion: str = 'concat'          # concat | weighted | gated
    use_lesion: bool = True
    use_peri: bool = True
    evidence_mode: str = 'ewp'      # none | ewp | egr | decoupled
    ring_cells: int = 2
    detach_mask: bool = True
    dropout: float = 0.3
    pretrained: bool = True
    evidence_tau: float = 1.0
    # --- Nâng cấp V3 (mặc định False để X0-X6 giữ nguyên 100%) ---
    use_scse: bool = False
    use_posterior: bool = False
    post_depth: int = 6
    evidence_coupling_all: bool = False


class BreastVisionXAI(nn.Module):
    # Suy luận CHỈ dùng ảnh; mọi vùng giải phẫu (lesion, peri, posterior) được suy ra từ mask DỰ ĐOÁN.
    def __init__(self, cfg: XAIConfig):
        super().__init__()
        assert cfg.evidence_mode in ('none', 'ewp', 'egr', 'decoupled')
        self.cfg = cfg
        self.enc = EffNetB3Encoder(cfg.pretrained)
        self.pretrained_loaded = self.enc.pretrained_loaded
        self.dec = SegDecoder(use_scse=cfg.use_scse)
        self.neck = Neck(cfg.dim, use_scse=cfg.use_scse)
        n = 1 + int(cfg.use_lesion) + int(cfg.use_peri) + int(cfg.use_posterior)
        self.fusion = RegionFusion(cfg.fusion, n, cfg.dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(self.fusion.out_dim, cfg.num_classes))
        if cfg.evidence_mode != 'none':
            self.ev_head = nn.Sequential(nn.Conv2d(cfg.dim, cfg.dim // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(cfg.dim // 2, 1, 1))

    def _compute_regions(self, Z, m, ring, post, E, ev_mask=None):
        cfg = self.cfg
        if cfg.evidence_mode == 'none' or cfg.evidence_mode == 'decoupled':
            g = Z.mean((2, 3))
            Eu = None
            wg = torch.ones_like(m) / (m.shape[-2] * m.shape[-1])
        else:
            Eu = E if ev_mask is None else E * ev_mask
            wg = Eu if cfg.evidence_mode == 'ewp' else Eu / (Z.shape[-2] * Z.shape[-1])
            g = (Z * Eu).sum((2, 3)) if cfg.evidence_mode == 'ewp' else (Z * Eu).mean((2, 3))
        regions = [g]
        weights = [wg]
        zone_masks = []
        if cfg.use_lesion:
            zone_masks.append(m)
        if cfg.use_peri:
            zone_masks.append(ring)
        if cfg.use_posterior:
            zone_masks.append(post)
        for zm in zone_masks:
            if cfg.evidence_coupling_all and Eu is not None:
                wz = evidence_zone_weights(Eu, zm)
                regions.append((Z * wz).sum((2, 3)))
                weights.append(wz)
            else:
                regions.append(masked_avg_pool(Z, zm))
                weights.append(zm / (zm.sum((2, 3), keepdim=True) + 1e-3))
        return regions, weights

    def forward_heads_from_Z(self, Z, m, ring, post, E, ev_mask=None):
        # Dùng trong counterfactual faithfulness loss: chạy lại phần pooling + fusion + classifier mà KHÔNG chạy lại encoder/decoder
        regions, weights = self._compute_regions(Z, m, ring, post, E, ev_mask=ev_mask)
        h_vec = self.fusion(regions)
        return self.head(h_vec), regions, weights

    def forward(self, x, ev_mask=None):
        cfg = self.cfg
        f = self.enc(x)
        seg = self.dec(f, x.shape[-2:])
        Z = self.neck(f)
        m = F.adaptive_avg_pool2d(torch.sigmoid(seg.float()), Z.shape[-2:])
        if cfg.detach_mask:
            m = m.detach()
        E = None
        if cfg.evidence_mode != 'none':
            s = self.ev_head(Z).float() / cfg.evidence_tau
            B, _, h, w = s.shape
            if cfg.evidence_mode in ('ewp', 'decoupled'):
                E = F.softmax(s.flatten(2), -1).view(B, 1, h, w)
            else:
                E = torch.sigmoid(s)
        ring = lesion_ring(m, cfg.ring_cells) if cfg.use_peri else None
        post = posterior_zone(m, cfg.post_depth) if cfg.use_posterior else None
        regions, weights = self._compute_regions(Z, m, ring, post, E, ev_mask=ev_mask)
        logits = self.head(self.fusion(regions))
        return dict(seg=seg, logits=logits, evidence=E, mask_feat=m, ring=ring, post=post, Z=Z, regions=regions, region_weights=weights)

    @torch.no_grad()
    def infer(self, x):
        self.eval()
        o = self.forward(x)
        return dict(prob=F.softmax(o['logits'].float(), 1), seg_prob=torch.sigmoid(o['seg'].float()), evidence=o['evidence'])

    @torch.no_grad()
    def decompose_logits(self, x):
        # Phân rã chính xác logit:
        # - Khi evidence_coupling_all=False (V2): C_k(x) = E(x)*(W_g(A_g Z(x) + b_g))_k, rest_k chứa các vùng còn lại + bias.
        # - Khi evidence_coupling_all=True  (V3): TẤT CẢ các vùng đều đi qua weights(x) dựa trên E(x), nên
        #   C_k(x) = sum_i w_i(x) * [W_i A_i Z(x)]_k và rest_k = bias_k + sum_i [W_i b_i]_k (hằng số thuần túy!).
        assert self.cfg.evidence_mode == 'ewp' and self.fusion.mode == 'concat', 'phân rã chính xác chỉ có cho ewp + concat'
        self.eval()
        o = self.forward(x)
        D = self.cfg.dim
        W, bias = self.head[1].weight, self.head[1].bias
        if not self.cfg.evidence_coupling_all:
            A, b = self.fusion.proj[0].weight, self.fusion.proj[0].bias
            Wg = W[:, :D]
            C = o['evidence'] * (torch.einsum('kd,bdhw->bkhw', Wg @ A, o['Z']) + (Wg @ b).view(1, -1, 1, 1))
            rest = bias.expand(x.shape[0], -1).clone()
            for i, r in enumerate(o['regions'][1:], start=1):
                v = self.fusion.proj[i](r)
                rest = rest + v @ W[:, i * D:(i + 1) * D].T
            return C, rest, o['logits']

        C = torch.zeros(x.shape[0], self.cfg.num_classes, o['Z'].shape[2], o['Z'].shape[3], device=x.device, dtype=o['Z'].dtype)
        rest = bias.expand(x.shape[0], -1).clone()
        for i, w_map in enumerate(o['region_weights']):
            Ai, bi = self.fusion.proj[i].weight, self.fusion.proj[i].bias
            Wi = W[:, i * D:(i + 1) * D]
            C = C + w_map * torch.einsum('kd,bdhw->bkhw', Wi @ Ai, o['Z'])
            rest = rest + (Wi @ bi).view(1, -1)
        return C, rest, o['logits']


# ============================ Bọc mô hình cũ (E0–E4) để dùng chung engine ============================
class LegacyWrapper(nn.Module):
    def __init__(self, model, task, pretrained_loaded=None):
        super().__init__()
        self.model, self.task = model, task
        self.pretrained_loaded = getattr(model, 'pretrained_loaded', False) if pretrained_loaded is None else bool(pretrained_loaded)

    def forward(self, x, ev_mask=None):
        o = self.model(x)
        if self.task == 'cls':
            return dict(seg=None, logits=o, evidence=None)
        if self.task == 'seg':
            return dict(seg=o, logits=None, evidence=None)
        return dict(seg=o[0], logits=o[1], evidence=None)


# ============================ Danh mục thực nghiệm ============================
EXPERIMENTS = {
    # --- baseline gốc ---
    'E0': dict(kind='legacy', arch='effnet_b3', task='cls', desc='EfficientNet-B3 phân loại'),
    'E1': dict(kind='legacy', arch='unet', task='seg', desc='U-Net'),
    'E2': dict(kind='legacy', arch='attunet', task='seg', desc='Attention U-Net'),
    'E3': dict(kind='legacy', arch='attunet_mt', task='mt', desc='Attention U-Net đa nhiệm'),
    'E4': dict(kind='legacy', arch='attunet', task='seg', augment=False, desc='Attention U-Net không augmentation (ablation)'),
    # --- chuỗi X (V2) ---
    'X0': dict(kind='xai', model=dict(use_lesion=False, use_peri=False, evidence_mode='none'), loss={}, desc='Encoder chung đa nhiệm (baseline)'),
    'X1': dict(kind='xai', model=dict(use_lesion=True, use_peri=False, evidence_mode='none'), loss={}, desc='X0 + gộp global/lesion'),
    'X2': dict(kind='xai', model=dict(use_lesion=True, use_peri=True, evidence_mode='none'), loss={}, desc='X1 + vùng quanh tổn thương'),
    'X3': dict(kind='xai', model=dict(evidence_mode='ewp'), loss={}, desc='X2 + evidence-guided classification (EWP)'),
    'X4': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(expl_mode='lesion_context'), desc='X3 + explanation constraint'),
    'X5': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(lam_cons_pred=0.5, lam_cons_ev=0.5), desc='X3 + consistency'),
    'X6': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(expl_mode='lesion_context', lam_cons_pred=0.5, lam_cons_ev=0.5), desc='X3 + constraint + consistency'),
    # --- chuỗi V3 (nâng cấp tính mới & hiệu năng) ---
    'V3_base': dict(kind='xai', model=dict(evidence_mode='ewp', use_scse=True),
                    loss=dict(expl_mode='lesion_context', lam_cons_pred=0.5, lam_cons_ev=0.5, label_smoothing=0.05, lam_edge=0.5),
                    ema=True, tta=True, desc='V3 Base: X6 + SCSE + boundary loss + label smoothing + EMA'),
    'V3_birads': dict(kind='xai', model=dict(evidence_mode='ewp', use_scse=True, use_posterior=True, evidence_coupling_all=True),
                      loss=dict(expl_mode='lesion_context', lam_cons_pred=0.5, lam_cons_ev=0.5, label_smoothing=0.05, lam_edge=0.5),
                      ema=True, tta=True, desc='V3 BI-RADS: + vùng hậu âm siêu âm + evidence coupling toàn vùng'),
    'V3_full': dict(kind='xai', model=dict(evidence_mode='ewp', use_scse=True, use_posterior=True, evidence_coupling_all=True),
                    loss=dict(expl_mode='lesion_context', lam_cons_pred=0.5, lam_cons_ev=0.5, label_smoothing=0.05, lam_edge=0.5, lam_faith=0.3),
                    ema=True, tta=True, desc='V3 Full: + Counterfactual Faithfulness Loss (đề xuất chính V3)'),
    # --- ablation bổ sung ---
    'A_fusion_weighted': dict(kind='xai', model=dict(evidence_mode='none', fusion='weighted'), loss={}, desc='X2 với weighted fusion'),
    'A_fusion_gated': dict(kind='xai', model=dict(evidence_mode='none', fusion='gated'), loss={}, desc='X2 với gated fusion'),
    'A_evidence_egr': dict(kind='xai', model=dict(evidence_mode='egr'), loss={}, desc='X3 formulation đối chứng: evidence-gated features (EGR)'),
    'A_evidence_decoupled': dict(kind='xai', model=dict(evidence_mode='decoupled'), loss=dict(expl_mode='lesion_context'), desc='Evidence KHÔNG nối vào logit (kiểm tra vai trò coupling)'),
    'A_exp_mask_only': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(expl_mode='mask_only'), desc='X3 + ràng buộc mask-only'),
    'A_exp_density': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(expl_mode='density'), desc='X3 + ràng buộc density-based'),
    'A_cons_pred_only': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(lam_cons_pred=0.5), desc='X3 + chỉ prediction consistency'),
    'A_cons_ev_only': dict(kind='xai', model=dict(evidence_mode='ewp'), loss=dict(lam_cons_ev=0.5), desc='X3 + chỉ evidence consistency'),
}
MAIN_CONFIGS = ['X0', 'X3', 'X4', 'X6', 'V3_full']      # cấu hình chính: chạy nhiều seed


def build_experiment(name, num_classes=2, pretrained=True, dim=128, base=32):
    spec = EXPERIMENTS[name]
    if spec['kind'] == 'xai':
        cfg = XAIConfig(num_classes=num_classes, pretrained=pretrained, dim=dim, **spec['model'])
        return BreastVisionXAI(cfg), asdict(cfg)
    from bus_core import build_model
    model, loaded = build_model(spec['arch'], num_classes, base=base, pretrained=pretrained)
    return LegacyWrapper(model, spec['task'], pretrained_loaded=loaded), dict(arch=spec['arch'], task=spec['task'], base=base)
