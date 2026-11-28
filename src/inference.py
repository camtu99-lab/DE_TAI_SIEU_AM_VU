"""Bộ chuyển đổi PyTorch -> predict_fn của BUS-VERIFY. Cần torch (Kaggle GPU hoặc Windows CPU). Tái dùng nguyên module V3 đã kiểm thử.
`logits` trả về = log(prob) để T-scaling khớp ĐÚNG pipeline V3 (softmax(log p / T)); với TTA, prob là trung bình hai góc nhìn như predict_arrays V3."""
import numpy as np
from ._paths import ensure_v3
ensure_v3()


class TorchPredictor:
    def __init__(self, model, device, tta=False, batch_size=32, amp=False, meta=None):
        import torch
        self.torch = torch
        self.model, self.device, self.tta, self.bs, self.amp, self.meta = model.eval().to(device), device, bool(tta), batch_size, amp, meta or {}

    def __call__(self, images):
        import torch
        import torch.nn.functional as F
        import bus_core as core
        probs, segp, evid = [], [], []
        for i in range(0, len(images), self.bs):
            x = torch.stack([core.normalize_to_tensor(a) for a in images[i:i + self.bs]]).to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.amp):
                o = self.model(x)
                of = self.model(x.flip(-1)) if self.tta else None
            p = F.softmax(o['logits'].float(), 1)
            if self.tta:
                p = 0.5 * (p + F.softmax(of['logits'].float(), 1))
            probs.append(p.cpu().numpy())
            if o.get('seg') is not None:
                sp = torch.sigmoid(o['seg'].float())
                if self.tta:
                    sp = 0.5 * (sp + torch.sigmoid(of['seg'].float()).flip(-1))
                segp.append(sp.squeeze(1).cpu().numpy().astype(np.float32))
            if o.get('evidence') is not None:
                ev = o['evidence'].float()
                if self.tta:
                    ev = 0.5 * (ev + of['evidence'].float().flip(-1))
                    ev = ev / (ev.sum((2, 3), keepdim=True) + 1e-8)
                evid.append(ev.cpu().numpy()[:, 0])
        cat = lambda a: np.concatenate(a) if a else None
        P = cat(probs)
        return dict(logits=np.log(np.clip(P, 1e-12, 1.0)), seg_prob=cat(segp), evidence=cat(evid))

    def gradcam(self, img):
        import bus_core as core
        import bus_core_v2 as C2
        x = core.normalize_to_tensor(img).unsqueeze(0).to(self.device)
        cam, _ = C2.gradcam_dict(self.model, x)
        return cam


def load_predictor(ckpt_path, device, tta=None):
    import bus_core_v2 as C2
    import models_xai as MX
    model, meta = C2.load_model_from_checkpoint(ckpt_path, device)
    if tta is None:
        tta = bool(MX.EXPERIMENTS.get(meta.get('experiment'), {}).get('tta', False))
    return TorchPredictor(model, device, tta=tta, meta=meta)
