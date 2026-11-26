# bus_core.py - module dùng chung cho huấn luyện, đánh giá và ứng dụng
import os, csv
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------- Tiền xử lý -------------------------
def ensure_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        mx = float(img.max()) if img.size else 1.0
        img = (img.astype(np.float32) / (mx if mx > 0 else 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    return img


def letterbox_square(arr, pad_value=0):
    h, w = arr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    canvas = np.full((s, s) + arr.shape[2:], pad_value, dtype=arr.dtype)
    canvas[top:top + h, left:left + w] = arr
    return canvas


def preprocess_image(img_rgb, size):
    sq = letterbox_square(ensure_rgb_uint8(img_rgb))
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(sq, (size, size), interpolation=interp)


def preprocess_mask(mask_bin, size):
    sq = letterbox_square((np.asarray(mask_bin) > 0).astype(np.uint8) * 255)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    r = cv2.resize(sq, (size, size), interpolation=interp)
    return (r > 127).astype(np.uint8)


def normalize_to_tensor(img_uint8):
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# ------------------------- Mô hình -------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if drop > 0:
            layers.append(nn.Dropout2d(drop))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    # g: tín hiệu gating (đã upsample, cùng kích thước với x); x: đặc trưng skip-connection
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.last_alpha = None

    def forward(self, g, x):
        a = self.relu(self.w_g(g) + self.w_x(x))
        alpha = self.psi(a)
        self.last_alpha = alpha.detach()
        return x * alpha


class UNetFamily(nn.Module):
    # attention=False -> U-Net cơ bản; attention=True -> Attention U-Net
    # num_cls>0 -> thêm đầu phân loại trên bottleneck (mô hình đa nhiệm); forward trả (seg_logits, cls_logits)
    def __init__(self, in_ch=3, base=32, attention=True, num_cls=0, cls_drop=0.3):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.attention = attention
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_ch, ch[0])
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])
        self.bott = ConvBlock(ch[3], ch[4], drop=0.2)
        self.ups = nn.ModuleList()
        self.atts = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in [4, 3, 2, 1]:
            self.ups.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            if attention:
                self.atts.append(AttentionGate(ch[i - 1], ch[i - 1], max(ch[i - 1] // 2, 8)))
            self.decs.append(ConvBlock(ch[i - 1] * 2, ch[i - 1]))
        self.out = nn.Conv2d(ch[0], 1, 1)
        self.cls_head = nn.Sequential(nn.Dropout(cls_drop), nn.Linear(ch[4], num_cls)) if num_cls > 0 else None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d = b
        for k, skip in enumerate([e4, e3, e2, e1]):
            d = self.ups[k](d)
            if self.attention:
                skip = self.atts[k](d, skip)
            d = self.decs[k](torch.cat([d, skip], dim=1))
        seg = self.out(d)
        if self.cls_head is not None:
            return seg, self.cls_head(b.mean(dim=(2, 3)))
        return seg

    def attention_maps(self):
        return [a.last_alpha for a in self.atts] if self.attention else []


def build_effnet_b3(num_classes, pretrained=True):
    loaded = False
    if pretrained:
        try:
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
            m = torchvision.models.efficientnet_b3(weights=weights)
            loaded = True
        except Exception as e:
            print('[CẢNH BÁO] Không tải được trọng số pretrained (' + type(e).__name__ + ': ' + str(e)[:150] + ').')
            print('           EfficientNet-B3 sẽ được khởi tạo NGẪU NHIÊN -> không còn là transfer learning.')
            print('           Bật Internet trên Kaggle (Settings) rồi chạy lại nếu muốn dùng pretrained.')
            m = torchvision.models.efficientnet_b3(weights=None)
    else:
        m = torchvision.models.efficientnet_b3(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_f, num_classes))
    return m, loaded


class FusionClassifier(nn.Module):
    # E4: EfficientNet-B3 nhận 4 kênh = RGB + xác suất phân vùng do seg_net (Attention U-Net, đóng băng) dự đoán.
    # Không dùng mask ground-truth khi suy luận. Kênh thứ 4 khởi tạo trọng số 0 nên lúc bắt đầu mạng tương đương E0.
    def __init__(self, num_classes, seg_base=32, pretrained=False):
        super().__init__()
        self.seg_net = UNetFamily(3, seg_base, attention=True)
        for p in self.seg_net.parameters():
            p.requires_grad = False
        net, self.pretrained_loaded = build_effnet_b3(num_classes, pretrained)
        conv = net.features[0][0]
        new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = conv.weight
        net.features[0][0] = new
        self.cls_net = net
        self.last_seg_prob = None

    @property
    def features(self):
        return self.cls_net.features

    def train(self, mode=True):
        super().train(mode)
        self.seg_net.eval()          # phân vùng luôn ở chế độ eval (BN, dropout cố định)
        return self

    def forward(self, x):
        with torch.no_grad():
            sp = torch.sigmoid(self.seg_net(x).float())
        self.last_seg_prob = sp.detach()
        return self.cls_net(torch.cat([x, sp.to(x.dtype)], dim=1))


def build_model(arch, num_classes, base=32, pretrained=False):
    # trả về (model, pretrained_loaded)
    if arch == 'effnet_b3':
        return build_effnet_b3(num_classes, pretrained)
    if arch == 'fusion':
        m = FusionClassifier(num_classes, seg_base=base, pretrained=pretrained)
        return m, m.pretrained_loaded
    if arch == 'unet':
        return UNetFamily(3, base, attention=False), False
    if arch == 'attunet':
        return UNetFamily(3, base, attention=True), False
    if arch == 'attunet_mt':
        return UNetFamily(3, base, attention=True, num_cls=num_classes), False
    raise ValueError('arch không hợp lệ: ' + str(arch))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ------------------------- Checkpoint -------------------------
def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)   # chỉ dùng cho file do chính notebook tạo
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path, model, meta):
    torch.save({'model': model.state_dict(), 'meta': meta}, path)


def load_checkpoint(path, device):
    ck = torch_load(path, device)
    meta = ck['meta']
    model, _ = build_model(meta['arch'], meta['num_classes'], base=meta.get('base', 32), pretrained=False)
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), meta


# ------------------------- Grad-CAM -------------------------
def gradcam(model, x, class_idx=None, target_layer=None):
    # x: tensor (1,3,H,W). Mặc định dùng khối đặc trưng cuối của EfficientNet (model.features[-1]).
    layer = target_layer if target_layer is not None else model.features[-1]
    store = {}

    def fwd_hook(module, inp, out):
        store['a'] = out
        out.register_hook(lambda g: store.__setitem__('g', g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.enable_grad():
            xin = x.detach().clone().requires_grad_(True)
            logits = model(xin)
            logits = logits[1] if isinstance(logits, tuple) else logits
            probs = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy()
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        acts = store['a'].detach()
        grads = store['g'].detach()
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), probs, idx
    finally:
        handle.remove()


# ------------------------- Detection từ mask -------------------------
def mask_to_boxes(mask, min_area=100):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))   # (x1, y1, x2, y2), x2/y2 không bao gồm
    return boxes


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------- Suy luận ảnh mới -------------------------
def predict_array(img_rgb, cls_model, seg_model, size, device, thr=0.5, min_area=100, with_cam=True):
    img = preprocess_image(img_rgb, size)
    x = normalize_to_tensor(img).unsqueeze(0).to(device)
    res = {'image': img, 'probs': None, 'pred_idx': None, 'cam': None,
           'seg_prob': None, 'mask': None, 'boxes': []}
    if cls_model is not None:
        cls_model.eval()
        with torch.no_grad():
            out = cls_model(x)
            logits = out[1] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), 1)[0].cpu().numpy()
        res['probs'] = probs
        res['pred_idx'] = int(probs.argmax())
        if with_cam and hasattr(cls_model, 'features'):
            cam, _, _ = gradcam(cls_model, x, class_idx=res['pred_idx'])
            res['cam'] = cam
    if seg_model is not None:
        seg_model.eval()
        with torch.no_grad():
            out = seg_model(x)
            seg = out[0] if isinstance(out, tuple) else out
            sp = torch.sigmoid(seg.float())[0, 0].cpu().numpy()
        res['seg_prob'] = sp
        res['mask'] = (sp >= thr).astype(np.uint8)
        res['boxes'] = mask_to_boxes(res['mask'], min_area)
    return res


# ------------------------- Hiển thị -------------------------
def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.4):
    out = img.copy()
    m = np.asarray(mask) > 0
    if m.any():
        out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    out = np.ascontiguousarray(img.copy())
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), color, thickness)
    return out


def overlay_cam(img, cam, alpha=0.45):
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return ((1 - alpha) * img + alpha * heat).astype(np.uint8)


def visualize_prediction(res, class_names, title=None):
    panels = [('Ảnh đã tiền xử lý', res['image'])]
    if res['mask'] is not None:
        panels.append(('Mask dự đoán', overlay_mask(res['image'], res['mask'])))
        panels.append((f"Bounding box ({len(res['boxes'])})", draw_boxes(res['image'], res['boxes'])))
    if res['cam'] is not None:
        panels.append(('Grad-CAM', overlay_cam(res['image'], res['cam'])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    if res['probs'] is not None:
        txt = ' | '.join(f'{c}: {p * 100:.1f}%' for c, p in zip(class_names, res['probs']))
        head = f"Dự đoán: {class_names[res['pred_idx']]}  ({txt})"
    else:
        head = ''
    fig.suptitle((title + '\n' if title else '') + head, fontsize=11)
    fig.tight_layout()
    return fig


def log_prediction(log_path, filename, class_names, res):
    header = ['timestamp', 'filename', 'predicted_class'] + ['prob_' + c for c in class_names] + \
             ['mask_area_ratio', 'n_boxes', 'boxes_xyxy']
    new = not os.path.exists(log_path)
    probs = res['probs'] if res['probs'] is not None else [float('nan')] * len(class_names)
    area = float(res['mask'].mean()) if res['mask'] is not None else float('nan')
    row = [datetime.now().isoformat(timespec='seconds'), filename,
           class_names[res['pred_idx']] if res['pred_idx'] is not None else ''] + \
          [f'{float(p):.4f}' for p in probs] + [f'{area:.5f}', len(res['boxes']), str(res['boxes'])]
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)