import inspect
import numpy as np
import cv2
import pytest
from src import counterfactual_verifier as CF
from src import segmentation as SG
from src import preprocessing as PP
from tests.synth import make_us_image, lesion_model, background_shortcut_model, center_mask


def _setup(seed=0, **kw):
    rng = np.random.default_rng(seed)
    img, gt = make_us_image(rng, **kw)
    return img, center_mask(), rng


def test_regions_disjoint_and_cover_valid():
    img, m, _ = _setup(pad=20)
    valid = PP.valid_content_mask(img)
    r = SG.build_regions(m, valid, ring_px=12)
    assert r.ok and SG.regions_are_disjoint(r)
    assert (r.lesion | r.peri | r.background).sum() == valid.sum()       # hợp = vùng hợp lệ
    assert not (r.lesion & ~valid).any() and not (r.peri & ~valid).any()


def test_counterfactual_region_matching_area_location_intensity():
    """Đối chứng matched-area: CÙNG diện tích tuyệt đối, không giao tổn thương/vùng cấm, nằm trong vùng hợp lệ, cường độ gần."""
    img, m, rng = _setup(1, pad=16)
    valid = PP.valid_content_mask(img)
    reg = SG.build_regions(m, valid, 12)
    gray = img.astype(np.float32).mean(2)
    forb = cv2.dilate(reg.lesion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    ctrls, info = CF.matched_area_controls(reg.lesion, forb, valid, gray, rng, k=3, n_candidates=60, depth_tol=0.10)
    assert len(ctrls) >= 1                                            # cùng độ sâu: chỉ còn khe hẹp hai bên tổn thương lớn (đúng kỳ vọng)
    for c, i in zip(ctrls, info):
        assert abs(i['dy']) <= int(0.10 * 256)                        # cùng độ sâu (dung sai)
    ctrls, info = CF.matched_area_controls(reg.lesion, forb, valid, gray, rng, k=3, n_candidates=60, depth_tol=None)
    assert len(ctrls) == 3                                            # nới độ sâu (verify_image tự làm khi thiếu)
    for c, i in zip(ctrls, info):
        assert c.sum() == reg.lesion.sum()                            # diện tích khớp chính xác
        assert not (c & forb).any() and not (c & ~valid).any()
    assert not (ctrls[0] & ctrls[1]).any()                            # đối chứng không chồng nhau
    # khớp cường độ: chọn trong nhiều ứng viên cho khoảng cách cường độ nhỏ hơn hẳn chọn ngẫu nhiên (1 ứng viên)
    many = np.mean([CF.matched_area_controls(reg.lesion, forb, valid, gray, np.random.default_rng(s), 1, 60, None)[1][0]['intensity_gap'] for s in range(15)])
    one = np.mean([CF.matched_area_controls(reg.lesion, forb, valid, gray, np.random.default_rng(s), 1, 1, None)[1][0]['intensity_gap'] for s in range(15)])
    assert many < one
    rnd, _ = CF.random_controls(int(reg.lesion.sum()), forb, valid, gray, rng, k=3)
    assert all(not (c & forb).any() for c in rnd)


def test_lesion_sensitive_model_has_positive_specificity():
    img, m, rng = _setup(2)
    res = CF.verify_image(img, m, lesion_model(), T=1.0, cfg=CF.CFConfig(seed=0), rng=np.random.default_rng(0))
    f = res['features']
    assert res['status'] == 'ok'
    assert f['lesion_sens'] > 3 * f['matched_les_sens']               # tổn thương nhạy hơn đối chứng cùng diện tích
    assert f['cf_specificity'] > 0.3 and f['cf_z'] > 2
    assert f['lesion_bg_contrast'] > 0


def test_background_shortcut_model_is_exposed():
    """Mô hình đi tắt vào nền: nhạy với nền, KHÔNG nhạy tổn thương -> lesion_bg_contrast âm, cf_specificity ~ 0."""
    img, m, rng = _setup(3)
    res = CF.verify_image(img, m, background_shortcut_model(), cfg=CF.CFConfig(seed=0), rng=np.random.default_rng(0))
    f = res['features']
    assert f['bg_sens'] > f['lesion_sens']
    assert f['lesion_bg_contrast'] < 0
    assert abs(f['cf_specificity']) < 0.35


def test_empty_mask_is_reported_not_faked():
    img, _, _ = _setup(4)
    res = CF.verify_image(img, np.zeros((256, 256), np.uint8), lesion_model())
    assert res['status'] == 'empty_mask'
    assert all(np.isnan(v) for v in res['features'].values())          # không bịa số


def test_no_ground_truth_in_inference_api():
    """Suy luận không được có đường vào cho GT mask."""
    sig = inspect.signature(CF.verify_image)
    assert all('gt' not in p.lower() and 'ground' not in p.lower() for p in sig.parameters)
    src = inspect.getsource(CF)
    for forbidden in ('gt_mask', 'ground_truth'):
        assert forbidden not in src.replace('Không dùng ground-truth', '').lower().replace('ground-truth', '')


def test_ops_preserve_documented_properties():
    img, m, rng = _setup(5)
    reg = m > 0
    # permute: giữ nguyên histogram vùng, đổi cấu trúc; ngoài vùng không đổi
    p = CF.apply_op(img, reg, 'permute', np.random.default_rng(0))
    assert np.array_equal(np.sort(img[reg].ravel()), np.sort(p[reg].ravel()))
    assert np.array_equal(img[~reg], p[~reg]) and not np.array_equal(img[reg], p[reg])
    # blur: giảm độ lệch chuẩn cục bộ
    b = CF.apply_op(img, reg, 'blur', np.random.default_rng(0))
    assert b[reg].std() < img[reg].std()
    assert np.array_equal(img[~reg], b[~reg])
    # replace: giá trị mới lấy từ vùng tham chiếu
    ref = np.zeros_like(reg); ref[:30, :30] = True
    r = CF.apply_op(img, reg, 'replace', np.random.default_rng(0), ref=ref)
    refvals = {tuple(v) for v in img[ref].reshape(-1, 3)}
    assert all(tuple(v) in refvals for v in r[reg].reshape(-1, 3)[:200])


def test_determinism_with_seed():
    img, m, _ = _setup(6)
    a = CF.verify_image(img, m, lesion_model(), rng=np.random.default_rng(11))['features']
    b = CF.verify_image(img, m, lesion_model(), rng=np.random.default_rng(11))['features']
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)


def test_qa_records_expose_ood_artifacts():
    img, m, _ = _setup(7)
    res = CF.verify_image(img, m, lesion_model(), return_records=True)
    rec = res['records']
    assert {'target' if r['group'] == 'region' else r['group'] for r in rec} >= {'target', 'matched', 'random'}
    assert all('mean_abs_change' in r and 'seam_ratio' in r for r in rec)


def test_evidence_intervention_uses_model_evidence_not_gt():
    img, m, _ = _setup(8)
    ev = np.zeros((32, 32), np.float32); ev[14:19, 14:19] = 1; ev /= ev.sum()
    fn = lesion_model()
    res = CF.verify_image(img, m, lambda x: dict(fn(x), evidence=None), evidence=ev)
    assert np.isfinite(res['features']['ev_faithfulness'])
    assert res['features']['ev_drop_top'] != 0


def test_shortcut_probe_detects_pad_reliance():
    rng = np.random.default_rng(0)
    img, _ = make_us_image(rng, pad=24, annotation=True)
    valid = PP.valid_content_mask(img)
    assert valid.sum() < valid.size

    def pad_model(x):                                  # chỉ nhìn dải pad
        return dict(logits=np.array([[0.0, 0.05 * float(im[:, :24].mean())] for im in x]))
    res = CF.shortcut_probe(img, valid, pad_model)
    assert res['pad_sens'] > 0.5
