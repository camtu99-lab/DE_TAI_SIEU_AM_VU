import numpy as np
import pytest
from src import robustness as RB, segmentation as SG
from tests.synth import make_us_image, lesion_model, center_mask


def _img(seed=0):
    return make_us_image(np.random.default_rng(seed))[0]


def test_taxonomy_separates_label_preserving_from_counterfactual():
    lp = {k for k, v in RB.CORRUPTIONS.items() if v['label_preserving']}
    cf = {k for k, v in RB.CORRUPTIONS.items() if not v['label_preserving']}
    assert cf == {'occlusion_lesion'} and {'speckle', 'gaussian', 'brightness', 'contrast', 'blur', 'geometric', 'occlusion_background'} <= lp


@pytest.mark.parametrize('name', [k for k in RB.CORRUPTIONS if k != 'occlusion_lesion'])
def test_corruptions_run_all_levels_and_keep_dtype(name):
    img = _img(); m = center_mask()
    for lv in range(1, len(RB.CORRUPTIONS[name]['levels']) + 1):
        if name == 'occlusion_background' and lv == 5:
            continue                                                       # mức 5 không đặt được quanh tổn thương giữa ảnh: kiểm ở test riêng
        out, M = RB.apply_corruption(img, name, lv, np.random.default_rng(0), pred_mask=m, valid=np.ones(m.shape, bool))
        assert out.shape == img.shape and out.dtype == np.uint8


def test_background_occlusion_avoids_lesion_and_lesion_occlusion_hits_it():
    img = _img(); m = center_mask(); v = np.ones(m.shape, bool)
    out, _ = RB.apply_corruption(img, 'occlusion_background', 3, np.random.default_rng(1), pred_mask=m, valid=v)
    assert (out[m > 0] == img[m > 0]).all() and (out != img).any()
    with pytest.raises(ValueError):                                      # không đủ chỗ: báo lỗi, không bỏ qua im lặng
        RB.apply_corruption(img, 'occlusion_background', 5, np.random.default_rng(1), pred_mask=m, valid=v)
    out2, _ = RB.apply_corruption(img, 'occlusion_lesion', 3, np.random.default_rng(1), pred_mask=m, valid=v)
    assert (out2[m > 0] != img[m > 0]).any() and (out2[m == 0] == img[m == 0]).all()
    with pytest.raises(ValueError):                                      # mask rỗng: báo lỗi, không im lặng
        RB.apply_corruption(img, 'occlusion_lesion', 1, np.random.default_rng(1), pred_mask=np.zeros_like(m), valid=v)


def test_geometric_mask_alignment_roundtrip():
    img = _img(); m = center_mask().astype(np.float32)
    out, M = RB.apply_corruption(img, 'geometric', 3, np.random.default_rng(0))
    warped = RB.cv2.warpAffine(m, M, (256, 256))
    back = RB.inverse_warp_map(warped, M, (256, 256))
    assert SG.dice(back > 0.5, m > 0.5) > 0.95


def test_hflip_inverse_map_exact():
    a = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    flipped, M = RB.make_view(np.stack([a * 255] * 3, -1).astype(np.uint8), RB.STABILITY_VIEWS[0])
    assert np.allclose(RB.inverse_warp_map(flipped[..., 0].astype(np.float32), M, (64, 64)), (a * 255).astype(np.uint8), atol=1)


def test_stability_features_bounded_and_invariant_model():
    img = _img(); valid = np.ones((256, 256), bool)
    const = lambda ims: dict(logits=np.tile([[0.0, 0.3]], (len(ims), 1)), seg_prob=None, evidence=None)
    f = RB.stability_features(img, const, T=1.0, thr=0.5)['features']
    assert f['view_js'] == pytest.approx(0, abs=1e-12) and f['prediction_stability'] == 1.0 and np.isnan(f['mask_consistency'])
    g = RB.stability_features(img, lesion_model(), T=1.0, thr=0.5, valid=valid)['features']
    assert 0 <= g['mask_consistency'] <= 1 and 0 <= g['prediction_stability'] <= 1 and g['view_js'] >= 0


def test_unstable_model_gets_lower_stability():
    img = _img()
    rng = np.random.default_rng(0)
    noisy = lambda ims: dict(logits=np.stack([np.zeros(len(ims)), rng.standard_normal(len(ims)) * 3], 1), seg_prob=None, evidence=None)
    f = RB.stability_features(img, noisy, T=1.0, thr=0.5)['features']
    assert f['view_js'] > 0.01 and f['view_dz_sd'] > 0.5


def test_corruption_changes_image_but_not_lesion_label_semantics():
    img = _img()
    for n in ('speckle', 'gaussian', 'blur', 'contrast'):
        out, _ = RB.apply_corruption(img, n, 5, np.random.default_rng(0))
        assert (out.astype(int) != img.astype(int)).any()


def test_run_robustness_lab_rows_and_skips_are_recorded():
    import numpy as np
    rng = np.random.default_rng(0)
    imgs = [make_us_image(rng, pad=4)[0] for _ in range(3)]
    risk = lambda im: float(im.mean() / 255)
    df = RB.run_robustness_lab(imgs, lesion_model(), levels=(1, 5), risk_fn=risk)
    assert {'abs_dp', 'flip', 'seg_dice', 'abs_drisk', 'label_preserving', 'status'} <= set(df.columns)
    assert df[df.corruption == 'occlusion_lesion'].label_preserving.eq(False).all()
    s = RB.summarize_robustness(df)
    assert (s.n > 0).all()
    sk = df[df.status != 'ok']                                           # ca không thực hiện được phải được ghi, không biến mất
    assert (sk.status.str.startswith('skipped')).all()
    strong = df[(df.corruption == 'gaussian') & (df.level == 5)].mean_abs_pixel_change.mean()
    weak = df[(df.corruption == 'gaussian') & (df.level == 1)].mean_abs_pixel_change.mean()
    assert strong > weak
