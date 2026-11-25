import numpy as np
import pytest

from src import config, perturbation as PT
from src.security import UserFacingError


def _synthetic_regions(h=256, w=256):
    mask = np.zeros((h, w), np.uint8)
    mask[100:140, 100:150] = 1
    post = np.zeros((h, w), np.float32)
    post[140:200, 95:155] = 0.9
    valid = np.ones((h, w), bool)
    valid[:, :20] = False
    valid[:, -20:] = False
    return mask, post, valid


def _img(h=256, w=256, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_regions_are_disjoint_and_valid_only():
    mask, post, valid = _synthetic_regions()
    regions, skipped = PT.build_regions(mask, post, valid)
    assert set(regions) == {'lesion', 'ring', 'posterior', 'background'}
    keys = list(regions)
    for i in range(len(keys)):
        assert not (regions[keys[i]] & ~valid).any()
        for j in range(i + 1, len(keys)):
            assert not (regions[keys[i]] & regions[keys[j]]).any(), (keys[i], keys[j])


def test_perturbation_equal_area():
    mask, post, valid = _synthetic_regions()
    regions, _ = PT.build_regions(mask, post, valid)
    plan = PT.plan_pixels(regions, 0.5, seed=1, equal_area=True)
    sizes = {k: v.size for k, v in plan.items()}
    assert len(set(sizes.values())) == 1
    assert sizes['lesion'] == int(np.floor(0.5 * min(int(v.sum()) for v in regions.values())))
    for k, idx in plan.items():
        assert regions[k].ravel()[idx].all()                    # pixel chọn nằm trong đúng vùng
        assert len(np.unique(idx)) == idx.size


def test_non_equal_mode_differs():
    mask, post, valid = _synthetic_regions()
    regions, _ = PT.build_regions(mask, post, valid)
    plan = PT.plan_pixels(regions, 0.5, seed=1, equal_area=False)
    assert len({v.size for v in plan.values()}) > 1


def test_perturbation_reproducible():
    mask, post, valid = _synthetic_regions()
    regions, _ = PT.build_regions(mask, post, valid)
    img = _img()
    for kind in PT.KINDS:
        p1, p2 = PT.plan_pixels(regions, 0.3, 7), PT.plan_pixels(regions, 0.3, 7)
        for k in p1:
            assert np.array_equal(p1[k], p2[k])
            a, b = PT.apply_perturbation(img, p1[k], kind, 7), PT.apply_perturbation(img, p2[k], kind, 7)
            assert np.array_equal(a, b)
    p3 = PT.plan_pixels(regions, 0.3, 8)
    assert not np.array_equal(PT.plan_pixels(regions, 0.3, 7)['lesion'], p3['lesion'])


def test_shuffle_preserves_histogram_and_only_touches_selected_pixels():
    mask, post, valid = _synthetic_regions()
    regions, _ = PT.build_regions(mask, post, valid)
    img = _img()
    idx = PT.plan_pixels(regions, 0.5, 3)['ring']
    out = PT.apply_perturbation(img, idx, 'shuffle', 3)
    flat_a, flat_b = img.reshape(-1, 3), out.reshape(-1, 3)
    untouched = np.ones(len(flat_a), bool)
    untouched[idx] = False
    assert np.array_equal(flat_a[untouched], flat_b[untouched])
    assert np.array_equal(np.sort(flat_a[idx], axis=0), np.sort(flat_b[idx], axis=0))


def test_empty_or_tiny_mask_and_single_region_fail_gracefully():
    _, _, valid = _synthetic_regions()
    with pytest.raises(UserFacingError):
        PT.build_regions(np.zeros((256, 256), np.uint8), None, valid)
    with pytest.raises(UserFacingError):
        PT.plan_pixels({'lesion': np.ones((10, 10), bool)}, 0.5, 0)
    with pytest.raises(UserFacingError):
        PT.plan_pixels({'a': np.ones((10, 10), bool), 'b': np.ones((10, 10), bool)}, 0.0, 0)


def test_run_lab_with_real_model_equal_area_and_reproducible(rt, synth_result):
    mask, post, valid = _synthetic_regions()
    regions, _ = PT.build_regions(mask, post, valid)
    o1 = PT.run_lab(rt, synth_result, regions, 'shuffle', 0.2, 5, True)
    o2 = PT.run_lab(rt, synth_result, regions, 'shuffle', 0.2, 5, True)
    assert len({o.n_perturbed for o in o1}) == 1
    for a, b in zip(o1, o2):
        assert a.region == b.region and a.pert_logit_malignant == b.pert_logit_malignant and np.array_equal(a.image, b.image)
        assert a.delta_logit_malignant == pytest.approx(a.pert_logit_malignant - a.orig_logit_malignant)
        assert a.orig_p_malignant_cal == synth_result.p_malignant
        assert not np.array_equal(a.image, synth_result.image)
