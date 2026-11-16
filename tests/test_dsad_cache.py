import numpy as np
from PIL import Image

from davinci.data import dsad_cache


def test_cache_roundtrip_is_lossless(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (64, 80, 3), np.uint8)).save(tmp_path / "im.png")
    for name, s in (("a.png", 0), ("b.png", 10)):
        Image.fromarray(((rng.random((64, 80)) > 0.5) * 255).astype(np.uint8)).save(tmp_path / name)
    record = {"frame_md5": "abc", "image": "im.png",
              "masks": {"abdominal_wall": "a.png", "vesicular_glands": "b.png"}}
    image, masks = dsad_cache.load_resized(tmp_path, record, (32, 40))
    out = tmp_path / "cache"
    out.mkdir()
    dsad_cache.write_entry(out, record, image, masks)
    c_image, c_masks = dsad_cache.read_entry(out, record)
    assert np.array_equal(image, c_image)
    assert c_masks.keys() == masks.keys()
    assert all(np.array_equal(masks[s], c_masks[s]) for s in masks)


def test_dice_restricted_to_mask_channels_ignores_empty_targets():
    import torch
    from davinci.models.seg import masked_bce_dice
    logits = torch.full((1, 2, 8, 8), 3.0)        # confident positive everywhere
    target = torch.zeros(1, 2, 8, 8)
    target[0, 0] = 1.0                            # channel 0: mask, all foreground
    weight = torch.ones(1, 2, 8, 8)               # channel 1: absent, all negative
    full = masked_bce_dice(logits, target, weight)
    only_mask = masked_bce_dice(logits, target, weight, dice_channels=torch.tensor([[True, False]]))
    assert only_mask < full                       # the empty-target Dice term is gone
