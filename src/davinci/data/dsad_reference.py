"""Published DSAD results used as reproduction references (mean image IoU).

Kolbinger et al., Int J Surg 2023;109(10):2962-2974, doi:10.1097/JS9.0000000000000595.
Test: main-text Tables 1-2. Validation: Supplementary Tables 1b/1d and 2b/2d
(Supplemental Digital Content 1, http://links.lww.com/JS9/A792).
"Specialist" = one model per structure; "combined" = shared encoder with one
decoder per structure.
"""
from __future__ import annotations

from davinci.data.dsad import STRUCTURES


def _table(*values: float) -> dict[str, float]:
    assert len(values) == len(STRUCTURES)
    return dict(zip(STRUCTURES, values))


KOLBINGER_IOU = {
    ("deeplabv3", "specialist", "val"): _table(.85, .74, .49, .53, .66, .58, .81, .63, .51, .34, .32),
    ("deeplabv3", "specialist", "test"): _table(.83, .69, .41, .44, .71, .28, .80, .69, .60, .36, .28),
    ("deeplabv3", "combined", "val"): _table(.78, .62, .40, .48, .49, .49, .74, .57, .37, .22, .32),
    ("deeplabv3", "combined", "test"): _table(.77, .63, .40, .35, .57, .23, .72, .69, .53, .29, .30),
    ("segformer", "specialist", "val"): _table(.85, .70, .54, .56, .76, .64, .83, .73, .64, .35, .37),
    ("segformer", "specialist", "test"): _table(.85, .66, .46, .52, .75, .37, .83, .78, .66, .46, .31),
    ("segformer", "combined", "val"): _table(.69, .57, .32, .43, .43, .47, .67, .55, .40, .24, .38),
    ("segformer", "combined", "test"): _table(.66, .52, .28, .33, .53, .26, .67, .59, .55, .27, .26),
}
