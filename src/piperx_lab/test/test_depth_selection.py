import numpy as np
import pytest

from piperx_lab.depth_selection import select_depth_component


def scene():
    y, x = np.mgrid[:100, :120]
    xyz = np.stack((x * .002, y * .002, np.full_like(x, .8, dtype=float)), axis=-1)
    xyz[25:65, 30:70, 2] = .5
    return xyz


def test_selects_depth_connected_surface_without_colour():
    xyz = scene()
    mask = select_depth_component(xyz, (45, 40))
    assert mask.sum() == 40 * 40
    assert mask[40, 45]
    assert not mask[40, 80]


def test_rejects_invalid_pixel_and_excessively_large_selection():
    xyz = scene()
    xyz[40, 45] = 0
    with pytest.raises(ValueError, match='没有有效深度'):
        select_depth_component(xyz, (45, 40))
    with pytest.raises(ValueError, match='outside'):
        select_depth_component(xyz, (-1, 40))
    with pytest.raises(ValueError, match='选区过大'):
        select_depth_component(xyz, (0, 0), radius_m=.2, max_fraction=.1)


def test_touching_image_regions_with_depth_edge_stay_separate():
    xyz = scene()
    xyz[25:65, 70:90, 2] = .52  # within the seed depth window, but a 20 mm edge
    mask = select_depth_component(xyz, (60, 40), radius_m=.12)
    assert mask[40, 60]
    assert not mask[40, 75]
