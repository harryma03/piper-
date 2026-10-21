import numpy as np
import pytest
from types import SimpleNamespace

from piperx_lab.cylinder_geometry import fit_cylinder
from piperx_lab.rgbd_geometry import aligned_frames_valid


def test_partial_noisy_cylinder_at_unknown_orientation():
    rng = np.random.default_rng(19)
    axis = np.array([0.3, -0.4, np.sqrt(0.75)])
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t, angle = np.meshgrid(np.linspace(-0.035, 0.035, 20), np.linspace(-1.1, 1.1, 30))
    points = ([0.08, 0.03, 0.5] + t.reshape(-1, 1) * axis +
              0.025 * (np.cos(angle).reshape(-1, 1) * u +
                       np.sin(angle).reshape(-1, 1) * v))
    points += rng.normal(0, 0.0003, points.shape)
    result = fit_cylinder(points)
    assert abs(result['diameter_m'] - 0.05) < 0.002
    assert abs(np.dot(result['axis_direction'], axis)) > 0.99
    assert result['full_height_m'] is None
    assert result['object_center_xyz_m'] is None


def test_planar_red_patch_is_not_a_measured_cylinder():
    x, y = np.meshgrid(np.linspace(-0.02, 0.02, 20), np.linspace(-0.03, 0.03, 20))
    plane = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))
    with pytest.raises(ValueError, match='curvature'):
        fit_cylinder(plane)


def test_vidu_aligned_alias_is_accepted_but_raw_color_is_not():
    def msg(frame):
        return SimpleNamespace(header=SimpleNamespace(frame_id=frame))
    bundle = {'depth': msg('camera_depth_frame'), 'info': msg('camera_depth_frame'),
              'color': msg('camera_aligned_color_frame')}
    assert aligned_frames_valid(bundle)
    bundle['color'] = msg('camera_color_frame')
    assert not aligned_frames_valid(bundle)
