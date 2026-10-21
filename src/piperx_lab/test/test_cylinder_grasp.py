import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_lab.cylinder_grasp import grasp_candidates, support_patch


def test_tip_to_tcp_mapping_and_orthogonal_grasp_frames():
    tip = np.eye(4)
    tip[2, 3] = 0.1425
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('z', 0.3).as_matrix()
    tcp[:3, 3] = [0.01, 0.0, 0.08]
    base = {'axis_reference_xyz_m': [0.4, -0.1, 0.1],
            'axis_direction': [0.1, 0.2, np.sqrt(0.95)], 'diameter_m': 0.043}
    candidates = grasp_candidates(base, tip, tcp)
    assert len(candidates) == 16
    for candidate in candidates:
        goal = np.array(candidate['T_base_tcp'])
        recovered_tip = goal @ np.linalg.inv(tcp) @ tip
        np.testing.assert_allclose(recovered_tip, candidate['T_base_tip'], atol=1e-10)
        rotation = recovered_tip[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        assert np.linalg.det(rotation) > 0.99999
        assert abs(np.dot(rotation[:, 1], base['axis_direction'])) < 1e-10
        assert candidate['opening_m'] == pytest.approx(0.063)
    with pytest.raises(ValueError, match='opening'):
        grasp_candidates({**base, 'diameter_m': 0.095}, tip, tcp)


def test_local_support_plane_with_noise_and_outliers():
    rng = np.random.default_rng(42)
    xy = rng.uniform([-0.15, -0.15], [0.15, 0.15], (600, 2)) + [0.4, -0.1]
    plane = np.column_stack((xy, 0.1 + rng.normal(0, 0.001, 600)))
    noise = rng.uniform([0.2, -0.3, -0.05], [0.6, 0.1, 0.13], (100, 3))
    patch = support_patch(np.vstack((plane, noise)), [0.4, -0.1, 0.15])
    assert abs(patch['surface_point_m'][2] - 0.1) < 0.002
    assert patch['normal'][2] > 0.99
    assert patch['inliers'] > 550


def test_measured_full_cylinder_uses_support_not_red_band_center():
    from piperx_lab.cylinder_grasp import full_cylinder_geometry
    base = {'axis_reference_xyz_m': [.4, -.1, .025], 'axis_direction': [0., 0., -1.],
            'diameter_m': .044, 'observed_red_span_m': .04,
            'support_patch': {'surface_point_m': [.4, -.1, 0.], 'normal': [0., 0., 1.]}}
    spec = {'diameter_m': .046, 'height_m': .08,
            'upright_on_observed_support_confirmed': True,
            'white_ends_within_measured_diameter_confirmed': True}
    full = full_cylinder_geometry(base, spec)
    np.testing.assert_allclose(full['center_xyz_m'], [.4, -.1, .04])
    assert not full['pose_independently_validated']
    with pytest.raises(ValueError, match='conflicts'):
        full_cylinder_geometry({**base, 'axis_reference_xyz_m': [.4, -.1, .09]}, spec)
    with pytest.raises(ValueError, match='confirmed'):
        full_cylinder_geometry(base, {**spec, 'upright_on_observed_support_confirmed': False})
