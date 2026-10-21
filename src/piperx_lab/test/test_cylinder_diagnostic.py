import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).resolve().parents[3] / 'scripts/diagnose_cylinder_geometry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_packed_sdk_cloud_restores_only_valid_pixels():
    msg = SimpleNamespace(is_bigendian=False, fields=[SimpleNamespace(name=a, offset=i*4) for i,a in enumerate(['x','y','z'])],
                          point_step=12, row_step=24, height=1, width=2,
                          data=np.array([[.1,.2,.5],[.3,.4,.6]],dtype='<f4').tobytes())
    depth = np.array([[0, 100], [100, 0]], dtype=np.uint16)
    cloud = module.organized_sdk_cloud(msg, depth)
    np.testing.assert_allclose(cloud[0,1], [.1,.2,.5])
    np.testing.assert_allclose(cloud[1,0], [.3,.4,.6])
    np.testing.assert_allclose(cloud[0,0], [0,0,0])
    with pytest.raises(ValueError, match='counts differ'):
        module.organized_sdk_cloud(msg, np.ones((2,2),dtype=np.uint16))
