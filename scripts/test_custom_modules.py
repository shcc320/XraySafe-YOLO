from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xraysafe_yolo.modules import CGOA, LSAFF


def main():
    torch.manual_seed(0)
    p3 = torch.randn(2, 64, 80, 80)
    p4 = torch.randn(2, 128, 40, 40)
    p5 = torch.randn(2, 256, 20, 20)

    cgoa3 = CGOA(64, 64)
    cgoa4 = CGOA(128, 128)
    p3e = cgoa3(p3)
    p4e = cgoa4(p4)
    assert p3e.shape == p3.shape, (p3e.shape, p3.shape)
    assert p4e.shape == p4.shape, (p4e.shape, p4.shape)

    out3 = LSAFF([64, 128, 256], 64, target_index=0)([p3e, p4e, p5])
    out4 = LSAFF([64, 128, 256], 128, target_index=1)([p3e, p4e, p5])
    out5 = LSAFF([64, 128, 256], 256, target_index=2)([p3e, p4e, p5])
    out3_res = LSAFF([64, 128, 256], 64, target_index=0, residual=True)([p3e, p4e, p5])
    out4_spatial = LSAFF(
        [64, 128, 256],
        128,
        target_index=1,
        residual=True,
        spatial_fusion=True,
    )([p3e, p4e, p5])
    assert out3.shape == p3.shape, (out3.shape, p3.shape)
    assert out4.shape == p4.shape, (out4.shape, p4.shape)
    assert out5.shape == p5.shape, (out5.shape, p5.shape)
    assert out3_res.shape == p3.shape, (out3_res.shape, p3.shape)
    assert out4_spatial.shape == p4.shape, (out4_spatial.shape, p4.shape)
    print("CGOA/LSAFF synthetic shape test passed.")


if __name__ == "__main__":
    main()
