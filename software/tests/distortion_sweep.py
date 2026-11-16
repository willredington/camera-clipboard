"""Not a test: how much does uncorrected barrel distortion cost? Run: python tests/distortion_sweep.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synth import render_scene  # noqa: E402
from test_rectify import corner_error_mm, disc_error_mm  # noqa: E402
from clipcam.rectify import Rectifier  # noqa: E402

print("k1 (normalised coords)  edge shrink at r=0.9  rms_mm  page-corner err  disc err (mm)")
for k1 in (0.0, -0.01, -0.02, -0.05, -0.1):
    img, t = render_scene(seed=1, pitch=3, yaw=-2, k1=k1)
    r = Rectifier(t.geom, 200).process(img)
    print(f"{k1:>8}  {abs(k1) * 0.81 * 100:10.1f}%  {r.rms_mm:7.3f}  {corner_error_mm(r, t):8.2f}  {disc_error_mm(r, t):8.2f}")
