#!/usr/bin/env python3
"""Assemble the clipboard, export Blender-ready STLs + JSON, and render a quick matplotlib preview PNG."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import make_clipboard as M
import assembly as A

out = sys.argv[1] if len(sys.argv) > 1 else 'out'
parts = M.build(out)
items, cable = A.export(parts, out)
BW, BL = M.BOARD_W, M.BOARD_L

def draw(ax, elev, azim):
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
    for it in sorted(items, key=lambda i: i['mesh'].centroid[2]):
        m = it['mesh']
        shade = 0.45 + 0.55 * np.clip(m.face_normals @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(it['color']))
        fc = np.clip(base[None, :] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=fc, edgecolors='none'))
    ax.plot(cable[:, 0], cable[:, 1], cable[:, 2], color='#ff2020', lw=2.4, zorder=10)
    ax.set_xlim(-60, BW + 20); ax.set_ylim(-20, BL + 20); ax.set_zlim(0, BL)
    ax.set_box_aspect((BW + 80, BL + 40, BL))
    ax.view_init(elev, azim); ax.set_axis_off()

fig = plt.figure(figsize=(14, 7), dpi=110)
ax1 = fig.add_subplot(1, 2, 1, projection='3d'); draw(ax1, 22, -35)
ax2 = fig.add_subplot(1, 2, 2, projection='3d'); draw(ax2, 89, -90)
ax1.set_title('Assembled (red = 5 V power path)'); ax2.set_title('Top-down (camera view)')
fig.tight_layout()
fig.savefig(os.path.join(out, 'assembly_preview.png'))
print('exported', len(items), 'items to', os.path.join(out, 'assembly'))
