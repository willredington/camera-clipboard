#!/usr/bin/env python3
"""
Builds the assembled clipboard in the GLOBAL frame (mm, board top surface = z 0) as a list of items,
and exports them for Blender:  <out>/assembly/<name>.stl  +  <out>/assembly/assembly.json
"""
import os, json
import numpy as np
import trimesh
import make_clipboard as M

COLORS = dict(printed='#d4572a', printed_blue='#2f6fbf', tube='#2b2b2b', board='#c9b48a', paper='#f2f2ee',
              pcb='#1a7f3c', pi='#2e9e57', usbc='#b22222', spring='#9aa0a6', rod='#c0c4c8')

def xf(m, T):
    m = m.copy(); m.apply_transform(T); return m

def build_items(parts):
    BW, BL = M.BOARD_W, M.BOARD_L
    items = []
    def add(name, mesh, color, kind='part', rough=0.55, metal=0.0):
        items.append(dict(name=name, mesh=mesh, color=COLORS.get(color, color), kind=kind, rough=rough, metal=metal))

    add('board', M.box(0, BW, 0, BL, -M.BOARD_T, 0), 'board', 'board', 0.8)
    add('paper', M.box((BW - M.PAPER_W) / 2, (BW + M.PAPER_W) / 2, (BL - M.PAPER_L) / 2, (BL + M.PAPER_L) / 2, 0, M.PAPER_T),
        'paper', 'paper', 0.9)

    # corner holders (board corner sits at mesh (WALL, WALL); slot centre z=5 -> board centre z=-1.5)
    c0 = parts['corner_holder_x4'].copy(); c0.apply_translation([-M.WALL, -M.WALL, -1.5 - 5.0])
    names = {(0, 0): 'corner_BL', (1, 0): 'corner_BR', (0, 1): 'corner_TL', (1, 1): 'corner_TR'}
    for (mx, my), n in names.items():
        c = c0.copy()
        if mx: c.apply_transform(np.diag([-1, 1, 1, 1]).astype(float)); c.apply_translation([BW, 0, 0])
        if my: c.apply_transform(np.diag([1, -1, 1, 1]).astype(float)); c.apply_translation([0, BL, 0])
        add(n, c, 'printed_blue')

    # spring clip
    T = M.clip_xf()
    add('clip_base', xf(parts['clip_base_x1'], T), 'printed_blue')
    add('clip_lever', xf(parts['clip_lever_x1'], T), 'printed_blue')
    py, pz = M.CLIP_PIV
    add('clip_pivot_rod', xf(M.cyl('x', [M.CLIP_L / 2, py, pz], 1.5, M.CLIP_L + 4), T), 'rod', 'metal', 0.3, 1.0)
    for i, sp in enumerate(M.clip_spring_meshes(0.0)):      # seat floor -> dimple ceiling, bent to meet both end faces
        add(f'clip_spring_{i}', xf(sp, T), 'spring', 'metal', 0.35, 1.0)

    # boom
    v_len, h_len, (mx_, my_), (cx, cy), rod_axis, rod_top, floor = M.rod_lengths()
    add('mast_base', xf(parts['mast_base_x1'], M.mast_base_xf()), 'printed')
    add('tube_vertical', M.cylz(mx_, my_, floor, rod_top, 5.0), 'tube', 'tube', 0.4, 0.4)
    add('elbow', xf(parts['elbow_cross_clamp_x1'], M.elbow_xf()), 'printed')
    s = 1.0 if M.MAST_SIDE == 'left' else -1.0
    x_a = mx_ + s * (-M.ELBOW['hend']); x_b = cx - s * M.HEAD_SOCKET_FLOOR_Y
    add('tube_horizontal', M.cyl('x', [(x_a + x_b) / 2, cy, rod_axis], 5.0, abs(x_b - x_a)), 'tube', 'tube', 0.4, 0.4)
    HT = M.head_xf()
    add('camera_head', xf(parts['camera_head_x1'], HT), 'printed')

    # electronics stand-ins (local to the head)
    pt = M.HEAD_PT
    # Camera Module 3 Wide stand-in (dims from the official STEP): PCB on the plate, lens block + sensor flex + lens in the
    # window/slot, FPC connector on the back at the lens-end edge
    y0 = -M.CAM_LENS_FROM_EDGE; y1 = y0 + M.CAM_PCB[1]
    cam = M.U(*[
        M.box(-12.5, 12.5, y0, y1, pt, pt + 0.7),
        M.box(-5.4, 5.4, -5.4, 5.4, pt - 2.35, pt),
        M.box(-5.4, 5.75, -12.7, 5.4, pt - 2.1, pt),
        M.cylz(0, 0, pt - M.CAM_LENS_TIP, pt - 2.35, 3.5),
        M.box(-11.5, 11.5, 2.55, 9.1, pt + 0.7, pt + 3.3)])
    add('camera_pcb', xf(cam, HT), 'pcb', 'pcb', 0.5)
    add('pi_zero_2w', xf(M.box(-32.5, 32.5, M.PI_CY - 15, M.PI_CY + 15, pt + 3.0, pt + 4.4), HT), 'pi', 'pcb', 0.5)
    cx_m = M.MAST_W / 2
    zt1m = M._mast_z()[0]
    ub = M.U(*[M.box(cx_m - M.USBC_PCB[0] / 2, cx_m + M.USBC_PCB[0] / 2, 54 - M.USBC_PCB[1], 54, zt1m + 0.5, zt1m + 2.1),
                                   M.box(cx_m - 4.47, cx_m + 4.47, 54 - 7.4, 54, zt1m + 2.1, zt1m + 5.3)])   # board + receptacle
    add('usbc_breakout', xf(ub,
                           M.mast_base_xf()), 'usbc', 'pcb', 0.5)

    # power cable polyline (global)
    local = M.wire_paths()
    def g(pts, T):
        P = np.c_[np.array(pts, float), np.ones(len(pts))]
        return (P @ T.T)[:, :3]
    cable = np.vstack([
        g(local['mast_base_x1'], M.mast_base_xf()),
        g([(0, 0, 4.5), (0, 0, 11.0), (0, -1.0, 14.0), (0, -5.5, 14.0)], M.elbow_xf()),
        g([(0, 26, M.HEAD_ROD_Z), (0, 22.5, M.HEAD_ROD_Z), (0, 22.5, pt + 2), (0, 19.0, pt + 2),
           (0, 15.0, pt - 0.6), (-39.0, 15.0, pt - 0.6), (-39.0, -12.0, pt - 0.6)], HT),
    ])
    return items, cable

def export(parts, outdir):
    items, cable = build_items(parts)
    d = os.path.join(outdir, 'assembly'); os.makedirs(d, exist_ok=True)
    meta = []
    for it in items:
        fn = it['name'] + '.stl'
        it['mesh'].export(os.path.join(d, fn))
        meta.append(dict(name=it['name'], file=fn, color=it['color'], kind=it['kind'], rough=it['rough'], metal=it['metal']))
    with open(os.path.join(d, 'assembly.json'), 'w') as f:
        json.dump(dict(units='mm', board=[M.BOARD_W, M.BOARD_L, M.BOARD_T], ground_z=-6.5,
                       cable=cable.round(3).tolist(), items=meta), f, indent=1)
    return items, cable
