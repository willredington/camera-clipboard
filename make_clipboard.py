#!/usr/bin/env python3
"""
Parametric generator for the "camera clipboard" frame hardware, rev 3.

Architecture
  * Backing board (YOU cut it): BOARD_W x BOARD_L x BOARD_T, held by 4 printed corner holders.
  * A printed spring clip (lever + pivot rod + 2 compression springs) sits at the top edge, like a real clipboard.
  * Mast base clamps the board's LEFT edge (MAST_SIDE='right' for left-handed writers). A USB-C power breakout drops into a pocket on its
    back; a 5 V pair runs up through the HOLLOW boom (10 mm OD x 8 mm ID tube).
  * Cross-clamp elbow joins the vertical and horizontal tubes; the wires pass through a cavity inside it.
  * Camera head carries a Raspberry Pi Zero 2 W and a Pi Camera Module 3, looking straight down.
    Only the 2-wire 5 V feed travels up the boom; the camera ribbon stays inside the head.

All dimensions mm. Board top surface (= paper plane) is z = 0.
Run:  python3 make_clipboard.py [outdir]      Needs: pip install trimesh manifold3d numpy
"""
import sys, os
import numpy as np
import trimesh
from trimesh.boolean import union, difference

# ----------------------------------------------------------------- parameters
PAPER_W, PAPER_L = 215.9, 279.4      # US Letter (use 210, 297 for A4)
BOARD_W, BOARD_L = 260.0, 330.0      # backing board you cut
BOARD_T = 3.0
PAPER_T = 0.2
FIT = 0.25                           # printer clearance per mating surface
ROD_D = 10.0                         # boom tube OD (use 10 x 8 hollow carbon/aluminium)
ROD_HOLE = ROD_D + 0.3
WALL = 3.0
SCREW_PILOT = 3.4                    # M4 self-tapping pilot (5.6 for heat-set insert)
WIRE_D = 6.0                         # internal wire channel diameter
CAM_LENS_H = 205.0                   # lens tip height above paper (Cam Module 3 WIDE ~205)
# Pi Camera Module 3 Wide, from RP-008155 drawing + official STEP. Long (25 mm) side = sensor long axis, along head x.
CAM_PCB = (25.0, 23.862)             # PCB (x, y)
CAM_HOLES = (21.0, 12.5)             # M2 hole spacing (dia 2.2 on the PCB); lower holes 2.0 from the bottom edge
CAM_LENS_FROM_EDGE = 14.4            # lens axis from the bottom edge: NOT centred between the holes (in line with the upper pair)
CAM_LENS_TIP = 8.1                   # lens tip below the PCB front face
CAM_WINDOW = 17.0                    # clears the 10.8 mm square lens block
CAM_FLEX = (12.5, 13.2)              # through-slot for the front-side sensor flex (2.1 mm proud): width, reach from lens axis
PI_SIZE = (65.0, 30.0)               # Pi Zero 2 W (RP-008358 drawing); holes 3.5 from each edge
PI_HOLES = (58.0, 23.0)              # M2.5 hole spacing
SEGS = 64
MAST_SIDE = 'left'                   # 'left' suits right-handed writers; 'right' mirrors it

# spring clip
CLIP_L = 100.0                       # clip base length (x)
CLIP_EDGE = 25.0                     # board edge at y=25; paper top edge at y=0
CLIP_EAR_T = 6.0
CLIP_PIV = (10.0, 12.75)             # pivot (y, z)
CLIP_TIP = (-6.0, 4.0)               # centre of lever front end (y, z)
CLIP_REAR = 22.0                     # lever length behind the pivot
CLIP_BAR_T = 7.0
CLIP_SPRING_Y = 19.0
CLIP_SPRING_D = 6.6                  # spring seat dia (6 mm OD spring)
CLIP_PIN_D = 3.3                     # M3 rod bore
CLIP_SEAT_DEPTH = 1.5                # spring seat recess in the base
CLIP_SEAT_BOSS = (10.0, 3.0)         # boss under each seat (dia, height): sets closed spring length (~11.4 mm -> preload)
CLIP_DIMPLE_DEPTH = 2.5              # lever dimple ceiling above the lever underside (at the spring axis)
CLIP_DIMPLE_TILT = 16.0              # deg; dimple pre-tilted by ~half the opening stroke so spring end misalignment is +/-16, not 0..32
CLIP_SPRING_FREE = 14.0              # spring free length the geometry is checked against
CLIP_LIFT = 8.0                      # jaw opening the checks require

# head geometry
HEAD_PT = 3.5                        # plate thickness
HEAD_ROD_Z = HEAD_PT + 12.0          # rod axis height above plate underside
HEAD_SOCKET_FLOOR_Y = 25.0           # rod bottoms out here (local y)
PI_CY = -29.0                        # Pi centre (local y), long axis along x
CAM_LENS_DROP = CAM_LENS_TIP - HEAD_PT   # lens tip below the plate underside (camera PCB sits on the plate)

# ----------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return m

def cyl(axis, c, r, h):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=SEGS)
    if axis == 'x':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 'y':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(c)
    return m

def cylz(x, y, z0, z1, r):           # vertical cylinder between two z values
    return cyl('z', [x, y, (z0 + z1) / 2], r, z1 - z0)

def cyly(x, z, y0, y1, r):           # cylinder along y between two y values
    return cyl('y', [x, (y0 + y1) / 2, z], r, y1 - y0)

def U(*ms):
    return union(list(ms), engine='manifold')

def D(a, *ms):
    return difference([a] + list(ms), engine='manifold')

# ----------------------------------------------------------------- frame parts
SLOT = BOARD_T + 2 * FIT

def corner_holder():
    S, H = 24.0, 10.0
    z0 = (H - SLOT) / 2
    body = box(0, S, 0, S, 0, H)
    slot = box(WALL, S + 1, WALL, S + 1, z0, z0 + SLOT)
    screw = cyl('z', [14, 14, H / 2], SCREW_PILOT / 2, H + 2)
    return D(body, slot, screw)

def channel(length, depth, gap, leg_t=3.0, wall_t=3.0):
    """C-channel along +x. Board edge butts the wall inner face at y=depth."""
    z_gap0 = -BOARD_T - FIT
    z_gap1 = z_gap0 + gap
    bot = box(0, length, 0, depth + wall_t, z_gap0 - leg_t, z_gap0)
    top = box(0, length, 0, depth + wall_t, z_gap1, z_gap1 + leg_t)
    wall = box(0, length, depth, depth + wall_t, z_gap0 - leg_t, z_gap1 + leg_t)
    return U(bot, top, wall), z_gap1, z_gap1 + leg_t

def _clip_geom():
    py, pz = CLIP_PIV
    ty, tz = CLIP_TIP
    d = np.array([py - ty, pz - tz]); front = np.linalg.norm(d); d = d / front
    theta = np.arctan2(d[1], d[0])
    rear = np.array([py, pz]) + CLIP_REAR * d
    zc = pz + (CLIP_SPRING_Y - py) * np.tan(theta)
    z_under = zc - (CLIP_BAR_T / 2) / np.cos(theta)
    return theta, front, rear, z_under

CLIP_SPRING_XS = (CLIP_L * 0.25, CLIP_L * 0.75)

def clip_spring_ends(open_deg=0.0):
    """Spring end centres (seat floor, dimple ceiling) in the clip_base frame with the lever rotated open_deg
    about the pivot (negative = jaw up). Returns [(bottom, top), ...] per spring."""
    py, pz = CLIP_PIV
    _, _, _, z_under = _clip_geom()
    zt1 = -BOARD_T - FIT + (BOARD_T + 2 * FIT) + 3.0
    floor = zt1 + CLIP_SEAT_BOSS[1] - CLIP_SEAT_DEPTH
    T = trimesh.transformations.rotation_matrix(np.radians(open_deg), [1, 0, 0], [0, py, pz])
    return [(np.array([x, CLIP_SPRING_Y, floor]), (T @ [x, CLIP_SPRING_Y, z_under + CLIP_DIMPLE_DEPTH, 1])[:3])
            for x in CLIP_SPRING_XS]

def clip_spring_meshes(open_deg=0.0, r=3.0, n=16):
    """Spring stand-ins (clip_base frame): tubes leaving the seat square to it and entering the dimple square to its
    ceiling, i.e. the bent shape a compression spring takes between the two non-parallel end faces."""
    py, pz = CLIP_PIV
    a = np.radians(open_deg + CLIP_DIMPLE_TILT)              # dimple axis angle about x in the base frame
    t0, t1 = np.array([0, 0, 1.0]), np.array([0, -np.sin(a), np.cos(a)])
    out = []
    for bot, top in clip_spring_ends(open_deg):
        L = np.linalg.norm(top - bot)
        P = [bot, bot + t0 * L / 3, top - t1 * L / 3, top]
        s = np.linspace(0, 1, n)[:, None]
        C = (1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3]
        tan = 3 * ((1 - s) ** 2 * (P[1] - P[0]) + 2 * (1 - s) * s * (P[2] - P[1]) + s ** 2 * (P[3] - P[2]))
        tan /= np.linalg.norm(tan, axis=1)[:, None]          # exact: end caps lie flat on the seat floor and dimple ceiling
        ex = np.array([1.0, 0, 0])                          # curve lies in a y-z plane, so x is a valid normal everywhere
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        rings = [c + r * (np.cos(ang)[:, None] * ex + np.sin(ang)[:, None] * np.cross(t, ex)) for c, t in zip(C, tan)]
        V = np.vstack(rings + [C[0], C[-1]]); m = len(ang); F = []
        for i in range(n - 1):
            for j in range(m):
                a0, a1, b0, b1 = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
                F += [[a0, b0, a1], [a1, b0, b1]]
        for j in range(m):
            F += [[n * m, j, (j + 1) % m], [n * m + 1, (n - 1) * m + (j + 1) % m, (n - 1) * m + j]]
        mesh = trimesh.Trimesh(V, F, process=True)
        if mesh.volume < 0: mesh.invert()
        out.append(mesh)
    return out

def clip_base():
    """C-channel on the top edge with two pivot ears and two spring seats."""
    gap = BOARD_T + 2 * FIT
    ch, zt0, zt1 = channel(CLIP_L, CLIP_EDGE, gap)
    py, pz = CLIP_PIV
    ears = [box(0, CLIP_EAR_T, py - 6, py + 6, zt1 - 0.5, pz + 5),
            box(CLIP_L - CLIP_EAR_T, CLIP_L, py - 6, py + 6, zt1 - 0.5, pz + 5)]
    bd, bh = CLIP_SEAT_BOSS
    bosses = [cylz(x, CLIP_SPRING_Y, zt1 - 0.5, zt1 + bh, bd / 2) for x in CLIP_SPRING_XS]
    body = U(ch, *ears, *bosses)
    pin = cyl('x', [CLIP_L / 2, py, pz], CLIP_PIN_D / 2, CLIP_L + 2)
    scr = [cyl('z', [x, 21.0, (zt0 + zt1) / 2], SCREW_PILOT / 2, 10) for x in (3.0, CLIP_L - 3.0)]
    seats = [cylz(x, CLIP_SPRING_Y, zt1 + bh - CLIP_SEAT_DEPTH, zt1 + bh + 0.1, CLIP_SPRING_D / 2) for x in CLIP_SPRING_XS]
    return D(body, pin, *scr, *seats)

def clip_lever():
    theta, front, rear, z_under = _clip_geom()
    py, pz = CLIP_PIV
    ty, tz = CLIP_TIP
    w = CLIP_L - 2 * CLIP_EAR_T - 0.8
    x0 = (CLIP_L - w) / 2
    total = front + CLIP_REAR
    bar = trimesh.creation.box(extents=[w, total, CLIP_BAR_T])
    bar.apply_transform(trimesh.transformations.rotation_matrix(theta, [1, 0, 0]))
    c = (np.array([ty, tz]) + rear) / 2
    bar.apply_translation([CLIP_L / 2, c[0], c[1]])
    boss = cyl('x', [CLIP_L / 2, py, pz], 5.0, w)
    foot = box(x0, x0 + w, ty - 4.0, ty + 4.0, 0.2, 5.5)
    body = U(bar, boss, foot)
    bore = cyl('x', [CLIP_L / 2, py, pz], CLIP_PIN_D / 2, w + 2)
    dimples = []
    for x in CLIP_SPRING_XS:                      # pre-rotated +TILT about the ceiling centre: square to the seat at lever angle -TILT
        c = [x, CLIP_SPRING_Y, z_under + CLIP_DIMPLE_DEPTH]
        dm = cylz(x, CLIP_SPRING_Y, z_under - 6.0, c[2], CLIP_SPRING_D / 2)
        dm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(CLIP_DIMPLE_TILT), [1, 0, 0], c))
        dimples.append(dm)
    return D(body, bore, *dimples)

# ----------------------------------------------------------------- mast base (USB-C power in)
MAST_COL_R = 10.5
MAST_COL_Y = 11.0
MAST_COL_H = 40.0
USBC_PCB = (20.4, 14.2)              # Adafruit 4090 USB-C breakout (port on the 20.4 edge), drop in from top
USBC_PORT_Z = 3.2                    # receptacle centre above the pocket floor (1.6 mm PCB + half a 3.2 mm top-mount shell)
MAST_W = 40.0
MAST_DEPTH = 20.0

def _mast_z():
    gap = BOARD_T + 2 * FIT
    z_gap0 = -BOARD_T - FIT
    zt1 = z_gap0 + gap + 3.0         # top of top leg
    wire_z = zt1 + 3.5               # wire channel centre height
    floor = zt1 + 8.0                # rod bottoms out here (wire bore continues up through it)
    top = zt1 + MAST_COL_H
    return zt1, wire_z, floor, top, z_gap0 - 3.0

def mast_socket_depth():
    _, _, floor, top, _ = _mast_z()
    return top - floor

def mast_base():
    w = MAST_W
    gap = BOARD_T + 2 * FIT
    ch, zt0, zt1_ = channel(w, MAST_DEPTH, gap)
    zt1, wire_z, floor, top, zbot = _mast_z()
    cx = w / 2
    col = cylz(cx, MAST_COL_Y, zt1, top, MAST_COL_R)
    hx0, hx1 = cx - 13, cx + 13
    hous_up = box(hx0, hx1, MAST_COL_Y, 56, zt1, zt1 + 14)
    hous_dn = box(hx0, hx1, MAST_DEPTH + 3.0, 56, zbot, zt1)
    body = U(ch, col, hous_up, hous_dn)
    socket = cylz(cx, MAST_COL_Y, floor, top + 1, ROD_HOLE / 2)
    wire_v = cylz(cx, MAST_COL_Y, wire_z - WIRE_D / 2, floor + 1, WIRE_D / 2)
    wire_h = cyly(cx, wire_z, MAST_COL_Y, 46, WIRE_D / 2)
    py0 = 56 - 2 - USBC_PCB[1] - FIT
    pw = USBC_PCB[0] / 2 + FIT
    pocket = box(cx - pw, cx + pw, py0, 54, zt1 + 0.5, zt1 + 15)                             # open top
    pz_ = zt1 + 0.5 + USBC_PORT_Z
    port = box(cx - 5.2, cx + 5.2, 53, 57, pz_ - 1.9, pz_ + 1.9)                              # USB-C opening, +y face
    set_screw = cyl('x', [cx + MAST_COL_R / 2, MAST_COL_Y, top - 12], SCREW_PILOT / 2, MAST_COL_R + 2)
    clamp_scr = [cyl('z', [w * f, 6.0, (zt0 + zt1) / 2], SCREW_PILOT / 2, 10) for f in (0.2, 0.8)]
    return D(body, socket, wire_v, wire_h, pocket, port, set_screw, *clamp_scr)

# ----------------------------------------------------------------- elbow (vertical tube -> horizontal tube)
ELBOW = dict(x=(-14.0, 14.0), y=(-31.0, 14.0), z=(-22.0, 24.0), vtop=4.0, axis_z=14.0, hend=-5.0)

def elbow():
    e = ELBOW
    body = box(e['x'][0], e['x'][1], e['y'][0], e['y'][1], e['z'][0], e['z'][1])
    vert = cylz(0, 0, e['z'][0] - 1, e['vtop'], ROD_HOLE / 2)               # blind, rod top at z=+4
    wire_v = cylz(0, 0, e['vtop'] - 1, 12.0, 3.5)
    cavity = box(-4, 4, e['hend'] - 1.0, 3.5, 10.0, 18.5)                    # wires turn the corner here
    horiz = cyly(0, e['axis_z'], e['y'][0] - 1, e['hend'], ROD_HOLE / 2)     # blind, tube end at y=-5
    s_v = cyl('x', [7, 0, -10.0], SCREW_PILOT / 2, 16)
    s_h = cyl('x', [7, -18.0, e['axis_z']], SCREW_PILOT / 2, 16)
    return D(body, vert, wire_v, cavity, horiz, s_v, s_h)

# ----------------------------------------------------------------- camera head (Pi Zero 2 W + camera)
def camera_head():
    pt = HEAD_PT
    plate = box(-45, 45, -45, 52, 0, pt)
    blk = box(-15, 15, 18, 52, pt, pt + 24)
    # Pi Zero standoffs (M2.5), long axis along x
    posts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            posts.append(cylz(sx * PI_HOLES[0] / 2, PI_CY + sy * PI_HOLES[1] / 2, pt - 0.2, pt + 3.0, 3.0))
    body = U(plate, blk, *posts)
    z_rod = HEAD_ROD_Z
    # camera
    win = box(-CAM_WINDOW / 2, CAM_WINDOW / 2, -CAM_WINDOW / 2, CAM_WINDOW / 2, -1, pt + 1)
    hx, y_lo = CAM_HOLES[0] / 2, -CAM_LENS_FROM_EDGE + 2.0     # lens on the head axis; camera bottom edge toward -y
    mh = [cyl('z', [sx * hx, y, pt / 2], 0.9, pt + 2) for sx in (-1, 1) for y in (y_lo, y_lo + CAM_HOLES[1])]
    flex = box(-CAM_FLEX[0] / 2, CAM_FLEX[0] / 2, -CAM_FLEX[1], -CAM_WINDOW / 2 + 0.01, -1, pt + 1)
    pilots = [cylz(sx * PI_HOLES[0] / 2, PI_CY + sy * PI_HOLES[1] / 2, 1.6, pt + 3.5, 1.05)   # stops 0.6 above the LED groove
              for sx in (-1, 1) for sy in (-1, 1)]
    ring = D(box(-37, 37, -37, 37, -0.01, 1.0), box(-28, 28, -28, 28, -1, 2))   # LED strip groove (underside)
    # rod socket (tube arrives from +y), floor at y=25
    sock = cyly(0, z_rod, HEAD_SOCKET_FLOOR_Y, 53, ROD_HOLE / 2)
    set_screw = cyl('x', [7.5, 40, z_rod], SCREW_PILOT / 2, 17)
    # wire path: socket floor -> chamber -> down to plate -> notch out of -y face -> groove -> Pi 5V pins
    chamber = box(-4, 4, 20, 26, pt, z_rod + 4.5)
    notch = box(-4, 4, 17, 21, pt, pt + 4)
    g_top = pt - 1.2
    groove_x = box(-39, 4, 13.4, 16.6, g_top, pt + 0.01)
    groove_y = box(-40.6, -37.4, -14, 16.6, g_top, pt + 0.01)
    return D(body, win, flex, ring, sock, set_screw, chamber, notch, groove_x, groove_y, *mh, *pilots)

# ----------------------------------------------------------------- wire path (for verification / preview)
def wire_paths():
    """Polylines, in each part's LOCAL coordinates, that the 5 V pair must follow (must be void)."""
    zt1, wire_z, floor, top, zbot = _mast_z()
    cx = MAST_W / 2
    pt = HEAD_PT
    e = ELBOW
    mast = [(cx, 46, wire_z), (cx, MAST_COL_Y, wire_z), (cx, MAST_COL_Y, floor + 0.5)]
    elb = [(0, 0, e['vtop'] - 0.5), (0, 0, 11.0), (0, -1.0, e['axis_z']), (0, e['hend'] - 0.5, e['axis_z'])]
    head = [(0, 26, HEAD_ROD_Z), (0, 22.5, HEAD_ROD_Z), (0, 22.5, pt + 2), (0, 19.0, pt + 2),
            (0, 15.0, pt - 0.6), (-39.0, 15.0, pt - 0.6), (-39.0, -12.0, pt - 0.6)]
    return dict(mast_base_x1=mast, elbow_cross_clamp_x1=elb, camera_head_x1=head)

def check_paths(parts):
    ok = True
    for name, pts in wire_paths().items():
        pts = np.array(pts, float)
        samples = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.5))
            samples.append(np.linspace(a, b, n))
        s = np.vstack(samples)
        inside = parts[name].contains(s)
        bad = int(inside.sum())
        print(f'  wire path through {name:24s}: {len(s):4d} samples, {bad} inside solid ->', 'OK' if bad == 0 else 'BLOCKED')
        ok &= bad == 0
    return ok

# ----------------------------------------------------------------- placement (global frame)
def _rz(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])

def _tr(x, y, z):
    return trimesh.transformations.translation_matrix([x, y, z])

def side_angle():
    return 90.0 if MAST_SIDE == 'left' else -90.0

def mast_col_xy():
    """Global x,y of the boom's vertical axis."""
    x = (MAST_DEPTH - MAST_COL_Y) if MAST_SIDE == 'left' else BOARD_W - (MAST_DEPTH - MAST_COL_Y)
    return x, BOARD_L / 2

def mast_base_xf():
    cx = MAST_W / 2
    if MAST_SIDE == 'left':
        return _tr(MAST_DEPTH, BOARD_L / 2 - cx, 0) @ _rz(90)
    return _tr(BOARD_W - MAST_DEPTH, BOARD_L / 2 + cx, 0) @ _rz(-90)

def rod_lengths():
    zt1, wire_z, floor, top, zbot = _mast_z()
    e = ELBOW
    rod_axis_z = CAM_LENS_H + CAM_LENS_DROP + HEAD_ROD_Z       # elbow/head rod axis height
    rod_top = rod_axis_z - (e['axis_z'] - e['vtop'])
    v_len = rod_top - floor
    mx, my = mast_col_xy()
    s = 1.0 if MAST_SIDE == 'left' else -1.0                  # +x pointing toward the page
    x_elbow_end = mx + s * (-e['hend'])
    x_head_floor = BOARD_W / 2 - s * HEAD_SOCKET_FLOOR_Y
    h_len = abs(x_head_floor - x_elbow_end)
    return v_len, h_len, (mx, my), (BOARD_W / 2, BOARD_L / 2), rod_axis_z, rod_top, floor

def elbow_xf():
    v, h, (mx, my), c, rod_axis_z, rod_top, floor = rod_lengths()
    return _tr(mx, my, rod_top - ELBOW['vtop']) @ _rz(side_angle())

def head_xf():
    v, h, m, (cx, cy), rod_axis_z, rod_top, floor = rod_lengths()
    return _tr(cx, cy, rod_axis_z - HEAD_ROD_Z) @ _rz(side_angle())

def clip_xf():
    return _tr(BOARD_W / 2 - CLIP_L / 2, BOARD_L - CLIP_EDGE, 0)

# ----------------------------------------------------------------- build
def clip_checks(parts):
    """Sweep the lever about the pivot rod from closed to CLIP_LIFT open and check collisions and spring lengths."""
    base, lever = parts['clip_base_x1'], parts['clip_lever_x1']
    py, pz = CLIP_PIV
    theta, front, rear, z_under = _clip_geom()
    board = box(-5, CLIP_L + 5, -40, CLIP_EDGE, -BOARD_T, 0)
    def rot(deg):
        return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0], [0, py, pz])
    def ovl(a, b):
        i = trimesh.boolean.intersection([a, b], engine='manifold')
        return i.volume if len(i.faces) else 0.0
    # closed: jaw foot flat on the paper
    fb = lever.faces[(lever.face_normals[:, 2] < -0.999) & (lever.triangles_center[:, 1] < 0)]
    fz = lever.vertices[fb.ravel(), 2]
    flat = len(fb) > 0 and np.ptp(fz) < 1e-6 and abs(fz.min() - PAPER_T) < 1e-6
    # opening angle for CLIP_LIFT of clearance over the paper (paper top edge at y=0)
    pts = np.vstack([lever.vertices, lever.sample(20000, seed=1)])
    def gap(deg):
        q = trimesh.transform_points(pts, rot(deg))
        return q[q[:, 1] < 0, 2].min() - PAPER_T
    open_deg = next(-a for a in np.arange(0, 60, 0.5) if gap(-a) >= CLIP_LIFT)
    worst = worst_s = 0.0
    for deg in np.linspace(0, open_deg - 3.0, 9):          # 3 deg of over-travel past the required opening
        L = lever.copy(); L.apply_transform(rot(deg))
        worst = max(worst, ovl(L, base), ovl(L, board))
        for sp in clip_spring_meshes(deg, r=2.9):           # 6 mm spring, 0.1 mm allowance for tangent contact
            worst_s = max(worst_s, ovl(sp, L), ovl(sp, base))
    lens = lambda deg: [np.linalg.norm(t - b) for b, t in clip_spring_ends(deg)]
    l_closed, l_open = min(lens(0)), min(lens(open_deg))
    preload = CLIP_SPRING_FREE - max(lens(0))
    mis = max(abs(CLIP_DIMPLE_TILT), abs(open_deg + CLIP_DIMPLE_TILT))
    ok_col, ok_pre, ok_open, ok_spr = worst < 0.5, preload >= 1.5, l_open >= 5.5, worst_s < 0.5
    print(f'  clip: jaw foot flat on paper (z={PAPER_T}) when closed ->', 'OK' if flat else 'NOT FLAT')
    print(f'  clip: {CLIP_LIFT:.0f} mm jaw opening needs {-open_deg:.1f} deg; lever vs base/board over 0..{3 - open_deg:.0f} deg '
          f'worst overlap {worst:.2f} mm3 ->', 'OK' if ok_col else 'COLLIDES')
    print(f'  clip: spring length closed {l_closed:.1f} mm (preload {preload:.1f} mm on a {CLIP_SPRING_FREE:.0f} mm spring) ->',
          'OK' if ok_pre else 'NO CLAMP FORCE')
    print(f'  clip: spring length at {CLIP_LIFT:.0f} mm opening {l_open:.1f} mm (must stay above solid height, ~5 mm) ->',
          'OK' if ok_open else 'GOES SOLID')
    print(f'  clip: spring end misalignment max {mis:.0f} deg (dimple pre-tilt {CLIP_DIMPLE_TILT:.0f} deg); '
          f'springs vs seats/lever over the sweep worst overlap {worst_s:.2f} mm3 ->', 'OK' if ok_spr else 'BINDS')
    print(f'  clip: jaw overlaps paper by {-(CLIP_TIP[0] - 4.0):.0f} mm, lever angle {np.degrees(theta):.1f} deg')
    return ok_col and ok_pre and ok_open and ok_spr and flat

def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    parts = {
        'corner_holder_x4': corner_holder(),
        'clip_base_x1': clip_base(),
        'clip_lever_x1': clip_lever(),
        'mast_base_x1': mast_base(),
        'elbow_cross_clamp_x1': elbow(),
        'camera_head_x1': camera_head(),
    }
    for name, m in parts.items():
        assert m.is_watertight, f'{name} not watertight'
        m.export(os.path.join(outdir, name + '.stl'))
    return parts

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'out'
    parts = build(out)
    for n, m in parts.items():
        e = m.bounds[1] - m.bounds[0]
        print(f'{n:26s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm   vol {m.volume/1000:6.1f} cm3')
    print()
    ok = check_paths(parts)
    ok &= clip_checks(parts)
    v, h, (mx, my), (cx, cy), *_ = rod_lengths()
    print(f'\nvertical tube   {v:.1f} mm  (blind both ends: cut to this)')
    print(f'horizontal tube {h:.1f} mm  (blind both ends: cut to this)')
    print(f'boom on the {MAST_SIDE} edge at ({mx:.0f}, {my:.0f}); camera over ({cx:.0f}, {cy:.0f}); mast socket depth {mast_socket_depth():.0f} mm')
    sys.exit(0 if ok else 1)
