"""
Assemble the camera clipboard in Blender from the exported STLs.

Run inside Blender (Scripting tab -> open -> Run) on a new/empty file, or headless:

    blender --background --python blender_assemble.py -- <assembly_dir> [--render out.png] [--save out.blend] [--keep]

<assembly_dir> is the `assembly/` folder (contains assembly.json + one STL per part). If omitted, the script
looks for `assembly/` next to this file, then for the ASSEMBLY_DIR constant below.
Units: 1 Blender unit = 1 mm (scene unit scale 0.001). Every part is its own object so you can move it.
"""
import bpy, json, os, sys, math
from mathutils import Vector

ASSEMBLY_DIR = ""        # set this if you run it from the Text Editor and the folder isn't next to the script

# ----------------------------------------------------------------------------- args
def _cli():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    out = dict(dir=None, render=None, save=None, keep=False)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--render': out['render'] = argv[i + 1]; i += 2
        elif a == '--save': out['save'] = argv[i + 1]; i += 2
        elif a == '--keep': out['keep'] = True; i += 1
        else: out['dir'] = a; i += 1
    return out

def _find_dir(cli_dir):
    here = os.path.dirname(os.path.abspath(globals().get('__file__', bpy.data.filepath or '.')))
    for d in (cli_dir, ASSEMBLY_DIR, os.path.join(here, 'assembly'), os.path.join(here, 'out', 'assembly')):
        if d and os.path.isfile(os.path.join(d, 'assembly.json')):
            return d
    raise SystemExit('assembly.json not found: pass the assembly folder or set ASSEMBLY_DIR')

# ----------------------------------------------------------------------------- helpers
def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex_rgba(h):
    h = h.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0)

def make_material(name, color, rough, metal):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = hex_rgba(color)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    mat.diffuse_color = hex_rgba(color)
    return mat

def import_stl(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, 'stl_import'):            # Blender 4.0+
        bpy.ops.wm.stl_import(filepath=path)
    else:                                            # Blender 3.x
        bpy.ops.import_mesh.stl(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    return new[0]

def get_collection(name, parent):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in [x.name for x in parent.children]:
        parent.children.link(c)
    return c

def move_to(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)

# ----------------------------------------------------------------------------- build
def main():
    cli = _cli()
    d = _find_dir(cli['dir'])
    data = json.load(open(os.path.join(d, 'assembly.json')))
    BW, BL, BT = data['board']

    if not cli['keep']:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights, bpy.data.cameras):
            for x in list(blk):
                blk.remove(x)

    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = 'MILLIMETERS'

    root = get_collection('Camera Clipboard', scene.collection)
    groups = {k: get_collection(k, root) for k in ('Printed parts', 'Hardware', 'Electronics', 'Board and paper')}
    kind_group = dict(part='Printed parts', metal='Hardware', tube='Hardware', pcb='Electronics', board='Board and paper', paper='Board and paper')

    for it in data['items']:
        obj = import_stl(os.path.join(d, it['file']))
        obj.name = it['name']
        obj.data.name = it['name'] + '_mesh'
        obj.data.materials.append(make_material('mat_' + it['name'], it['color'], it['rough'], it['metal']))
        move_to(obj, groups[kind_group.get(it['kind'], 'Printed parts')])
        # origin to the part's centre, location unchanged (so parts can be moved/rotated about their middle)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # 5 V power cable as a bevelled poly curve
    pts = data['cable']
    cu = bpy.data.curves.new('power_cable', 'CURVE')
    cu.dimensions = '3D'; cu.bevel_depth = 0.8; cu.bevel_resolution = 3
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, co in zip(sp.points, pts):
        p.co = (co[0], co[1], co[2], 1.0)
    cable = bpy.data.objects.new('power_cable', cu)
    cable.data.materials.append(make_material('mat_cable', '#ff2020', 0.4, 0.0))
    groups['Electronics'].objects.link(cable)

    # table
    bpy.ops.mesh.primitive_plane_add(size=4000, location=(BW / 2, BL / 2, data['ground_z']))
    table = bpy.context.active_object; table.name = 'table'
    table.data.materials.append(make_material('mat_table', '#8a8f94', 0.9, 0.0))
    move_to(table, groups['Board and paper'])

    # world + lights
    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.82, 0.85, 0.9, 1.0)
    bg.inputs['Strength'].default_value = 0.8

    sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3.0; sun.angle = math.radians(4)
    sun_o = bpy.data.objects.new('sun', sun); root.objects.link(sun_o)
    sun_o.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))
    area = bpy.data.lights.new('softbox', 'AREA'); area.energy = 3.0e5; area.size = 500
    area_o = bpy.data.objects.new('softbox', area); root.objects.link(area_o)
    area_o.location = (BW / 2 + 100, BL / 2 - 100, 600)

    # camera framing the whole assembly
    cam = bpy.data.cameras.new('cam'); cam.lens = 40; cam.clip_end = 20000
    cam_o = bpy.data.objects.new('cam', cam); root.objects.link(cam_o)
    target = Vector((BW / 2 - 20, BL / 2, 90))
    cam_o.location = Vector((BW / 2 + 520, -430, 430))
    cam_o.rotation_euler = (target - cam_o.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam_o

    # render settings
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 48
        scene.cycles.use_denoising = True
    except Exception:
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in \
            bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys() else 'BLENDER_EEVEE'
    scene.view_settings.view_transform = 'Standard'

    if cli['save']:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(cli['save']))
    if cli['render']:
        scene.render.filepath = os.path.abspath(cli['render'])
        bpy.ops.render.render(write_still=True)
    print(f"Imported {len(data['items'])} parts from {d}")

main()
