"""Blender (5.x) script: build the animated scene from ./export and save FBX / GLB / .blend.

  /Applications/Blender.app/Contents/MacOS/Blender -b -P build_c4d.py

Hierarchy: DUAL2_rig (static, scene root)
             base (animated: world position + rotation)
               arm1_link1 (animated: rotation about its joint) ... down each arm
           every link empty carries its CAD mesh as a child with identity offset.
Bars: Bars/rod1..rod5 (capsules), Bars/rail_left|right, Bars/drop_link_* (static).
"""
import json, math
from pathlib import Path
import bpy
from mathutils import Quaternion, Vector

HERE = Path(__file__).resolve().parent
EXP = HERE / 'export'
info = json.loads((EXP / 'rollout.json').read_text())

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.fps = info['fps']
scn.frame_start, scn.frame_end = 0, info['frame_count'] - 1   # frame k = time k/fps
scn.unit_settings.system = 'METRIC'; scn.unit_settings.scale_length = 1.0

def material(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = .45
    m.diffuse_color = rgba
    return m

mats = {k: material(k, v) for k, v in info['materials'].items()}

def empty(name, parent=None, coll=None):
    o = bpy.data.objects.new(name, None); o.empty_display_size = .03; o.empty_display_type = 'ARROWS'
    (coll or scn.collection).objects.link(o)
    if parent: o.parent = parent
    return o

robot = bpy.data.collections.new('DUAL2'); scn.collection.children.link(robot)
root = empty('DUAL2_rig', coll=robot)
nodes = {}
for b in info['bodies']:
    e = empty(b['name'], nodes.get(b['parent'], root), robot)
    e.rotation_mode = 'QUATERNION'
    nodes[b['name']] = e
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(EXP / b['mesh']), forward_axis='Y', up_axis='Z')   # keep raw Z-up coordinates
    parts = [o for o in bpy.data.objects if o not in before]
    for o in parts:
        for c in o.users_collection: c.objects.unlink(o)
        robot.objects.link(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in parts: o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1: bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    mesh.name = mesh.data.name = b['name'] + '_mesh'
    for slot in mesh.material_slots:                                  # use our materials (OBJ import makes copies)
        base = slot.material.name.split('.')[0] if slot.material else None
        if base in mats: slot.material = mats[base]
    mesh.parent = e; mesh.matrix_parent_inverse.identity()
    mesh.location = (0, 0, 0); mesh.rotation_euler = (0, 0, 0)
    for poly in mesh.data.polygons: poly.use_smooth = False

# ---------- animation ----------
for k, fr in enumerate(info['frames']):
    for name, tr in fr['local'].items():
        e = nodes[name]
        e.location = Vector(tr['pos']); e.rotation_quaternion = Quaternion(tr['quat'])
        e.keyframe_insert('location', frame=k); e.keyframe_insert('rotation_quaternion', frame=k)
for e in nodes.values():
    for fc in e.animation_data.action.fcurves if hasattr(e.animation_data.action, 'fcurves') else []:
        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'

# ---------- bars ----------
bars = bpy.data.collections.new('Bars'); scn.collection.children.link(bars)
rod_mat = material('rod', (.85, .45, .15, 1)); frame_mat = material('bar_frame', (.42, .45, .49, 1))
def cylinder(spec, mat, caps):
    p0, p1 = Vector(spec['start']), Vector(spec['end']); axis = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=spec['radius'], depth=axis.length, location=(p0 + p1) / 2)
    o = bpy.context.active_object
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(axis.normalized())
    if caps:                                                            # capsule = cylinder + two hemispherical caps
        objs = [o]
        for p in (p0, p1):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=spec['radius'], location=p); objs.append(bpy.context.active_object)
        bpy.ops.object.select_all(action='DESELECT')
        for x in objs: x.select_set(True)
        bpy.context.view_layer.objects.active = o; bpy.ops.object.join()
    o.name = o.data.name = spec['name']; o.data.materials.append(mat)
    for c in o.users_collection: c.objects.unlink(o)
    bars.objects.link(o)
    bpy.ops.object.shade_smooth()
for r in info['bars']['rods']: cylinder(r, rod_mat, caps=True)
for f in info['bars']['frame']: cylinder(f, frame_mat, caps=False)

# ---------- export ----------
scn.frame_set(0)
bpy.ops.wm.save_as_mainfile(filepath=str(EXP / 'dual2_five_rod.blend'))
bpy.ops.export_scene.fbx(filepath=str(EXP / 'dual2_five_rod.fbx'), object_types={'EMPTY', 'MESH'}, apply_unit_scale=True,
                         bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                         bake_anim_force_startend_keying=True, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
                         add_leaf_bones=False, path_mode='COPY', embed_textures=False)
bpy.ops.export_scene.gltf(filepath=str(EXP / 'dual2_five_rod.glb'), export_format='GLB', export_animations=True,
                          export_frame_range=True, export_force_sampling=True)

# ---------- self-check: world pose of a gripper vs. MuJoCo ----------
import csv
rows = list(csv.reader(open(EXP / 'body_world_transforms.csv')))
hdr = rows[0]; err = 0
for k in (0, info['frame_count'] // 2, info['frame_count'] - 1):
    scn.frame_set(k)
    for name in ('arm1_gripper', 'arm2_gripper', 'base'):
        ref = Vector([float(rows[k + 1][hdr.index(f'{name}_{c}')]) for c in 'xyz'])
        err = max(err, (nodes[name].matrix_world.translation - ref).length)
print(f'SELF-CHECK max world position error vs MuJoCo: {err * 1000:.4f} mm')
