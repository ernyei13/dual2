"""Record the latest rollout and dump everything a 3D artist needs to rebuild it.

Writes ./export/:
  parts/<body>.obj        robot part meshes in their body frame (metres, Z-up), one material group per material
  rollout.json            hierarchy, joints, materials, bar dimensions, per-frame local + world body transforms
  joint_angles.csv        per frame: time, base position/quaternion, 8 joint angles (deg)
  body_world_transforms.csv  per frame: world position + quaternion of every body
Run build_c4d.py in Blender afterwards to get FBX / GLB / .blend.
"""
import csv, json
from pathlib import Path
import numpy as np
from search import a, d, mj, reset, stage, out
from model import SPACING

FPS = 50                      # one frame per 20 ms control step
INTRO_S = 1.0                 # the videos hold the start pose for 1 s
EXP = out / 'export'; (EXP / 'parts').mkdir(parents=True, exist_ok=True)

bodies = list(range(1, a.nbody))
names = [a.body(b).name for b in bodies]

# ---------- meshes (visual geoms only: skip collision group 3) ----------
def quat_mat(q):
    m = np.zeros(9); mj.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)

mats = {}
for b, name in zip(bodies, names):
    lines = [f'# {name}: body frame, metres, Z-up\n', 'mtllib ../materials.mtl\n']; off = 1
    for g in range(a.ngeom):
        if a.geom_bodyid[g] != b or a.geom_group[g] == 3 or a.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = a.geom_dataid[g]
        v = a.mesh_vert[a.mesh_vertadr[mid]:a.mesh_vertadr[mid] + a.mesh_vertnum[mid]]
        f = a.mesh_face[a.mesh_faceadr[mid]:a.mesh_faceadr[mid] + a.mesh_facenum[mid]]
        v = v @ quat_mat(a.geom_quat[g]).T + a.geom_pos[g]
        mat = a.mat(a.geom_matid[g]).name if a.geom_matid[g] >= 0 else 'default'
        mats[mat] = (a.mat_rgba[a.geom_matid[g]] if a.geom_matid[g] >= 0 else a.geom_rgba[g]).tolist()
        lines.append(f'o {name}_{a.mesh(mid).name}_{g}\nusemtl {mat}\n')
        lines += [f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in v]
        lines += [f'f {i + off} {j + off} {k + off}\n' for i, j, k in f]
        off += len(v)
    (EXP / 'parts' / f'{name}.obj').write_text(''.join(lines))
with open(EXP / 'materials.mtl', 'w') as fh:
    for m, c in mats.items():
        fh.write(f'newmtl {m}\nKd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\nd {c[3]:.3f}\n\n')

# ---------- rollout ----------
frames = []
def grab():
    frames.append(dict(t=float(d.time), qpos=d.qpos.copy(), xpos=d.xpos[bodies].copy(), xquat=d.xquat[bodies].copy()))
def cb(ph, k, f, j, hold): grab()

reset(0)
for _ in range(int(INTRO_S * FPS)): grab()
ok_all = True
for ph in range(4):
    _, ok = stage(ph, np.load(out / f'stage{ph}.npy'), cb, tail=100 if ph == 3 else 0)
    if not ok: ok_all = False; print('FAILED at transfer', ph + 1); break
grab()

def local(i, fr):
    """Transform of body i relative to its parent (world for the base)."""
    b = bodies[i]; p = a.body_parentid[b]
    pos, q = fr['xpos'][i], fr['xquat'][i]
    if p == 0: return pos.tolist(), q.tolist()
    j = bodies.index(p); pp, pq = fr['xpos'][j], fr['xquat'][j]
    inv = np.zeros(4); mj.mju_negQuat(inv, pq)
    lq = np.zeros(4); mj.mju_mulQuat(lq, inv, q)
    lp = np.zeros(3); mj.mju_rotVecQuat(lp, pos - pp, inv)
    return lp.tolist(), lq.tolist()

joints = [dict(name=a.joint(j).name, body=a.body(a.jnt_bodyid[j]).name, type=['free', 'ball', 'slide', 'hinge'][a.jnt_type[j]],
               axis_body_frame=a.jnt_axis[j].tolist(), anchor_body_frame=a.jnt_pos[j].tolist(),
               range_deg=np.degrees(a.jnt_range[j]).tolist(), qpos_index=int(a.jnt_qposadr[j])) for j in range(a.njnt)]
rods = [dict(name=f'rod{i + 1}', shape='capsule', radius=.011, start=[x, -.16, .55], end=[x, .16, .55], collides=True)
        for i, x in enumerate(np.arange(5) * SPACING)]
frame_parts = ([dict(name=f'rail_{s}', shape='cylinder', radius=.012, start=[-.30, y, .62], end=[4 * SPACING + .30, y, .62], collides=False,
                     note='visual only; extend freely') for s, y in (('left', -.16), ('right', .16))] +
               [dict(name=f'drop_link_rod{i + 1}_{s}', shape='cylinder', radius=.006, start=[x, y, .55], end=[x, y, .62], collides=False)
                for i, x in enumerate(np.arange(5) * SPACING) for s, y in (('left', -.16), ('right', .16))])
info = dict(
    units='metres, Z-up, right-handed (MuJoCo world frame). x = travel direction, rods run along y.',
    fps=FPS, frame_count=len(frames), intro_hold_s=INTRO_S, rollout_succeeded=ok_all,
    quaternion_order='w x y z',
    bodies=[dict(name=n, parent=(a.body(a.body_parentid[b]).name if a.body_parentid[b] else None), mesh=f'parts/{n}.obj',
                 mass_kg=float(a.body_mass[b])) for b, n in zip(bodies, names)],
    joints=joints, materials=mats, bars=dict(spacing_m=SPACING, rods=rods, frame=frame_parts),
    frames=[dict(t=round(fr['t'], 4), local={n: dict(zip(('pos', 'quat'), local(i, fr))) for i, n in enumerate(names)})
            for fr in frames])
(EXP / 'rollout.json').write_text(json.dumps(info))

hinges = [j for j in joints if j['type'] == 'hinge']
with open(EXP / 'joint_angles.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['frame', 'time_s', 'base_x', 'base_y', 'base_z', 'base_qw', 'base_qx', 'base_qy', 'base_qz'] + [j['name'] + '_deg' for j in hinges])
    for k, fr in enumerate(frames):
        w.writerow([k, round(fr['t'], 4)] + np.round(fr['qpos'][:7], 6).tolist() + np.round(np.degrees([fr['qpos'][j['qpos_index']] for j in hinges]), 4).tolist())
with open(EXP / 'body_world_transforms.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['frame', 'time_s'] + [f'{n}_{c}' for n in names for c in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')])
    for k, fr in enumerate(frames):
        w.writerow([k, round(fr['t'], 4)] + np.round(np.hstack([np.r_[fr['xpos'][i], fr['xquat'][i]] for i in range(len(names))]), 6).tolist())
print('frames', len(frames), 'duration', round(len(frames) / FPS, 2), 's, success', ok_all)
