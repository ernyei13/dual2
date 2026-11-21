"""Side view of the full traversal, no overlay, white background, darker green robot, open-ended rails -> videos/side_clean.mp4"""
import subprocess
import numpy as np
import mujoco as mj
import render3 as R
from search import a, d, reset, stage, out

r = mj.Renderer(R.ra, height=R.H, width=R.W)
pipe = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{R.W}x{R.H}', '-r', str(R.FPS),
                         '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                         str(out / 'videos' / 'side_clean.mp4')], stdin=subprocess.PIPE)
fl = R.ra.geom('floor').id; R.ra.geom_matid[fl] = -1; R.ra.geom_rgba[fl] = [1, 1, 1, 0]   # alpha 0 -> floor not drawn
WHITE = np.full((R.H, R.W, 3), 255, np.float32)
R.ra.mat_rgba[R.ra.mat('green').id] = [.20, .48, .07, 1]   # original green (.33 .79 .11) darkened

RAIL = (.42, .45, .49, 1)
def dress(scn):
    """Top rails run far past the frame on both sides; no posts to the ground."""
    for sgn in (-1, 1):
        R.add_cyl(scn, (-6, sgn * .16, .62), (7, sgn * .16, .62), .012, RAIL)
        for x in R.ROD_X:
            R.add_cyl(scn, (x, sgn * .16, .55), (x, sgn * .16, .62), .006, (.62, .65, .70, 1))

def shot(cam):
    R.sync()
    r.update_scene(R.rd, camera=cam); dress(r.scene)
    rgb = r.render().astype(np.float32)
    r.enable_depth_rendering(); r.update_scene(R.rd, camera=cam); dress(r.scene)
    dep = r.render(); r.disable_depth_rendering()
    return np.where((dep > 30)[..., None], WHITE, rgb).astype(np.uint8)
track = [None]; n = [0]

def frame():
    x = float(np.clip(d.subtree_com[1, 0], 0, 4 * R.SPACING))
    track[0] = x if track[0] is None else track[0] + .06 * (x - track[0])
    R.ra.vis.global_.fovy = 15
    pipe.stdin.write(shot(R.camera((track[0], 0, .38), 3.6, 90, 3)).tobytes())

def cb(ph, k, f, j, hold):
    n[0] += 1
    if n[0] % 2: frame()   # 20 ms control step -> 25 fps real time

reset(0)
for _ in range(int(1.0 * R.FPS)): frame()
for ph in range(4):
    _, ok = stage(ph, np.load(out / f'stage{ph}.npy'), cb, tail=100 if ph == 3 else 0)
    if not ok: print('FAILED at transfer', ph + 1); break
pipe.stdin.close(); assert pipe.wait() == 0; r.close()
print('done, sim time', round(d.time, 2))
