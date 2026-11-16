"""Render one uninterrupted rollout of the trained policy from three cameras.

Outputs (in ./video): side.mp4, three_quarter.mp4, gripper_closeup.mp4, triptych.mp4
"""
import ctypes, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from search import a, d, mj, reset, stage, out, steps, palm

W, H = 1920, 1080
FPS = 25
OUT = out / 'video'; OUT.mkdir(exist_ok=True)
ROD_COLORS = ['#5cc8b8', '#5aaef0', '#f2b54f', '#b48cf0', '#f5826c']
from model import SPACING
ROD_X = np.arange(5) * SPACING

# ---------- scene dressing (separate render model; physics model untouched) ----------
import copy, xml.etree.ElementTree as E
import model as M
rroot = copy.deepcopy(M.root)
vis = rroot.find('visual')
if vis is None: vis = E.SubElement(rroot, 'visual')
for tag, attrs in {'global': dict(offwidth=str(W), offheight=str(H)),
                   'quality': dict(shadowsize='8192', offsamples='8'),
                   'headlight': dict(ambient='.30 .30 .32', diffuse='.40 .40 .40', specular='.1 .1 .1'),
                   'map': dict(fogstart='2.5', fogend='7', znear='.01'),
                   'rgba': dict(fog='.80 .85 .90 1')}.items():
    el = vis.find(tag)
    if el is None: el = E.SubElement(vis, tag)
    el.attrib.update(attrs)
rroot.find('.//texture[@name="grid"]').attrib.update(rgb1='.50 .54 .58', rgb2='.60 .64 .68')
rroot.find('.//material[@name="grid_mat"]').set('texrepeat', '60 60')
rroot.find('.//geom[@name="floor"]').set('size', '12 12 .1')
ra = mj.MjModel.from_xml_string(E.tostring(rroot, encoding='unicode')); rd = mj.MjData(ra)
assert ra.nq == a.nq
for i, h in enumerate(ROD_COLORS):
    ra.geom_rgba[ra.geom('rung%d' % i).id, :3] = [int(h[n:n + 2], 16) / 255 for n in (1, 3, 5)]
ra.geom_rgba[ra.geom_group == 3, 3] = 0          # hide collision pieces
ra.site_rgba[:, 3] = 0
def sync():
    rd.qpos[:] = d.qpos; rd.time = d.time; mj.mj_kinematics(ra, rd)

def add_box(scn, pos, size, rgba):
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_BOX, np.array(size, float), np.array(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))
    scn.ngeom += 1

def add_cyl(scn, p0, p1, r, rgba):
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, np.float32))
    mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, r, np.array(p0, float), np.array(p1, float))
    scn.ngeom += 1

FRAME = (.30, .33, .37, 1)
def dress(scn):
    """Monkey-bar frame: rails above the rods, rods hang from short drop links."""
    x0, x1, y, zr = -.30, 4 * SPACING + .30, .16, .62
    for sgn in (-1, 1):
        add_cyl(scn, (x0, sgn * y, zr), (x1, sgn * y, zr), .012, FRAME)        # top rails
        for x in (x0, x1):
            add_cyl(scn, (x, sgn * y, 0), (x, sgn * y, zr), .014, FRAME)       # posts
            add_box(scn, (x, sgn * y, .004), (.045, .045, .004), FRAME)        # feet
        for x in ROD_X:
            add_cyl(scn, (x, sgn * y, .55), (x, sgn * y, zr), .006, (.62, .65, .70, 1))  # drop links
    for x in (x0, x1):
        add_cyl(scn, (x, -y, zr), (x, y, zr), .012, FRAME)                     # cross beams

def camera(look, dist, az, el):
    c = mj.MjvCamera()
    xyz = np.asarray(look, np.float64)
    ctypes.memmove(c.lookat.ctypes.data, xyz.ctypes.data, xyz.nbytes)
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c

def sky(h, w):
    t = np.linspace(0, 1, h)[:, None, None]
    top = np.array([.36, .50, .66]); bot = np.array([.80, .85, .90])
    return np.broadcast_to(((1 - t) * top + t * bot) * 255, (h, w, 3)).astype(np.float32)
SKY = sky(H, W)

def font(size, bold=False):
    for p in ['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc']:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()
F = {k: font(k) for k in (20, 24, 28, 34)}; FB = {k: font(k, True) for k in (24, 28, 40, 56)}

LABELS = ['Hand 1 reaches rod 2', 'Hand 2 reaches rod 3', 'Hand 1 reaches rod 4', 'Hand 2 reaches rod 5']

class Cam:
    def __init__(self, name, title, fn, fovy=45):
        self.name, self.title, self.fn, self.fovy = name, title, fn, fovy
        self.pipe = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                                      '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                      str(OUT / f'{name}.mp4')], stdin=subprocess.PIPE)

class Recorder:
    def __init__(self):
        self.r = mj.Renderer(ra, height=H, width=W)
        self.track = None
        self.cams = [
            Cam('side', 'Side view', lambda x: camera((x, 0, .38), 3.6, 90, 3), 15),
            Cam('three_quarter', 'Three-quarter view', lambda x: camera((x + .02, 0, .38), 1.0, 140, -14)),
            Cam('gripper_closeup', 'Gripper close-up', lambda x: camera(self.grip, .40, 58, 10)),
        ]
        self.tri = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                                     '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                     str(OUT / 'triptych.mp4')], stdin=subprocess.PIPE)
        self.n = 0; self.done = 0; self.phase = 0; self.forces = np.zeros(5); self.status = ''

    def shot(self, cam):
        sync()
        self.r.update_scene(rd, camera=cam)
        dress(self.r.scene)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_FOG] = 1
        rgb = self.r.render().astype(np.float32)
        self.r.enable_depth_rendering(); self.r.update_scene(rd, camera=cam); dress(self.r.scene); dep = self.r.render(); self.r.disable_depth_rendering()
        bg = (dep > 30)[..., None]
        return np.where(bg, SKY, rgb).astype(np.uint8)

    def overlay(self, img, title, big=True):
        im = Image.fromarray(img).convert('RGBA'); ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        s = 1 if big else .5
        def T(x, y, t, f, c=(245, 247, 250, 255)): dr.text((x, y), t, font=f, fill=c)
        # header
        dr.rounded_rectangle((30, 28, 760, 140), 18, fill=(14, 22, 30, 190))
        T(56, 44, 'DUAL2  ·  Five-rod traversal', FB[40]); T(58, 98, f'{title}   ·   learned policy, MuJoCo, real-time', F[24], (180, 196, 210, 255))
        # time + progress
        dr.rounded_rectangle((W - 470, 28, W - 30, 140), 18, fill=(14, 22, 30, 190))
        T(W - 446, 44, f't = {d.time:5.2f} s', FB[40])
        T(W - 446, 98, f'Transfers {self.done}/4', F[24], (180, 196, 210, 255))
        for i in range(4):
            x = W - 270 + i * 58
            dr.rounded_rectangle((x, 104, x + 46, 122), 6, fill=(110, 210, 140, 255) if i < self.done else (60, 74, 88, 255))
        # bottom: status + rod forces
        dr.rounded_rectangle((30, H - 170, W - 30, H - 30), 18, fill=(14, 22, 30, 190))
        T(56, H - 150, self.status, FB[28])
        T(56, H - 104, 'Grippers carry all 8.8 N · rods 22 cm apart · self-collision on', F[24], (180, 196, 210, 255))
        T(980, H - 152, 'Load on each rod', F[24], (180, 196, 210, 255))
        for i, col in enumerate(ROD_COLORS):
            x = 980 + i * 180; y = H - 108
            rgb = tuple(int(col[n:n + 2], 16) for n in (1, 3, 5))
            dr.rounded_rectangle((x, y, x + 150, y + 18), 8, fill=(48, 60, 72, 255))
            w = int(150 * np.clip(self.forces[i] / 10, 0, 1))
            if w > 8: dr.rounded_rectangle((x, y, x + w, y + 18), 8, fill=rgb + (255,))
            T(x, y + 26, f'rod {i + 1}  {self.forces[i]:4.1f} N', F[20], (220, 228, 236, 255))
        return np.array(Image.alpha_composite(im, ov).convert('RGB'))

    def frame(self, final=False):
        x = float(np.clip(d.subtree_com[1, 0], 0, 4 * SPACING))
        self.track = x if self.track is None else self.track + .06 * (x - self.track)
        g = (palm(1)[0] + palm(2)[0]) / 2; g[1] = 0; g[2] = .52
        self.grip = g if getattr(self, 'grip', None) is None else self.grip + .08 * (g - self.grip)
        imgs = []
        for c in self.cams:
            ra.vis.global_.fovy = c.fovy; img = self.shot(c.fn(self.track)); imgs.append(img)
            c.pipe.stdin.write(self.overlay(img, c.title).tobytes())
        # triptych: big three-quarter left, side + close-up stacked right
        canvas = Image.new('RGB', (W, H), (14, 22, 30))
        big = Image.fromarray(imgs[1]).crop((240, 0, 1680, 1080)).resize((1280, 960), Image.Resampling.LANCZOS)
        canvas.paste(big, (0, 120))
        canvas.paste(Image.fromarray(imgs[0]).resize((640, 360), Image.Resampling.LANCZOS), (1280, 120))
        canvas.paste(Image.fromarray(imgs[2]).resize((640, 360), Image.Resampling.LANCZOS), (1280, 480))
        dr = ImageDraw.Draw(canvas)
        dr.text((40, 32), 'DUAL2  ·  Five-rod traversal', font=FB[40], fill=(245, 247, 250))
        dr.text((W - 360, 32), f't = {d.time:5.2f} s', font=FB[40], fill=(245, 247, 250))
        dr.text((900, 44), f'Transfers {self.done}/4', font=F[28], fill=(180, 196, 210))
        for lbl, (x, y) in [('Three-quarter', (20, 132)), ('Side', (1296, 132)), ('Gripper close-up', (1296, 492))]:
            dr.rounded_rectangle((x, y, x + 20 + 13 * len(lbl), y + 36), 8, fill=(14, 22, 30))
            dr.text((x + 10, y + 5), lbl, font=F[24], fill=(245, 247, 250))
        dr.text((1300, 870), self.status, font=FB[24], fill=(245, 247, 250))
        dr.text((1300, 910), 'Load on each rod', font=F[20], fill=(180, 196, 210))
        for i, col in enumerate(ROD_COLORS):
            y = 944 + i * 26; rgb = tuple(int(col[n:n + 2], 16) for n in (1, 3, 5))
            dr.text((1300, y - 4), f'{i + 1}', font=F[20], fill=(220, 228, 236))
            dr.rounded_rectangle((1330, y, 1730, y + 14), 6, fill=(48, 60, 72))
            w = int(400 * np.clip(self.forces[i] / 10, 0, 1))
            if w > 8: dr.rounded_rectangle((1330, y, 1330 + w, y + 14), 6, fill=rgb)
            dr.text((1745, y - 4), f'{self.forces[i]:4.1f} N', font=F[20], fill=(220, 228, 236))
        self.tri.stdin.write(canvas.tobytes())

    def cb(self, ph, k, f, j, hold):
        self.n += 1
        self.forces = f.sum(0)
        self.status = ('Done: all four transfers · hanging from rod 5' if self.done == 4
                       else f'Transfer {ph + 1}/4 — {LABELS[ph]}' + ('  ·  holding' if hold else ''))
        if self.n % 2 == 0: return               # control step is 20 ms -> 25 fps real time
        self.frame()

    def close(self):
        for p in [c.pipe for c in self.cams] + [self.tri]:
            p.stdin.close(); assert p.wait() == 0
        self.r.close()

if __name__ == '__main__':
    from train import contacts
    rec = Recorder(); reset(0)
    rec.status = 'Start: hanging from rods 1 and 2'
    rec.forces = contacts()[0].sum(0)
    for _ in range(int(1.2 * FPS)): rec.frame()          # still intro
    for ph in range(4):
        if ph == 3:
            def cb_end(p_, k, f, j, hold, _c=rec.cb):
                if hold >= 50 and rec.done < 4: rec.done = 4
                _c(p_, k, f, j, hold)
                if rec.done == 4: rec.status = 'Done: all four transfers · hanging from rod 5'
            _, ok = stage(ph, np.load(out / f'stage{ph}.npy'), cb_end, tail=100)
        else:
            _, ok = stage(ph, np.load(out / f'stage{ph}.npy'), rec.cb)
        if not ok: print('FAILED at transfer', ph + 1); break
        rec.done = ph + 1
    else:
        pass                                            # final 2 s hang is part of stage(tail=100)
    rec.close(); print('cleared', rec.done, 'sim time', round(d.time, 2))
