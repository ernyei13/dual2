from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np, mujoco as mj
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from collections import Counter
import sys
repo=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
tree=E.parse(repo/'mujoco/robot.xml'); root=tree.getroot()
root.find('compiler').set('meshdir',str(repo/'assets/parts'))
for t in root.findall('.//texture'): t.attrib.pop('colorspace',None)
for l in root.findall('.//light'): l.attrib.pop('type',None); l.set('directional','true')
world=root.find('worldbody')
for b in list(world.findall('body')):
 if b.get('name')!='base': world.remove(b)
root.remove(root.find('keyframe'));root.remove(root.find('size'))
root.find('option').set('gravity','0 0 -9.81')
for act in root.find('actuator'):
 act.set('forcerange','-1.5 1.5');act.set('gainprm','28');act.set('biasprm','0 -28 -1')
world.find('geom[@name="floor"]').set('contype','3')
for g in world.findall('.//geom[@mesh]'):g.set('contype','0');g.set('conaffinity','6')
def model():return mj.MjModel.from_xml_string(E.tostring(root,encoding='unicode'))
a=model();d=mj.MjData(a);mj.mj_forward(a,d)
# Slice the two CAD fingers into convex collision pieces. Edges crossing
# each slice plane are retained so adjoining pieces meet exactly.
asset=root.find('asset')
for h in (1,2):
 for moving in (False,True):
  body=f'arm{h}_gripper' if moving else f'arm{h}_wrist_roll'
  source='moving_jaw_08d' if moving else 'wrist_roll_08c'
  node=world.find(f'.//body[@name="{body}"]')
  geom=node.find(f'geom[@mesh="{source}"]');geom.set('conaffinity','2')
  bid=a.body(body).id
  gid=next(g for g in range(a.ngeom) if a.geom_bodyid[g]==bid and a.geom_dataid[g]==a.mesh(source).id)
  mid=a.geom_dataid[gid]
  verts=a.mesh_vert[a.mesh_vertadr[mid]:a.mesh_vertadr[mid]+a.mesh_vertnum[mid]]
  local=(verts@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]-d.xpos[bid])@d.xmat[bid].reshape(3,3)
  faces=a.mesh_face[a.mesh_faceadr[mid]:a.mesh_faceadr[mid]+a.mesh_facenum[mid]]
  edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1),axis=0)
  axis=np.linalg.svd(local-local.mean(0),full_matrices=False)[2][0] if moving else np.array([0,0,1])
  coord=local@axis
  cuts=np.linspace(coord.min()-1e-6,coord.max()+1e-6,14)
  for part,(bot,top) in enumerate(zip(cuts,cuts[1:])):
   cloud=[local[(coord>=bot)&(coord<=top)]]
   for boundary in (bot,top):
    u,v=edges.T;cross=(coord[u]-boundary)*(coord[v]-boundary)<0;u=u[cross];v=v[cross]
    cloud.append(local[u]+((boundary-coord[u])/(coord[v]-coord[u]))[:,None]*(local[v]-local[u]))
   from scipy.spatial import ConvexHull
   points=np.unique(np.vstack(cloud).round(8),axis=0);points=points[ConvexHull(points).vertices]
   name=f'{body}_cad_{part}'
   E.SubElement(asset,'mesh',name=name,vertex=' '.join(f'{v:.8g}' for v in points.ravel()))
   E.SubElement(node,'geom',name=name,mesh=name,type='mesh',group='3',friction='1.1 .02 .001')
# Self-collision: one invisible, massless oriented box per robot link, fitted to
# the link's visual meshes. Boxes collide only with each other (bit 8); MuJoCo
# already skips parent-child pairs, structurally overlapping pairs are excluded below.
SELF=[]
for b in range(1,a.nbody):
 bname=a.body(b).name
 gs=[g for g in range(a.ngeom) if a.geom_bodyid[g]==b and a.geom_type[g]==7]
 if not gs:continue
 pts=[]
 for g in gs:
  mid=a.geom_dataid[g];v=a.mesh_vert[a.mesh_vertadr[mid]:a.mesh_vertadr[mid]+a.mesh_vertnum[mid]]
  pts.append((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-d.xpos[b])@d.xmat[b].reshape(3,3))
 pts=np.vstack(pts);c=pts.mean(0);U=np.linalg.svd(pts-c,full_matrices=False)[2]
 if np.linalg.det(U)<0:U[2]*=-1
 loc=(pts-c)@U.T;lo_,hi_=np.percentile(loc,1,0),np.percentile(loc,99,0)
 ctr=c+((lo_+hi_)/2)@U;half=(hi_-lo_)/2*.88
 q=R.from_matrix(U.T).as_quat()
 node=world.find(f'.//body[@name="{bname}"]')
 E.SubElement(node,'geom',name=f'{bname}_self',type='box',size=' '.join(f'{x:.5f}' for x in half),pos=' '.join(f'{x:.5f}' for x in ctr),
  quat=f'{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}',contype='8',conaffinity='8',group='3',mass='0',rgba='1 0 1 .3')
 SELF.append(bname)
SPACING=.22
for i,x in enumerate(np.arange(5)*SPACING):
 E.SubElement(world,'geom',name=f'rung{i}',contype='5',type='capsule',fromto=f'{x} -.16 .55 {x} .16 .55',size='.011',rgba=('0.16 0.47 .81 1' if i==0 else '.88 .32 .16 1'))
a=model();d=mj.MjData(a)
def palm(i):
 b=d.body(f'arm{i}_wrist_roll');M=b.xmat.reshape(3,3)
 return b.xpos+M@np.array([0,0,-.081]),M
lo,hi=a.jnt_range[1:].T
def setpose(v):
 d.qpos[:3]=v[:3];d.qpos[3:7]=np.roll(R.from_rotvec(v[3:6]).as_quat(),1);d.qpos[7:]=v[6:];mj.mj_forward(a,d)
