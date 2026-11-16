from model import *
from scipy.optimize import least_squares
from functools import lru_cache
import argparse
spacing=SPACING
arms=[2,1]
def tipgoal(hand,x,drop):
 return np.array([x, .045 if hand==1 else -.045, .55-drop])
def solve(v, positions, drops, reference):
 def residual(u):
  setpose(u)
  terms=[]
  for hand,x,drop in zip(arms,positions,drops):
   p,M=palm(hand)
   terms.extend(20*(p-tipgoal(hand,x,drop)))
   terms.extend([2*M[1,0],2*M[1,2],1.5*max(0,M[2,2]+.55)])
  terms.extend(.012*(u[6:]-reference[6:]))
  terms.append(12*(d.subtree_com[1,0]-positions[0 if positions[0]!=positions[1] and drops[0]==0 else 1]))
  return terms
 return least_squares(residual,v,bounds=(np.r_[[-2]*3,[-12]*3,lo],np.r_[[2]*3,[12]*3,hi]),max_nfev=80).x
def path():
 pathfile=out/'waypoints.npy'
 if pathfile.exists():return np.load(pathfile)
 seed=np.load(out/'pose.npy'); v=solve(seed,[0,spacing],[0,0],seed)
 poses=[]
 # start straddling rods one and two, free long hand starts below target
 for positions,drops in [([0,spacing],[0,.055]),([0,spacing],[0,0]),([0,spacing],[.044,0]),([spacing,spacing],[.044,0]),([spacing,spacing],[0,0]),([spacing,spacing],[0,.044]),([spacing,2*spacing],[0,.044]),([spacing,2*spacing],[0,0])]:
  v=solve(v,positions,drops,v);setpose(v);print('pose',[(palm(h)[0]-tipgoal(h,x,dr)).round(3) for h,x,dr in zip(arms,positions,drops)],flush=True);poses.append(v.copy())
 np.save(pathfile,poses);return np.array(poses)
P=path()
# target trackers in the constrained frame of the holding hand
def command(stance,free,ref,goal,gain):
 ps,Ms=palm(stance);pf,Mf=palm(free)
 def jac(p,h):
  jp=np.zeros((3,a.nv));jr=jp.copy();mj.mj_jac(a,d,jp,jr,p,a.body(f'arm{h}_wrist_roll').id)
  return jp[:,6:],jr[:,6:]
 Jp,Jr=jac(pf,free);Js,Rs=jac(ps,stance)
 # Account for settling about the rod under gravity.
 com=d.subtree_com[1];jc=np.zeros((3,a.nv));mj.mj_jacSubtreeCom(a,d,jc,1)
 relative=jc[:,6:]-Js-np.cross(Rs.T,com-ps).T
 pivot=-relative[0]/min(-.06,com[2]-ps[2])
 J=Jp-Js-np.cross(Rs.T,pf-ps).T+np.outer(np.cross([0,1,0],pf-ps),pivot)
 orient=np.cross(Mf[:,1],[0,np.sign(Mf[1,1]),0])
 A=np.vstack((J,.05*(Jr-Rs)[[0,2]]))
 error=np.r_[goal-pf,.05*orient[[0,2]]]
 pinv=A.T@np.linalg.inv(A@A.T+.00005*np.eye(5))
 dq=pinv@error+(np.eye(8)-pinv@A)@(ref-d.qpos[7:])
 return np.clip(d.qpos[7:]+gain*np.clip(dq,-.55,.55),lo,hi)
def contacts():
 forces=np.zeros((2,5));jaws=np.zeros((2,5),int);floor=False
 for ci,c in enumerate(d.contact):
  g1,g2=a.geom(c.geom1).name,a.geom(c.geom2).name
  if 'floor' in (g1,g2):floor=True
  if g2.startswith('rung'):g1,g2=g2,g1;sgn=-1
  else:sgn=1
  if not g1.startswith('rung') or not g2.startswith('arm'):continue
  h=int(g2[3])-1;r=int(g1[-1]);jaws[h,r]|=1 if 'gripper' in g2 else 2
  f=np.zeros(6);mj.mj_contactForce(a,d,ci,f)
  forces[h,r]+=sgn*np.dot(d.contact.frame[ci].reshape(3,3)[:,2],f[:3])
 return forces,jaws,floor
def rollout(param,noise=0,cb=None,verbose=False):
 mj.mj_resetData(a,d);setpose(P[0]);d.qpos[7]+=noise;d.qpos[11]=.1;d.qpos[14]=-.055;mj.mj_forward(a,d)
 steps=[(1,1,1)]
 for rung in range(1,4):steps.extend([(2,rung,4),(1,rung+1,7)])
 phase=0;sub=0;tick=0;stable=0;timer=0;minerr=1;released=False;air=0;floor=False
 for k in range(1800):
  free,tar,idx=steps[phase];stance=3-free;prev=tar-1 if free==1 else tar
  goal=tipgoal(free,tar*spacing,0)
  p,M=palm(free);err=np.linalg.norm((p-goal)[[0,2]])
  pos=goal.copy()
  if sub==0 and phase:
   wp=palm(free)[0];pos[0]=(tar-1)*spacing;pos[2]-=.053
   if tick>param[1] or wp[2]<.509:sub=1;tick=0
  if sub==1:pos[2]-=.048
  if sub==1 and (abs(p[0]-goal[0])<.015 or tick>55):sub=2;tick=0
  if not phase:sub=2
  if sub>=2:minerr=min(minerr,err)
  f,j,floor=contacts()
  if floor:break
  if sub==2 and (err<param[2] or j[free-1,tar]==3):sub=3;tick=0
  if sub==3 and tick>20 and err>.032:sub=2;tick=0
  stable=stable+1 if sub==3 and j[free-1,tar]==3 else 0
  if stable>=param[3]:sub=4;tick=0;released=True
  if sub==4 and f[free-1,tar]>3 and j[stance-1,prev]==0:
   timer+=1
  else:timer=0
  if timer>=(60 if phase==len(steps)-1 else 9):
   if verbose:print('transfer',k,phase,free,tar)
   phase+=1
   if phase==len(steps):return phase,0,k,False
   sub=0;stable=0;timer=0;tick=0;minerr=1;released=False;continue
  ref=P[idx if sub>=2 else max(0,idx-1),6:]
  cmd=command(stance,free,ref,pos,param[0])
  cmd=ref+np.clip(cmd-ref,-param[4],param[4]) if len(param)>4 else cmd
  cmd[4 if stance==1 else 7]=(.10 if stance==1 else .68) if released else -.48
  cmd[4 if free==1 else 7]=-.5 if sub>=3 else (.1 if free==1 else .68)
  if cb:cb(k,phase,sub,f.copy(),j.copy())
  d.ctrl[:]=cmd;mj.mj_step(a,d,nstep=10);tick+=1
  if tick>160 and sub!=4:break
  if sub==4 and tick>120:break
 return phase,minerr,k,floor
if __name__=='__main__':
 print(rollout([.7,25,.014,4],verbose=True))
