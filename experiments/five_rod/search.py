from train import *
from multiprocessing import get_context
import json,os
rng=np.random.default_rng(391)
steps=[(1,1),(2,2),(1,3),(2,4)]
def references():
 filepath=out/'gait.npy'
 if filepath.exists():return np.load(filepath)
 v=P[0].copy();g=[]
 for phase,(free,tar) in enumerate(steps):
  stance=3-free;old=tar-2 if phase else 1
  row=[]
  for fraction,depth in [(0,.05),(.5,.07),(1,.04),(1,0),(1,-1)]:
   dest=((1-fraction)*old+fraction*tar)*spacing
   def residual(u):
    setpose(u);res=[]
    for h in arms:
     x=dest if h==free else (tar-1)*spacing
     p,M=palm(h);res.extend((p-tipgoal(h,x,max(depth,0) if h==free else 0))*(5 if (h==free and fraction==.5) else 18))
     res.extend([M[1,0],M[1,2],max(0,M[2,2]+.60)])
    comx=tar*spacing if depth<0 else (tar-1)*spacing
    comy=tipgoal(free if depth<0 else stance,0,0)[1]
    res.extend([10*(d.subtree_com[1,0]-comx),10*(d.subtree_com[1,1]-comy)])
    res.extend(.014*(u[6:]-v[6:]))
    pen=np.zeros(24);n=0
    for c in d.contact[:d.ncon]:
     if a.geom_contype[c.geom1]==8 and a.geom_contype[c.geom2]==8 and n<24:pen[n]=40*max(0,.008-c.dist);n+=1
    res.extend(pen)
    return res
   sm=a.geom_margin.copy();a.geom_margin[a.geom_contype==8]=.008
   B=(np.r_[[-2]*3,[-12]*3,lo],np.r_[[2]*3,[12]*3,hi]);cands=[]
   for si in range(10):
    seed=v.copy() if si==0 else np.r_[v[:6],np.clip(v[6:]+np.random.default_rng(100*phase+si).normal(0,.8,8),lo,hi)]
    r=least_squares(residual,seed,bounds=B,max_nfev=300);cands.append((2*r.cost,r.x))
   u=min(cands,key=lambda c:c[0])[1];a.geom_margin[:]=sm;v=u;row.append(u.copy())
  g.append(row)
 np.save(filepath,g);return np.array(g)
G=references();G[3,:,6:]=G[1,:,6:]
def reset(noise=0):
 mj.mj_resetData(a,d);setpose(G[0,0]);d.qpos[7]+=noise;d.qpos[11]=-.85;d.qpos[14]=-.67;mj.mj_forward(a,d)
def restore(s):
 mj.mj_resetData(a,d);d.qpos[:]=s[0];d.qvel[:]=s[1];d.ctrl[:]=s[2];d.qacc_warmstart[:]=s[3];mj.mj_forward(a,d)
def snapshot():return tuple(x.copy() for x in (d.qpos,d.qvel,d.ctrl,d.qacc_warmstart))
def stage(phase,params,cb=None,tail=0):
 free,tar=steps[phase];stance=3-free;goal=tipgoal(free,tar*spacing,0)
 duration,gain,threshold,delay,*offset=params
 idx=[0,1,2,3,5,6];qstart=d.qpos[7:].copy()
 secure=0;ydev=0;shift=0.;released=False;closed=False;hold=0;near=1.;both=0;besthold=0;lift=0;air=0
 done_at=None
 for k in range(320+tail):
  t=(k+1)/duration
  seg=min(int(t),3);u=np.clip(t-seg,0,1);u=u*u*(3-2*u)
  q0=qstart if seg==0 else G[phase,seg-1,6:]
  ref=(1-u)*q0+u*G[phase,seg,6:]
  ref=(1-shift)*ref+shift*G[phase,4,6:]
  ref=ref.copy();ref[idx]+=np.array(offset[:6])*min(1,t/2)
  if len(offset)>6:
   ref[idx]+=np.array(offset[6:])*np.clip(1-abs(t-1.5)/1.5,0,1)
  p,_=palm(free);err=np.linalg.norm((goal-p)[[0,2]]);yerr=abs(goal[1]-p[1]);ydev=max(ydev,yerr)
  f,j,floor=contacts()
  if floor or air>=8:break
  if k>=320 and done_at is None:break
  air=air+1 if np.max(j)==0 else 0
  near=min(near,err)
  if t>=1.0 and yerr<.025 and (err<threshold or j[free-1,tar]==3):closed=True
  if closed and err>.032:closed=False
  if j[free-1,tar]==3 and closed:secure+=1;both=max(both,secure)
  else:secure=0
  if secure>=delay and t>=3:shift=min(1,shift+.04)
  if shift>=1:released=True
  holding=(released and j[stance-1,tar-1]==0 and j[free-1,tar]==3 and f[free-1,tar]>3 and yerr<.03)
  hold=hold+1 if holding else 0;besthold=max(hold,besthold)
  if cb:cb(phase,k,f,j,hold)
  if done_at is None and hold>=50:
   done_at=k
   if not tail:return 1000-k*.05,True
  if done_at is not None and k-done_at>=tail:return 1000-done_at*.05,True
  cmd=ref
  if gain>0 and seg>=2:
   corrected=command(stance,free,ref,goal,.7)
   cmd=ref+np.clip(corrected-ref,-gain,gain)
  cmd[4 if free==1 else 7]=(lo[4] if free==1 else lo[7]) if closed else (-.88 if free==1 else -.12)
  cmd[4 if stance==1 else 7]=(-.80 if stance==1 else -.06) if released else (lo[4] if stance==1 else lo[7])
  d.ctrl[:]=np.clip(cmd,lo,hi);mj.mj_step(a,d,nstep=10)
 return 12*np.exp(-near*55)+min(10,both)*.8+5*released+besthold*.9+max(0,1-err*20)*3-20*max(0,ydev-.03),False
phaseid=0;initial=None
def trial(p):
 restore(initial);return stage(phaseid,p)[0]
if __name__=='__main__':
 reset()
 bounds=np.array([[6,43],[0,.21],[.006,.022],[2,8]]+[[-1.2,1.2]]*6)
 for phaseid in range(4):
  initial=snapshot();pathfile=out/f'stage{phaseid}.npy'
  if pathfile.exists():
   par=np.load(pathfile);print('cached',phaseid,stage(phaseid,par),flush=True);continue
  mean=np.array([18,.01,.014,3,0,0,0,0,0,0]);std=np.array([7,.065,.004,1.5,.4,.4,.4,.4,.4,.4]);best=-1e9
  mean=np.r_[mean,np.zeros(6)]
  if phaseid==3:mean=np.r_[np.load(out/'stage1.npy'),np.zeros(16)][:16]
  if phaseid==2:mean=np.r_[np.load(out/'stage0.npy'),np.zeros(16)][:16]
  std=np.r_[std,np.ones(6)*.75];bounds=np.vstack([bounds[:10],np.tile([-1.9,1.9],(6,1))])
  with get_context('fork').Pool(8) as pool:
   for e in range(45):
    pop=np.clip(rng.normal(mean,std,(64,len(mean))),bounds[:,0],bounds[:,1]);pop[0]=mean
    vals=np.array(pool.map(trial,pop));order=np.argsort(vals);elite=pop[order[-9:]];i=order[-1]
    if vals[i]>best:best=vals[i];bestp=pop[i].copy()
    print(phaseid,e,round(best,2),flush=True)
    if best>900:break
    mean=elite.mean(0);std=np.maximum(elite.std(0)*1.2,[2,.022,.001,.6,.09,.09,.09,.09,.09,.09]+[.15]*6)
   if best<900:break
  restore(initial);print('replay',stage(phaseid,bestp),flush=True);np.save(pathfile,bestp)
