import json,numpy as np
from multiprocessing import get_context
from search import *
def run(spec):
 kind,val=spec
 reset(0)
 if kind=='shoulder':d.qpos[7]+=val
 else:d.qpos[7:]+=np.random.default_rng(val).uniform(-.03,.03,a.nq-7)
 mj.mj_forward(a,d);st={'pen':0.,'floor':0}
 def cb(ph,k,f,j,hold):
  for c in d.contact[:d.ncon]:
   if a.geom_contype[c.geom1]==8 and a.geom_contype[c.geom2]==8:st['pen']=max(st['pen'],-c.dist)
  st['floor']+=int(contacts()[2])
 c=0
 for ph in range(4):
  _,ok=stage(ph,np.load(out/f'stage{ph}.npy'),cb,tail=150 if ph==3 else 0)
  if not ok:break
  c+=1
 return kind,round(float(val),3),c,round(st['pen']*1000,1),st['floor'],round(d.time,2)
if __name__=='__main__':
 specs=[('shoulder',o) for o in np.linspace(-.05,.05,21)]+[('random',s) for s in range(20)]
 with get_context('fork').Pool(8) as p:res=p.map(run,specs)
 for r in res:print(*r)
 for k in ('shoulder','random'):
  v=[r for r in res if r[0]==k];print(k,'full success incl 3s final hang',sum(x[2]==4 for x in v),'/',len(v),'| max self-penetration mm',max(x[3] for x in v),'| floor frames',sum(x[4] for x in v))
 json.dump(res,open('audit.json','w'),indent=1)
