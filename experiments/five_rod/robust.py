"""Robust per-stage CEM: each candidate is scored on several perturbed start states."""
import sys,shutil
from search import *
from multiprocessing import get_context
OFFS=[('s',o) for o in np.linspace(-.05,.05,11)]+[('r',i) for i in range(6)]
def run_chain(spec,upto):
 kind,val=spec;reset(0)
 if kind=='s':d.qpos[7]+=val
 else:d.qpos[7:]+=np.random.default_rng(val).uniform(-.03,.03,a.nq-7)
 mj.mj_forward(a,d)
 for ph in range(upto):
  _,ok=stage(ph,np.load(out/f'stage{ph}.npy'))
  if not ok:return None
 return snapshot()
starts=[];TAIL=0
def trial(p):
 sc=[];ok=0
 for s in starts:
  restore(s);v,good=stage(PH,p,tail=TAIL);sc.append(min(v,1000) if good else v);ok+=good
 sc=np.array(sc)
 return ok*100+sc.mean()*.05+(sc.min() if ok<len(starts) else 50)*.02
if __name__=='__main__':
 PH=int(sys.argv[1]);gens=int(sys.argv[2]) if len(sys.argv)>2 else 30;TAIL=int(sys.argv[3]) if len(sys.argv)>3 else 0
 with get_context('fork').Pool(8) as pool:
  ss=pool.starmap(run_chain,[(o,PH) for o in OFFS])
 starts=[s for s in ss if s is not None];print('starts',len(starts),flush=True)
 rng=np.random.default_rng(7+PH)
 mean=np.load(out/f'stage{PH}.npy')
 if len(mean)<16:mean=np.r_[mean,np.zeros(16-len(mean))]
 lo_b=np.r_[[6,0,.006,2],[-1.2]*6,[-1.9]*6];hi_b=np.r_[[43,.21,.024,8],[1.2]*6,[1.9]*6]
 std=np.r_[[3,.03,.002,.8],[.15]*6,[.3]*6]
 with get_context('fork').Pool(8) as pool:
  base=trial(mean);best=base;bestp=mean.copy();print('base',round(base,2),flush=True)
  for e in range(gens):
   pop=np.clip(rng.normal(mean,std,(48,16)),lo_b,hi_b);pop[0]=bestp
   vals=np.array(pool.map(trial,pop));order=np.argsort(vals);elite=pop[order[-8:]]
   if vals[order[-1]]>best:best=vals[order[-1]];bestp=pop[order[-1]].copy()
   print(PH,e,round(best,2),'succ',int(best//100),'/',len(starts),flush=True)
   if best//100>=len(starts) and e>=4:break
   mean=.5*elite.mean(0)+.5*bestp;std=np.maximum(elite.std(0)*1.1,np.r_[[.8,.008,.0005,.2],[.04]*6,[.08]*6])
 if best>base:
  shutil.copy(out/f'stage{PH}.npy',out/f'stage{PH}.prev.npy');np.save(out/f'stage{PH}.npy',bestp);print('saved',flush=True)
