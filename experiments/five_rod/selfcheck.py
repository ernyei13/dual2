import numpy as np
from search import *
def selfpen():
 m=0;pairs=set()
 for c in d.contact[:d.ncon]:
  if a.geom_contype[c.geom1]==8 and a.geom_contype[c.geom2]==8:m=max(m,-c.dist);pairs.add((a.body(a.geom_bodyid[c.geom1]).name,a.body(a.geom_bodyid[c.geom2]).name))
 return m,pairs
if __name__=='__main__':
 for ph in range(4):
  for k in range(4):
   setpose(G[ph,k]);m,p=selfpen()
   print(ph,k,'pen mm',round(m*1000,1),p,'palms',[np.round(palm(h)[0],3).tolist() for h in (1,2)])
