"""Render or measure one uninterrupted rollout from the saved policy."""
import argparse
import ctypes
import csv
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from search import a,d,mj,reset,stage,out,palm,steps

def camera(x,y,z,dist,azimuth,elevation):
 c=mj.MjvCamera()
 xyz=np.array([x,y,z],dtype=np.float64)
 ctypes.memmove(c.lookat.ctypes.data,xyz.ctypes.data,xyz.nbytes)
 c.distance=dist;c.azimuth=azimuth;c.elevation=elevation
 return c

def font(size,bold=False):
 paths=['/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf','DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
 for path in paths:
  try:return ImageFont.truetype(path,size)
  except OSError:pass
 return ImageFont.load_default()

class Video:
 def __init__(self,path):
  a.vis.global_.offwidth=960;a.vis.global_.offheight=720
  self.r=mj.Renderer(a,height=650,width=900);self.colors=['#6dcabd','#67b7f1','#f4b858','#b894ef','#f78f79']
  for i,h in enumerate(self.colors):
   a.geom_rgba[a.geom('rung'+str(i)).id,:3]=[int(h[n:n+2],16)/255 for n in [1,3,5]]
  a.geom_rgba[a.geom_group==3,3]=0
  a.site_rgba[:,3]=0
  self.fonts={s:font(s) for s in [16,18,20,24,28]}
  self.pipe=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
  self.count=0;self.window=[]
 def draw(self,phase,k,f,j,hold):
  self.count+=1
  if self.count%2==0:return
  image=Image.new('RGB',(1280,720),'#132028');dr=ImageDraw.Draw(image)
  self.r.update_scene(d,camera=camera(.20,0,.404,.88,103,-12))
  overview=Image.fromarray(self.r.render());image.paste(overview,(0,65))
  free,tar=steps[phase];p=palm(free)[0]
  focus=p*.68+np.array([tar*.1,p[1],.55])*.32
  self.r.update_scene(d,camera=camera(*focus,.235,112,-16))
  closeup=Image.fromarray(self.r.render()).resize((380,275),Image.Resampling.LANCZOS);image.paste(closeup,(900,110))
  def txt(x,y,t,size=18,color='#f1f4ed'):dr.text((x,y),t,font=self.fonts[size],fill=color)
  txt(22,19,'DUAL2   /   Five-rod traversal',28);txt(950,26,f'Simulation time     {d.time:05.2f} s',20)
  txt(920,77,'Reaching gripper',20)
  txt(920,403,'Rod support force',24)
  for i,col in enumerate(self.colors):
   y=453+i*39;value=f[:,i].sum()
   dr.ellipse((921,y+3,938,y+20),fill=col);txt(949,y,f'{i+1}',18);dr.rounded_rectangle((983,y+6,1163,y+17),radius=4,fill='#2d3c44')
   n=int(180*np.clip(value/13,0,1))
   if n>1:dr.rounded_rectangle((983,y+6,983+n,y+17),radius=4,fill=col)
   txt(1178,y-2,f'{value:5.1f} N',18)
  txt(920,661,'Robot weight   8.77 N',18,'#b2c4c3')
  self.pipe.stdin.write(image.tobytes())
 def close(self):
  self.pipe.stdin.close()
  if self.pipe.wait()!=0:raise RuntimeError('FFmpeg failed')
  self.r.close()
def evaluate(noise,video=None,writer=None):
 reset(noise);stats={'cleared':0,'no_contact_frames':0,'nonjaw_rod_frames':0,'floor_frames':0,'duration':0}
 times=[]
 def sample(ph,k,f,j,hold):
  stats['no_contact_frames']+=int(j.max()==0)
  pairs=[(a.geom(c.geom1).name,a.geom(c.geom2).name) for c in d.contact]
  stats['nonjaw_rod_frames']+=int(any(('rung' in u+v and '_cad_' not in u+v) for u,v in pairs))
  stats['floor_frames']+=int(any('floor' in pair for pair in pairs))
  if writer:writer.writerow([round(d.time,4),ph+1,hold,*np.round(f.sum(0),4),*j.flatten(),*np.round(d.subtree_com[1],5)])
  if video:video.draw(ph,k,f,j,hold)
 for ph in range(4):
  _,ok=stage(ph,np.load(out/f'stage{ph}.npy'),sample)
  if not ok:break
  stats['cleared']+=1;times.append(round(d.time,2))
 stats['duration']=round(d.time,2);stats['completed_at']=times
 return stats
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=Path);p.add_argument('--csv',type=Path);p.add_argument('--noise',type=float,default=0);args=p.parse_args()
 video=Video(args.video) if args.video else None
 log=args.csv.open('w',newline='') if args.csv else None;writer=csv.writer(log) if log else None
 if writer:writer.writerow(['time_s','transfer','hold_samples',*[f'rod{i+1}_up_N' for i in range(5)],*[f'hand{h}_rod{r}_jaw_mask' for h in [1,2] for r in range(1,6)],'com_x','com_y','com_z'])
 try:print(evaluate(args.noise,video,writer))
 finally:
  if video:video.close()
  if log:log.close()
