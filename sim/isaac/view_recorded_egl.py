#!/usr/bin/env python3
"""Show Isaac recorded joint and cube poses using EGL and a Tk window."""
import json
import time
import tkinter as tk
import h5py
import mujoco
import numpy as np
from PIL import Image, ImageTk
from view_recorded_cube_grasp import DEFAULT_DATASET, load_model, index

with h5py.File(DEFAULT_DATASET, 'r') as f:
    episode = f['data/demo_0']
    joints = episode['states/articulation/robot/joint_position'][:]
    cubes = np.stack([episode[f'states/rigid_object/cube_{i}/root_pose'][:] for i in (1, 2, 3)], axis=1)
    cfg = json.loads(f['data'].attrs['env_args'])['sim_args']
    dt = cfg['dt'] * cfg['decimation']
model = load_model()
model.vis.global_.offwidth = 800
model.vis.global_.offheight = 600
data = mujoco.MjData(model)
arm = [model.jnt_qposadr[index(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in [*(f'joint{i}' for i in range(1,7)), 'gripper_joint1', 'gripper_joint2']]
boxes = [model.jnt_qposadr[index(model, mujoco.mjtObj.mjOBJ_JOINT, f'replay_cube_joint_{i}')] for i in (1,2,3)]
renderer = mujoco.Renderer(model, height=600, width=800)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.38, 0, 0.16]
cam.distance, cam.azimuth, cam.elevation = 1.05, 132, -20
root = tk.Tk()
root.title('Piper - Isaac Lab recording replay')
label = tk.Label(root)
label.pack()
tk.Label(root, text='Isaac Lab recording / MuJoCo replay    |    Space: pause    R: restart    Drag: rotate    Scroll: zoom').pack()
state = {'frame': 0, 'paused': False, 'last': None}
def keys(e):
    if e.keysym == 'space': state['paused'] = not state['paused']
    elif e.keysym.lower() == 'r': state['frame'] = 0
root.bind('<Key>', keys)
def press(e): state['last'] = (e.x,e.y)
def drag(e):
    if state['last']:
        x,y = state['last']; cam.azimuth -= (e.x-x)*0.4; cam.elevation = max(-89,min(89,cam.elevation+(e.y-y)*0.4)); state['last']=(e.x,e.y)
def zoom(e): cam.distance = max(.3,min(3.,cam.distance*(.9 if e.delta > 0 or e.num == 4 else 1.1)))
label.bind('<Button-1>', press); label.bind('<B1-Motion>', drag)
label.bind('<MouseWheel>', zoom); label.bind('<Button-4>', zoom); label.bind('<Button-5>', zoom)
def update():
    start=time.monotonic(); frame=state['frame']
    data.qpos[arm]=joints[frame]
    for i, address in enumerate(boxes): data.qpos[address:address+7]=cubes[frame,i]
    mujoco.mj_forward(model,data); renderer.update_scene(data,camera=cam)
    img=ImageTk.PhotoImage(Image.fromarray(renderer.render()))
    label.configure(image=img); label.image=img
    if not state['paused']: state['frame']=(frame+1)%len(joints)
    if frame==0: print('FIRST_FRAME_RENDERED',flush=True)
    root.after(max(1, int((dt-(time.monotonic()-start))*1000)),update)
def close():
    renderer.close(); root.destroy()
root.protocol('WM_DELETE_WINDOW',close)
root.after(0,update)
root.mainloop()
