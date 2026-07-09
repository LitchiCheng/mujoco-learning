import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

key_states = {
    keyboard.Key.up: False,
    keyboard.Key.down: False,
    keyboard.Key.left: False,
    keyboard.Key.right: False,
    keyboard.Key.alt_l: False,
    keyboard.Key.alt_r: False,
}

def on_press(key):
    if key in key_states:
        key_states[key] = True

def on_release(key):
    if key in key_states:
        key_states[key] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

XML = """
<mujoco>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-30"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    
    <body name="ball" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom type="sphere" size="0.2" rgba="1 0 0 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
model.opt.gravity = (0, 0, 0)
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.pos = 0.0001

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport
    
    def run_loop(self):
        while self.is_running():
            ball_body_name = "ball"
            pos = data.body(ball_body_name).xpos
            quat = data.body(ball_body_name).xquat
            print(f"Position: {pos}, Orientation: {quat}")

            if key_states[keyboard.Key.up]:
                data.qpos[0] += self.pos
            if key_states[keyboard.Key.down]:
                data.qpos[0] -= self.pos
            if key_states[keyboard.Key.left]:
                data.qpos[1] += self.pos
            if key_states[keyboard.Key.right]:
                data.qpos[1] -= self.pos
            if key_states[keyboard.Key.alt_l]:
                data.qpos[2] += self.pos
            if key_states[keyboard.Key.alt_r]:
                data.qpos[2] -= self.pos

            mujoco.mj_step(model, data)
            self.sync()


viewer = CustomViewer(model, data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30
viewer.run_loop()