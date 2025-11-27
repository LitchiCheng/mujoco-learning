import numpy as np
import pinocchio as pin
import time
import mujoco_viewer
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

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        theta = np.pi
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        se3 = pin.SE3(rotation_matrix, np.zeros(3))
        quaternion = pin.Quaternion(se3.rotation)
        self.quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    
    def runFunc(self):
        if key_states[keyboard.Key.up]:
            self.data.mocap_pos[0, 2] += 0.001
        if key_states[keyboard.Key.down]:
            self.data.mocap_pos[0, 2] -= 0.001
        if key_states[keyboard.Key.left]:
            self.data.mocap_pos[0, 0] -= 0.001
        if key_states[keyboard.Key.right]:
            self.data.mocap_pos[0, 0] += 0.001
        if key_states[keyboard.Key.alt_l]:
            self.data.mocap_pos[0, 1] += 0.001
        if key_states[keyboard.Key.alt_r]:
            self.data.mocap_pos[0, 1] -= 0.001

        self.data.mocap_quat[0] = self.quat
        time.sleep(0.01)

test = Test("./model/franka_emika_panda/scene_withmocap.xml")
test.run_loop()
