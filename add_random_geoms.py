import mujoco_viewer
import time
import mujoco
import numpy as np
import random
import src.key_listener as key_listener
from pynput import keyboard

key_states = {
    keyboard.Key.down: False,
}

class Env(mujoco_viewer.CustomViewer):
    def __init__(self, path, num_geoms=5):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
        self.num_geoms = num_geoms
        self.workspace = {
            'x': [-0.3, 0.3],
            'y': [-0.3, 0.3],
            'z': [0.0, 0.7]
        }
        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
    
    def get_random_position(self):
        """在工作空间内生成随机位置"""
        x = random.uniform(self.workspace['x'][0], self.workspace['x'][1])
        y = random.uniform(self.workspace['y'][0], self.workspace['y'][1])
        z = random.uniform(self.workspace['z'][0], self.workspace['z'][1])
        return np.array([x, y, z])
    
    def runBefore(self):
        self.usr_geom_size = []
        self.usr_geom_pos = []
        self.usr_geom_rgba = []
        self.usr_geom_type = []
        for i in range(self.num_geoms):
            pos = self.get_random_position()
            rgba = np.random.rand(4)
            rgba[3] = 0.8
            self.usr_geom_pos.append([pos[0], pos[1], pos[2]])
            self.usr_geom_type.append("sphere")
            self.geom_size = 0.05
            self.usr_geom_size.append([self.geom_size])
            self.usr_geom_rgba.append([rgba[0], rgba[1], rgba[2], 0.8])
        self.addVisuGeom(self.usr_geom_pos, self.usr_geom_type, self.usr_geom_size, self.usr_geom_rgba)

    def runFunc(self):
        if key_states[keyboard.Key.down]:
            tmp_size = []
            tmp_pos = []
            tmp_rgba = []
            tmp_type = []
            pos = self.get_random_position()
            rgba = np.random.rand(4)
            rgba[3] = 0.8
            tmp_pos.append([pos[0], pos[1], pos[2]])
            tmp_type.append("sphere")
            # tmp_type.append("box")
            self.geom_size = 0.01
            tmp_size.append([self.geom_size])
            # tmp_size.append([self.geom_size, self.geom_size, self.geom_size])
            tmp_rgba.append([rgba[0], rgba[1], rgba[2], 0.8])
            self.addVisuGeom(tmp_pos, tmp_type, tmp_size, tmp_rgba)
        time.sleep(0.01)

if __name__ == "__main__":
    env = Env("./model/franka_emika_panda/scene.xml", num_geoms=10)
    env.run_loop()
