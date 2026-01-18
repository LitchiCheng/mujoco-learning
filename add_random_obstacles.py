import src.mujoco_viewer as mujoco_viewer as mujoco_viewer
import time
import mujoco
import numpy as np
import random

class Env(mujoco_viewer.CustomViewer):
    def __init__(self, path, num_obstacles=5):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
        self.num_obstacles = num_obstacles
        self.workspace = {
            'x': [-0.3, 0.3],
            'y': [-0.3, 0.3],
            'z': [0.0, 0.7]
        }
        self.usr_obstacle_size = []
        self.usr_obstacle_pos = []
        self.usr_obstacle_rgba = []
        self.usr_obstacle_type = []
        for i in range(self.num_obstacles):
            pos = self.get_random_position()
            rgba = np.random.rand(4)
            rgba[3] = 0.8
            self.usr_obstacle_pos.append([pos[0], pos[1], pos[2]])
            # self.usr_obstacle_type.append("box")
            self.usr_obstacle_type.append("sphere")
            self.obstacle_size = 0.05
            # self.usr_obstacle_size.append([self.obstacle_size, self.obstacle_size, self.obstacle_size])
            self.usr_obstacle_size.append([self.obstacle_size])
            self.usr_obstacle_rgba.append([rgba[0], rgba[1], rgba[2], 0.8])
        self.addObstacles(self.usr_obstacle_pos, self.usr_obstacle_type, self.usr_obstacle_size, self.usr_obstacle_rgba)
    
    
    def get_random_position(self):
        """在工作空间内生成随机位置"""
        x = random.uniform(self.workspace['x'][0], self.workspace['x'][1])
        y = random.uniform(self.workspace['y'][0], self.workspace['y'][1])
        z = random.uniform(self.workspace['z'][0], self.workspace['z'][1])
        return np.array([x, y, z])
    
    def runBefore(self):
        pass

    def runFunc(self):
        time.sleep(0.01)

if __name__ == "__main__":
    env = Env("./model/franka_emika_panda/scene.xml", num_obstacles=10)
    env.run_loop()
