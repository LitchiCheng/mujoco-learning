import mujoco_viewer
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
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.8]
        }
        self.obstacle_size = 0.1
        
    def get_random_position(self):
        """在工作空间内生成随机位置"""
        x = random.uniform(self.workspace['x'][0], self.workspace['x'][1])
        y = random.uniform(self.workspace['y'][0], self.workspace['y'][1])
        z = random.uniform(self.workspace['z'][0], self.workspace['z'][1])
        return np.array([x, y, z])
    
    def runBefore(self):
        self.handle.user_scn.ngeom = self.num_obstacles
        for i in range(self.num_obstacles):
            pos = self.get_random_position()
            rgba = np.random.rand(4)
            rgba[3] = 0.8  # 固定透明度
            if np.sum(rgba[:3]) < 0.5:  # 确保颜色不太暗
                rgba[:3] += 0.5
                rgba[:3] = np.clip(rgba[:3], 0, 1)
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.obstacle_size, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
    
    def runFunc(self):
        time.sleep(0.01)

if __name__ == "__main__":
    env = Env("./model/franka_emika_panda/scene.xml", num_obstacles=5)
    env.run_loop()
