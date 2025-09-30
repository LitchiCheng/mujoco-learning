import mujoco_viewer
import time

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        # 第一个关键点的位置通常是home位置
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]  # 设定初始位置
    
    def runFunc(self):
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        time.sleep(0.01)

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene.xml")
    env.run_loop()
