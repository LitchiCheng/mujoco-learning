import mujoco_viewer
import time

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        pass
    
    def runFunc(self):
        time.sleep(0.01)

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene.xml")
    env.run_loop()
