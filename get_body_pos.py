import mujoco
import time
import mujoco_viewer as mujoco_viewer
import numpy as np

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
        print(f"End effector ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            # 如果未找到指定名称的末端执行器，打印警告信息并终止 GLFW
            print("Warning: Could not find the end effector with the given name.")
    
    def runFunc(self):
        end_effector_pos = self.data.body(self.end_effector_id).xpos, self.data.body(self.end_effector_id).xquat
        print(f"End effector position: {end_effector_pos}")
        # time.sleep(0.01)

test = Test("./model/franka_emika_panda/scene.xml")
test.run_loop()