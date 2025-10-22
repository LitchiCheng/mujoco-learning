import mujoco_viewer
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
    
    def runFunc(self):
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.initial_ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(self.initial_ee_quat)
        self.ee_quat_euler_rad = rot.as_euler('xyz')
        self.ee_quat_euler_angle = rot.as_euler('xyz', degrees=True)
        # 打印出body的位置，然后填入scene_withtarget.xml文件中,运行就可以校验ee_center_body是否是你想要控制的body
        print("Initial end-effector position and euler angles: ", self.initial_ee_pos, self.ee_quat_euler_angle) 
        self.ee_orient_norm = self.ee_quat_euler_angle / np.linalg.norm(self.ee_quat_euler_angle)
        print("ee normal orient: ", self.ee_orient_norm) 
        time.sleep(0.01)

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene_withtarget.xml")
    env.run_loop()
