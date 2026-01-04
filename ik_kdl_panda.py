import mujoco
import numpy as np
import mujoco_viewer as mujoco_viewer
import src.kdl_kinematic as kdl_kinematic
import time
import os
import utils

class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path):
        super().__init__(scene_path, distance=3.5, azimuth=180, elevation=-90)
        self.scene_path = scene_path
        self.arm_path = arm_path
        
        self.ee_body_name = "ee_center_body"
        # 初始化逆运动学
        self.arm = kdl_kinematic.Kinematics(self.ee_body_name)
        urdf_file = "model/franka_panda_urdf/robots/panda_arm.urdf"
        self.arm.buildFromURDF(urdf_file, "link0")
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        self.last_dof = None
        
    def runBefore(self):
        self.model.opt.timestep = 0.001
        # 设定初始位置
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        # mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        # self.x = ee_pos[0]
        # self.y = ee_pos[1]
        # self.z = ee_pos[2]
        self.x = 0.1
        self.y = 0.1
        self.z = 0.5
        self.last_dof = self.data.qpos[:9].copy()
    
    def runFunc(self):
        self.x += 0.001
        if self.x > 0.5:
            self.x = 0.5
            self.data.qpos[:7] = self.last_dof[:7]
            self.data.qpos[7:] = [0,0]
        else:
            tf = utils.transform2mat(self.x, self.y, self.z, np.pi, 0, 0)
            self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
            if info["success"]:
                self.last_dof = self.dof
                self.data.qpos[:7] = self.dof[:7]
                self.data.qpos[7:] = [0,0]
                # self.data.ctrl[:7] = self.dof[:7]
                print("ee pos", self.getBodyPosByName(self.ee_body_name))
                print("fk", self.arm.fk(self.dof[:7]))
        time.sleep(0.01)

if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_pos.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda_pos.xml'
    robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()
