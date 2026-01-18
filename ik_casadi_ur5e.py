import mujoco
import numpy as np
import src.mujoco_viewer as mujoco_viewer as mujoco_viewer
import src.pinocchio_kinematic as pinocchio_kinematic
import time
import os
import src.utils as utils

class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path):
        super().__init__(scene_path, distance=3.5, azimuth=180, elevation=-90)
        self.scene_path = scene_path
        self.arm_path = arm_path
        
        # 初始化逆运动学
        self.arm = pinocchio_kinematic.Kinematics("wrist_3_link")
        self.arm.buildFromMJCF(arm_path)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self.last_dof = None
        
    def runBefore(self):
        self.model.opt.timestep = 0.005
        # 设定初始位置
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.x = ee_pos[0]
        self.y = ee_pos[1]
        self.z = ee_pos[2]
        # self.x = 0.1
        # self.y = 0.0
        # self.z = 0.3
        print("ee pos", self.x, self.y, self.z)
        self.last_dof = self.data.qpos[:6].copy()
    
    def runFunc(self):
        self.x += 0.001
        if self.x > -0.62:
            self.x = -0.62
        self.z += 0.001
        if self.z > 0.3:
            self.z = 0.3
        tf = utils.transform2mat(self.x, self.y, self.z, np.pi, 0, 0)
        self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
        # print("ik result", info["success"])
        self.last_dof = self.dof
        self.data.qpos[:6] = self.dof[:6]
        # print("ee pos", self.getBodyPosByName("wrist_3_link"))


if __name__ == "__main__":
    SCENE_XML_PATH = 'model/mjcf/universal_robots_ur5e/scene.xml'
    ARM_XML_PATH = 'model/mjcf/universal_robots_ur5e/ur5e.xml'
    robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()
