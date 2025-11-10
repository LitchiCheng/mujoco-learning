import mujoco
import numpy as np
import mujoco_viewer
import casadi_ik
import time
import os

class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path):
        super().__init__(scene_path, distance=1.5, azimuth=135, elevation=-30)
        self.scene_path = scene_path
        self.arm_path = arm_path
        
        # 初始化逆运动学
        self.arm = casadi_ik.Kinematics("joint7")
        self.arm.buildFromMJCF(arm_path)
        
        self.last_dof = None
        
    def runBefore(self):
        self.x = 0.3
        pass
    
    def runFunc(self):
        self.x += 0.001
        tf = self.build_transform_simple(self.x, -0., 0.4, np.pi, 0, 0)
        
        # 求解逆运动学
        self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
        self.last_dof = self.dof
        
        self.data.qpos[:7] = self.dof[:7]

        mujoco.mj_step(self.model, self.data)
        time.sleep(0.01)
    
    def build_transform_simple(self, x, y, z, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
            [-sp,   cp*sr,            cp*cr,            z],
            [0,     0,                0,                1]
        ])

if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_withtarget.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda.xml'
    robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()
