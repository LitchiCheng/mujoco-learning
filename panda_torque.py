import src.mujoco_viewer as mujoco_viewer as mujoco_viewer
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time,math
import src.pid_controller as pid
import src.kdl_kinematic as kdl_kinematic
import mujoco
import src.utils as utils


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

        kp = 50.0
        ki = 1.0
        kd = 5.0
        self.pids = []
        for i in range(self.model.nq):
            self.pids.append(pid.PIDController(kp, ki, kd))
        self.target_qpos = []
        self.model.opt.timestep = 0.001
        self.dt = self.model.opt.timestep
    
    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        self.x = 0.4
        self.y = 0.0
        self.z = 0.5
        tf = utils.transform2mat(self.x, self.y, self.z, np.pi, 0, 0)
        self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
        self.target_qpos = self.dof[:7]
       
    def runFunc(self):
        current_qpos = self.data.qpos[:7]
        error = self.target_qpos - current_qpos
        target_torque = np.zeros(7)
        for i in range(7):
            target_torque[i] = self.pids[i].update(error[i], self.dt)
        min_torque = self.model.actuator_ctrlrange[:7, 0]
        max_torque = self.model.actuator_ctrlrange[:7, 1]
        clamped_torque = np.clip(target_torque, min_torque, max_torque)
        self.data.ctrl[:7] = clamped_torque[:7]

        print("error: ", error)
        print("target_torque: ", target_torque)
        

if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_tau.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda_tau.xml'
    robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()

    