import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pinocchio as pin
import src.pinocchio_kinematic as pinocchio_kinematic
import src.utils as utils
import src.lowpass_filter as lowpass_filter

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        
    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]
        self.data.ctrl[:7] = self.initial_pos[:7]

        self.ee_body_name = "ee_center_body"
        self.arm = pinocchio_kinematic.Kinematics("ee_center_body")
        self.arm.buildFromMJCF(self.arm_xml)
        
        self.last_dof = self.data.qpos
        self.setTimestep(0.001)
        self.delta_d_ee_des = np.zeros(6)
        self.delta_ee_des = np.zeros(6)
        self.first_goto_initial_pos_cnt = 100

        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        self.acc_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        import src.matplot as mp
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        self.plot_manager.addNewFigurePlotter("vel.x", "vel.x", row=0, col=0)
        self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)
        self.ik_stop = False

    def runFunc(self):
        if self.first_goto_initial_pos_cnt > 0:
            self.first_goto_initial_pos_cnt -= 1
            self.data.ctrl[:7] = self.initial_pos[:7]
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.desired_pos = self.ee_pos
            self.last_ee_pos = self.ee_pos
            self.start_ee_pos = self.ee_pos
        else:
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            self.now_ee_vel = (self.now_ee_pos - self.last_ee_pos) / self.model.opt.timestep
            self.last_ee_pos = self.now_ee_pos
            self.now_ee_vel_filter =self.vel_filter.update(self.now_ee_vel)
            ee_pos_err = self.now_ee_pos - self.desired_pos

            F_ref = np.zeros(6)
            F_meas = np.array([0, 0, 0, 0, 0, 0])
            self.axis_index = 0
            F_meas[self.axis_index] = -1
            F_e = F_meas - F_ref

            self.M_d = np.diag([10] * 6)
            self.B_d = np.diag([0] * 6)
            self.K_d = np.diag([50] * 6)
            # M_d·ddq + B_d·dq + K_d·(q_des - q) = F_e
            dd_ee = np.linalg.inv(self.M_d) @ (F_e - self.B_d @ self.now_ee_vel - self.K_d @ ee_pos_err)

            self.delta_d_ee_des += dd_ee * self.model.opt.timestep
            self.delta_ee_des += self.delta_d_ee_des * self.model.opt.timestep
            self.desired_pos[:6] = self.start_ee_pos[:6] + self.delta_ee_des[:6]

            tf = utils.transform2mat(self.desired_pos[0], self.desired_pos[1], self.desired_pos[2], self.desired_pos[3], self.desired_pos[4], self.desired_pos[5])
            self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
            if self.desired_pos[self.axis_index] < 0.001 or not info["success"]:
                self.ik_stop = True
            if not self.ik_stop:
                self.last_dof = self.dof
                self.data.qpos[:7] = self.dof[:7]
                self.plot_manager.updateDataToPlotter("vel.x", "now_ee_vel.x", self.now_ee_vel[0])
                self.plot_manager.updateDataToPlotter("vel.x", "now_ee_velfilter.x", self.now_ee_vel_filter[0])
                self.plot_manager.updateDataToPlotter("delta.x", "delta.x", self.desired_pos[0])
                self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
                self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
            else:
                self.data.qpos[:7] = self.last_dof[:7]
            
if __name__ == "__main__":
    SCENE_XML = "model/franka_emika_panda/scene_pos.xml"
    ARM_XML = "model/franka_emika_panda/panda_pos.xml"
    env = PandaEnv(SCENE_XML, ARM_XML)
    env.run_loop()