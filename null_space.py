import pinocchio 
import mujoco_viewer as mujoco_viewer
import time
import numpy as np
import mujoco
import utils
import src.pinocchio_kinematic as pinocchio_kinematic

class PandaPbvs(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, pin_xml):
        super().__init__(scene_xml, 3, azimuth=-90, elevation=-30)
        self.scene_xml = scene_xml
        self.pin_xml = pin_xml

    def runBefore(self):
        self.model.opt.timestep = 0.01
        self.kine = pinocchio_kinematic.Kinematics("ee_center_body")
        self.kine.buildFromMJCF(self.pin_xml)
        tf = utils.transform2mat(0.3, 0.1, 0.5, np.pi, 0, 0)
        dof, info = self.kine.ik(tf)
        self.data.qpos[:7] = dof[:7]
        self.prev_joint_vel = np.zeros(9, dtype=np.float32)  
        self.keep_pos = False
        self.integral_qpos = self.data.qpos.copy()

        self.joint_mid = np.zeros(self.model.nq)
        for i in range(self.model.nq):
            self.joint_mid[i] = (self.model.jnt_range[i][0] + self.model.jnt_range[i][1]) / 2

    def nullSpaceProjection(self, J, vel_candidate):
        # 零空间投影矩阵：P_null = I - J^+J
        I = np.eye(J.shape[1])
        J_pinv = np.linalg.pinv(J, rcond=1e-6)
        P_null = I - np.dot(J_pinv, J)
        return np.dot(P_null, vel_candidate)
    
    def geVelCandidate(self, q, q_mid, gain=0.1):
        vel_candidate = np.zeros(self.model.nq)
        vel_candidate[2] = -0.5
        return vel_candidate

    def runFunc(self):
        np.set_printoptions(precision=3)
        J = self.kine.getJac(self.data.qpos)
        vel_candidate = self.geVelCandidate(self.data.qpos, self.joint_mid)
        vel_null = self.nullSpaceProjection(J, vel_candidate)
        self.integral_qpos[:7] += vel_null[:7] * self.model.opt.timestep  # 更新关节位置
        for i in range(self.model.nq):
            self.integral_qpos[i] = np.clip(
                self.integral_qpos[i],
                self.model.jnt_range[i][0],
                self.model.jnt_range[i][1]
            )
        # print("Integral position: ", self.integral_qpos)
        self.data.ctrl[:7] = self.integral_qpos[:7]

if __name__ == "__main__":
    CONTROLER = "pos"
    scene_xml_path = "model/franka_emika_panda/scene_"+ CONTROLER + ".xml"
    pin_xml_path = "model/franka_emika_panda/panda_"+ CONTROLER + ".xml"
    env = PandaPbvs(scene_xml_path, pin_xml_path)
    env.run_loop()