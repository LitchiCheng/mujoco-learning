import pinocchio 
import mujoco_viewer
import time
import numpy as np
import mujoco

class PandaPbvs(mujoco_viewer.CustomViewer):
    def __init__(self, mujoco_xml):
        super().__init__(mujoco_xml, 3, azimuth=-45, elevation=-30)
        self.mujoco_xml = mujoco_xml

    def runBefore(self):
        self.pin_model = pinocchio.RobotWrapper.BuildFromMJCF(self.mujoco_xml)
        self.urdf_model = self.pin_model.model
        reduce_joint_ids = [8,9]
        q0 = pinocchio.neutral(self.urdf_model)
        self.urdf_model = pinocchio.buildReducedModel(
            self.urdf_model, 
            reduce_joint_ids,
            q0,         
        )
        self.urdf_data = self.urdf_model.createData()  # 简化模型的数据对象
        self.pin_id = self.urdf_model.getFrameId("ee_center_body")        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]  # 设定初始位置
        self.prev_joint_vel = np.zeros(9, dtype=np.float32)  # 记录上一步速度
        self.keep_pos = False
        self.start_qpos = self.data.qpos.copy()


    def dampedPinv(self, J, lambda_d=0.1):
        J_T = J.T
        # 阻尼项：λ²·I₆（6×6单位矩阵）
        damping = lambda_d ** 2 * np.eye(J.shape[0])
        # 计算 (J·Jᵀ + λ²I)⁻¹·Jᵀ
        J_pinv_damped = np.dot(J_T, np.linalg.inv(np.dot(J, J_T) + damping))
        return J_pinv_damped

    def runFunc(self):
        np.set_printoptions(precision=3)
        pinocchio.forwardKinematics(self.urdf_model, self.urdf_data, self.data.qpos[:7])
        pinocchio.updateFramePlacements(self.urdf_model, self.urdf_data)
        J = pinocchio.computeFrameJacobian(self.urdf_model, self.urdf_data, self.data.qpos[:7], self.pin_id, pinocchio.ReferenceFrame.WORLD)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        k_p = 1.0
        p_des = np.array([0.1, -0.2, 0.6])
        e_p =  ee_pos - p_des
        v_des_lin = -k_p * e_p
        # 获取当前末端位姿
        oMf = self.urdf_data.oMf[self.pin_id]
        R_ee = oMf.rotation
        roll = 0.0 
        pitch = 0.0
        yaw = np.pi
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_des = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])
        # 姿态误差
        e_rot = 0.5 * (np.cross(R_ee[:, 0], R_des[:, 0]) +
                    np.cross(R_ee[:, 1], R_des[:, 1]) +
                    np.cross(R_ee[:, 2], R_des[:, 2]))
        # 最终期望末端速度（6维）
        v_e_desired = np.concatenate([v_des_lin, -k_p * e_rot])
        # v_e_desired = np.concatenate([v_des_lin, [0, 0, 0]])  # [线速度3维, 角速度3维]
        J_pinv_damped = self.dampedPinv(J, lambda_d=0.1)  # 阻尼伪逆（抗奇异）
        q_dot_core = J_pinv_damped @ v_e_desired  # 核心任务关节速度
        q_dot_max = 6.24
        q_dot_des = np.clip(q_dot_core, -q_dot_max, q_dot_max)  # 限幅后关节速度
        pin_pose = self.urdf_data.oMf[self.pin_id].translation
        self.start_qpos[:7] += q_dot_des * self.model.opt.timestep  # 更新关节位置
        print(f"{e_p.round(3)}, {e_rot.round(3)}")
        if np.linalg.norm(e_p) < 1e-3:
            self.data.ctrl[:7] = 0
            self.keep_pos = True
        elif not self.keep_pos:
            self.data.ctrl[:7] = self.start_qpos[:7]

if __name__ == "__main__":
    mujoco_xml_path = "model/franka_emika_panda/panda_pos.xml"
    env = PandaPbvs(mujoco_xml_path)
    env.run_loop()