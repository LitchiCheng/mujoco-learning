import pinocchio 
import mujoco_viewer as mujoco_viewer
import time
import numpy as np
import mujoco
import utils

class PandaPbvs(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, pin_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.pin_xml = pin_xml

    def runBefore(self):
        self.model.opt.timestep = 0.02
        self.pin_model = pinocchio.RobotWrapper.BuildFromMJCF(self.pin_xml)
        self.urdf_model = self.pin_model.model
        # 删除8,9夹抓部分
        reduce_joint_ids = [8,9]
        q0 = pinocchio.neutral(self.urdf_model)
        self.urdf_model = pinocchio.buildReducedModel(
            self.urdf_model, 
            reduce_joint_ids,
            q0,         
        )
        # 简化模型的数据对象
        self.urdf_data = self.urdf_model.createData()
        BODY_NAME = "ee_center_body"
        self.pin_id = self.urdf_model.getFrameId(BODY_NAME)        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        # 设定初始位置
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        # 记录上一步速度  
        self.prev_joint_vel = np.zeros(9, dtype=np.float32)  
        self.keep_pos = False
        self.integral_qpos = self.data.qpos.copy()


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
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        # 线速度控制
        k_p_lin = 1.0
        x = 0.5
        y = 0.0
        z = 0.1
        p_des = np.array([x, y, z])
        e_p =  ee_pos - p_des
        v_des_lin = -k_p_lin * e_p
        # 旋转速度控制
        k_p_rot = 1.0
        R_ee = utils.quat2rotmat(ee_quat)
        roll = np.pi
        pitch = 0.0
        yaw = 0.0
        R_des = utils.euler2rotmat(roll, pitch, yaw)
        # 姿态误差
        e_rot = 0.5 * (np.cross(R_des[:, 0], R_ee[:, 0]) +
                    np.cross(R_des[:, 1], R_ee[:, 1]) +
                    np.cross(R_des[:, 2], R_ee[:, 2]))
        v_des_rot = -k_p_rot * e_rot
        # 最终期望末端速度（6维）
        v_e_desired = np.concatenate([v_des_lin, v_des_rot])
        # v_e_desired = np.concatenate([v_des_lin, [0, 0, 0]])
        # v_e_desired = np.concatenate([[0, 0, 0], v_des_rot])
        J_pinv_damped = self.dampedPinv(J, lambda_d=0.05)
        q_dot_core = J_pinv_damped @ v_e_desired
        q_dot_max = 3.14
        q_dot_des = np.clip(q_dot_core, -q_dot_max, q_dot_max)  # 限幅后关节速度
        self.integral_qpos[:7] += q_dot_des * self.model.opt.timestep  # 更新关节位置
        print(f"{e_p.round(3)}, {e_rot.round(3)}")
        if np.linalg.norm(e_p) < 1e-3:
            self.keep_pos = True
        elif not self.keep_pos:
            placeholder = 0
            # self.data.ctrl[:7] = self.integral_qpos[:7]
            self.data.ctrl[:7] = q_dot_des[:7]

if __name__ == "__main__":
    CONTROLER = "vel"
    scene_xml_path = "model/franka_emika_panda/scene_"+ CONTROLER + ".xml"
    pin_xml_path = "model/franka_emika_panda/panda_"+ CONTROLER + ".xml"
    env = PandaPbvs(scene_xml_path, pin_xml_path)
    env.run_loop()