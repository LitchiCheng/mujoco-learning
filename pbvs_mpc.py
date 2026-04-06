import time 
import mujoco
import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import src.mujoco_viewer as mujoco_viewer
import src.matplot as mp
import src.utils as utils

class PbvsMpc:
    def __init__(self, mjcf, frame_name, N, Ts) -> None:
        self.mjcf = mjcf
        self.frame_name = frame_name
        self.buildFromMJCF(mjcf)
        self.joints_num = self.mjcf_model.nq
        self.ee_id = self.mjcf_model.getFrameId(self.frame_name)

        self.N = N          # 预测时域
        self.Ts = Ts        # 控制周期
        self.Q_p = 20.0 * np.eye(3)   # 位置权重
        self.Q_r = 5.0  * np.eye(3)   # 旋转权重
        self.R = 0.001 * np.eye(self.joints_num)  # 速度惩罚
        self.q_min = self.mjcf_model.lowerPositionLimit
        self.q_max = self.mjcf_model.upperPositionLimit
        self.dq_min = np.array([-6.175]*self.joints_num)
        self.dq_max = np.array([ 6.175]*self.joints_num)

        # 符号变量
        self.q_sym = ca.SX.sym("q", self.joints_num, 1)
        self.ref_tf_sym = ca.SX.sym("ref_tf", 4, 4)

        # 正运动学
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.q_sym)

        # 位置误差函数
        self.translational_error = ca.Function(
            "translational_error",
            [self.q_sym, self.ref_tf_sym],
            [self.cdata.oMf[self.ee_id].translation - self.ref_tf_sym[:3, 3]]
        )

        # 旋转误差函数
        self.rotational_error = ca.Function(
            "rotational_error",
            [self.q_sym, self.ref_tf_sym],
            [cpin.log3(self.cdata.oMf[self.ee_id].rotation @ self.ref_tf_sym[:3,:3].T)]
        )

        self.opti = ca.Opti()

        # 预测时域内的关节速度 dq[0...N-1]
        self.var_dq = self.opti.variable(self.joints_num, self.N)

        # 当前关节角、目标位姿
        self.param_q0 = self.opti.parameter(self.joints_num)
        self.param_ref_tf = self.opti.parameter(4, 4)

        total_cost = 0
        q_prev = self.param_q0

        for k in range(self.N):
            dq_k = self.var_dq[:, k]
            q_k = q_prev + dq_k * self.Ts

            e_p = self.translational_error(q_k, self.param_ref_tf)
            e_r = self.rotational_error(q_k, self.param_ref_tf)

            total_cost += ca.bilin(self.Q_p, e_p, e_p)
            total_cost += ca.bilin(self.Q_r, e_r, e_r)
            total_cost += ca.bilin(self.R, dq_k, dq_k)

            self.opti.subject_to(self.q_min <= q_k)
            self.opti.subject_to(q_k <= self.q_max)
            self.opti.subject_to(self.dq_min <= dq_k)
            self.opti.subject_to(dq_k <= self.dq_max)

            q_prev = q_k

        self.opti.minimize(total_cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            'ipopt.tol': 1e-3,
            "ipopt.sb": "yes"
        }

        self.opti.solver("ipopt", opts)

    def buildFromMJCF(self, mjcf_file):
        self.arm = pin.RobotWrapper.BuildFromMJCF(mjcf_file)
        self.mjcf_model = self.arm.model
        self.mjcf_data = self.arm.data
        self.cmodel = cpin.Model(self.mjcf_model)
        self.cdata = self.cmodel.createData()

    def solve(self, T_object, current_q, initial_dq=None):
        self.opti.set_value(self.param_q0, current_q)
        self.opti.set_value(self.param_ref_tf, T_object)

        if initial_dq is None:
            self.opti.set_initial(self.var_dq, np.zeros((self.joints_num, self.N)))
        else:
            initial_dqs = np.zeros((self.joints_num, self.N))
            initial_dqs[:, 0] = initial_dq
            self.opti.set_initial(self.var_dq, initial_dqs)

        try:
            sol = self.opti.solve()
            dq_sol = sol.value(self.var_dq)
            dq_cmd = dq_sol[:, 0]
            return dq_cmd, True
        except:
            print("solve failed")
            return np.zeros(self.joints_num), False

class PandaPbvs(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, calc_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.model.opt.timestep = 0.005
        self.mpc = PbvsMpc(calc_xml, frame_name="ee_center_body", N=3, Ts=self.model.opt.timestep)
        self.joints_num = self.model.nq
        self.plot_manager = mp.MultiChartRealTimePlotManager()
        for i in range(self.joints_num):
            self.plot_manager.addNewFigurePlotter("j"+str(i), "j"+str(i)+"_vel", row=i, col=0)

    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]

        # 给一个虚拟位置
        self.T_object = np.eye(4)
        self.T_object[:3,3] = [0.5, 0.1, 0.2]
        roll = np.pi
        pitch = 0.0
        yaw = 0.0
        R_des = utils.euler2rotmat(roll, pitch, yaw)
        self.T_object[:3, :3] = R_des

    def runFunc(self):
        dp_cmd, ok = self.mpc.solve(self.T_object, self.data.qpos, self.data.qvel)
        if ok:
            # print(dp_cmd)
            for i in range(self.joints_num):
                self.plot_manager.updateDataToPlotter("j"+str(i), "j"+str(i)+"_vel", dp_cmd[i])
            self.data.ctrl[:7] = dp_cmd[:7]
        else:
            self.data.ctrl[:7] = np.zeros(7)
        print(self.getBodyPositionByName("ee_center_body"))

if __name__ == "__main__":
    scene_xml_path = "model/franka_emika_panda/scene_vel.xml"
    calc_xml = "model/franka_emika_panda/panda_vel.xml"
    env = PandaPbvs(scene_xml_path, calc_xml)
    env.run_loop()
