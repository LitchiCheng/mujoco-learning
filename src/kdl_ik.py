import PyKDL
from kdl_parser.urdf import treeFromFile, treeFromUrdfModel
from urdf_parser_py.urdf import URDF
import numpy as np
import os        

class Kinematics:
    def __init__(self, ee_frame):
        self.frame_name = ee_frame

    def buildFromURDF(self, urdf_file, base_link):
        if not os.path.exists(urdf_file):
            print(f"urdf not exists - {urdf_file}")
            return 
        try:
            success, tree = treeFromFile(urdf_file)
            if success:
                print(f"load sucess: {urdf_file}")
                print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
        except Exception as e:
            print(f"methods1 failed: {e}")

        try:
            self.chain = tree.getChain(base_link, self.frame_name)
            print(f"create_chain_from_tree create chain from {base_link} to {self.frame_name}")
            print(f"create_chain_from_tree chain has {self.chain.getNrOfSegments()} segs and {self.chain.getNrOfJoints()} joints")
        except Exception as e:
            print(f"create chain failed: {e}")

        self.createSolver()

    def createSolver(self):
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.chain)
        max_iterations = 500
        eps = 1e-6
        # self.ik_pos_solver = PyKDL.ChainIkSolverPos_NR(
        #     self.chain, self.fk_solver, self.ik_vel_solver, max_iterations, eps
        # )
        L = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]).reshape(6, 1)  # 位置权重1.0，姿态权重0.1
        self.ik_pos_solver = PyKDL.ChainIkSolverPos_LMA(
            self.chain,
            L,
            eps=1e-5,              # 收敛精度
            maxiter=500,           # 最大迭代次数
            eps_joints=1e-15,      # 关节变化收敛阈值
        )
        self.jac_solver = PyKDL.ChainJntToJacSolver(self.chain)

    def fk(self, q):
        num_joints = self.chain.getNrOfJoints()
        q_kdl = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            q_kdl[i] = q[i]
        end_frame = PyKDL.Frame()
        self.fk_solver.JntToCart(q_kdl, end_frame)
        tf = np.eye(4, dtype=np.float64)
        rot_mat = np.array([
            [end_frame.M[0,0], end_frame.M[0,1], end_frame.M[0,2]],
            [end_frame.M[1,0], end_frame.M[1,1], end_frame.M[1,2]],
            [end_frame.M[2,0], end_frame.M[2,1], end_frame.M[2,2]]
        ], dtype=np.float64)
        tf[:3, :3] = rot_mat
        tf[:3, 3] = np.array([end_frame.p.x(), end_frame.p.y(), end_frame.p.z()])
        return tf
      
    def ik(self, tf , current_arm_motor_q = None, current_arm_motor_dq = None):
        num_joints = self.chain.getNrOfJoints()
        if current_arm_motor_q is None:
            q_init = PyKDL.JntArray(num_joints)
        else:
            q_init = PyKDL.JntArray(num_joints)
            for i in range(num_joints):
                q_init[i] = current_arm_motor_q[i]
        q_out = PyKDL.JntArray(num_joints)
        ## 提取平移向量（前3行第4列）
        trans = PyKDL.Vector(tf[0, 3], tf[1, 3], tf[2, 3])
        ## 提取旋转矩阵（前3行前3列），构造PyKDL.Rotation
        # Rotation构造参数：xx, xy, xz, yx, yy, yz, zx, zy, zz（行优先展开）
        rot = PyKDL.Rotation(
            tf[0, 0], tf[0, 1], tf[0, 2],
            tf[1, 0], tf[1, 1], tf[1, 2],
            tf[2, 0], tf[2, 1], tf[2, 2]
        )
        frame = PyKDL.Frame(rot, trans)
        dof = []
        status = self.ik_pos_solver.CartToJnt(q_init, frame, q_out)
        if status >= 0:
            dof = [q_out[i] for i in range(num_joints)]
            info = {"success": True}
        else:
            dof = q_init
            info = {"success": False, "error_code": status}
        return dof, info

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import utils

    arm = Kinematics("ee_center_body")
    urdf_file = "../model/franka_panda_urdf/robots/panda_arm.urdf"
    arm.buildFromURDF(urdf_file, "link0")
    tf = utils.transform2mat(0.3, 0.0, 0.3, np.pi, 0, 0)
    dof, info = arm.ik(tf)
    print(f"DoF: {dof}, Info: {info}")
    print(f"FK: {arm.fk(dof)}")