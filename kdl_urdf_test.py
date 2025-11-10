import PyKDL
from kdl_parser.urdf import treeFromFile, treeFromUrdfModel
from urdf_parser_py.urdf import URDF
import numpy as np
import os

def load_urdf_to_kdl(urdf_file_path):
    if not os.path.exists(urdf_file_path):
        print(f"urdf not exists - {urdf_file_path}")
        return False, None
    try:
        success, tree = treeFromFile(urdf_file_path)
        if success:
            print(f"load sucess: {urdf_file_path}")
            print(f"chain has {tree.getNrOfSegments()} segs and {tree.getNrOfJoints()} joints")
            return success, tree
    except Exception as e:
        print(f"methods1 failed: {e}")
    return False, None

def create_chain_from_tree(tree, base_link, tip_link):
    try:
        chain = tree.getChain(base_link, tip_link)
        print(f"create_chain_from_tree create chain from {base_link} to {tip_link}")
        print(f"create_chain_from_tree chain has {chain.getNrOfSegments()} segs and {chain.getNrOfJoints()} joints")
        return chain
    except Exception as e:
        print(f"create chain failed: {e}")
        return None

def setup_kinematics(chain):
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(chain)
    max_iterations = 100
    eps = 1e-6
    ik_pos_solver = PyKDL.ChainIkSolverPos_NR(
        chain, fk_solver, ik_vel_solver, max_iterations, eps
    )
    jac_solver = PyKDL.ChainJntToJacSolver(chain)
    return fk_solver, ik_vel_solver, ik_pos_solver, jac_solver, chain

def perform_forward_kinematics(fk_solver, joint_positions):
    q = PyKDL.JntArray(len(joint_positions))
    for i in range(len(joint_positions)):
        q[i] = joint_positions[i]
    frame = PyKDL.Frame()
    status = fk_solver.JntToCart(q, frame)
    if status >= 0:
        position = [frame.p.x(), frame.p.y(), frame.p.z()]
        rotation = frame.M.GetQuaternion() 
        print(f"fk sucess:")
        print(f"pos: {position}")
        print(f"quat: {rotation}")
        return frame
    else:
        print("fk failed")
        return None

def compute_jacobian(jac_solver, joint_positions):
    num_joints = len(joint_positions)
    q = PyKDL.JntArray(num_joints)
    for i in range(num_joints):
        q[i] = joint_positions[i]
    jac = PyKDL.Jacobian(num_joints)
    status = jac_solver.JntToJac(q, jac)
    if status >= 0:
        jac_matrix = np.zeros((6, num_joints))
        for i in range(6):
            for j in range(num_joints):
                jac_matrix[i, j] = jac[i, j]
        return jac_matrix
    else:
        print("jacobian computation failed")
        return None

def perform_inverse_kinematics(ik_pos_solver, chain, target_frame, initial_joints=None):
    num_joints = chain.getNrOfJoints()
    if initial_joints is None:
        q_init = PyKDL.JntArray(num_joints)
    else:
        q_init = PyKDL.JntArray(num_joints)
        for i in range(num_joints):
            q_init[i] = initial_joints[i]
    q_out = PyKDL.JntArray(num_joints)
    status = ik_pos_solver.CartToJnt(q_init, target_frame, q_out)
    if status >= 0:
        joint_positions = [q_out[i] for i in range(num_joints)]
        print(f"ik sucess:")
        print(f"Joints: {joint_positions}")
        return joint_positions
    else:
        print("ik failed")
        return None

def main():
    urdf_file = "model/franka_panda_urdf/robots/panda_arm.urdf"
    success, tree = load_urdf_to_kdl(urdf_file)
    if not success:
        return
    base_link = "panda_link0"
    tip_link = "panda_link7"
    chain = create_chain_from_tree(tree, base_link, tip_link)
    if chain is None:
        return
    fk_solver, ik_vel_solver, ik_pos_solver, jac_solver, chain = setup_kinematics(chain)
    num_joints = chain.getNrOfJoints()
    # 打印末端姿态
    joint_positions = []
    for i in range(num_joints):
        joint_positions.append(0.1)
    end_effector_frame = perform_forward_kinematics(fk_solver, joint_positions)
    print(end_effector_frame)
    # 打印jacobian
    jac_matrix = compute_jacobian(jac_solver, joint_positions)
    if jac_matrix is not None:
        # [vx, vy, vz, wx, wy, wz]
        print(np.round(jac_matrix, 3))
    # 打印ee速度
    qvel = []
    for i in range(num_joints):
        qvel.append(0.2)
    ee_vel = jac_matrix @ np.array(qvel)
    print(f"ee_vel: {ee_vel}")
    # ik
    if end_effector_frame is not None:
        inverse_joints = perform_inverse_kinematics(ik_pos_solver, chain, end_effector_frame, joint_positions)
        if inverse_joints is not None:
            verified_frame = perform_forward_kinematics(fk_solver, inverse_joints)
            if verified_frame is not None:
                # 计算位置误差
                error_pos = np.sqrt(
                    (verified_frame.p.x() - end_effector_frame.p.x())**2 +
                    (verified_frame.p.y() - end_effector_frame.p.y())** 2 +
                    (verified_frame.p.z() - end_effector_frame.p.z())**2
                )
                print(f"位置误差: {error_pos:.6f} m")

if __name__ == "__main__":
    main()