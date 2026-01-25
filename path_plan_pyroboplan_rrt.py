import src.mujoco_viewer as mujoco_viewer
import mujoco,time,threading
import numpy as np
import pinocchio
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path
    
    def runBefore(self):
        # Create models and data
        self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
        add_self_collisions(self.model_roboplan, self.collision_model)
        add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)

        data = self.model_roboplan.createData()
        collision_data = self.collision_model.createData()

        self.target_frame = "panda_hand"
        ignore_joint_indices = [
            self.model_roboplan.getJointId("panda_finger_joint1") - 1,
            self.model_roboplan.getJointId("panda_finger_joint2") - 1,
        ]
        np.set_printoptions(precision=3)
        self.distance_padding = 0.001

        self.init_state = self.data.qpos.copy()

        while True:            
            q_start = self.random_valid_state()
            q_goal = self.random_valid_state()

            # Search for a path
            options = RRTPlannerOptions(
                max_step_size=0.05,
                max_connection_dist=5.0,
                rrt_connect=False,
                bidirectional_rrt=True,
                rrt_star=True,
                max_rewire_dist=5.0,
                max_planning_time=20.0,
                fast_return=True,
                goal_biasing_probability=0.15,
                collision_distance_padding=0.01,
            )
            print("")
            print(f"Planning a path...")
            planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
            q_path = planner.plan(q_start, q_goal)
            if len(q_path) > 0:
                print(f"Got a path with {len(q_path)} waypoints")
            else:
                print("Failed to plan.")

            # Perform trajectory optimization.
            dt = 0.025
            options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=7,
                min_segment_time=0.5,
                max_segment_time=10.0,
                min_vel=-1.5,
                max_vel=1.5,
                min_accel=-0.75,
                max_accel=0.75,
                min_jerk=-1.0,
                max_jerk=1.0,
                max_planning_time=30.0,
                check_collisions=True,
                min_collision_dist=self.distance_padding,
                collision_influence_dist=0.05,
                collision_avoidance_cost_weight=0.0,
                collision_link_list=[
                    "obstacle_box_1",
                    "obstacle_box_2",
                    "obstacle_sphere_1",
                    "obstacle_sphere_2",
                    "ground_plane",
                    "panda_hand",
                ],
            )
            print("Optimizing the path...")
            optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
            traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

            if traj is None:
                print("Retrying with all the RRT waypoints...")
                traj = optimizer.plan(q_path, init_path=q_path)

            if traj is not None:
                print("Trajectory optimization successful")
                traj_gen = traj.generate(dt)
                self.q_vec = traj_gen[1]
                print(f"path has {self.q_vec.shape[1]} points")
                tforms = extract_cartesian_poses(self.model_roboplan, "panda_hand", self.q_vec.T)
                # 提取位置信息
                positions = []
                for tform in tforms:
                    position = tform.translation
                    positions.append(position)

                positions = np.array(positions)

                # 创建 3D 图形
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 绘制位置轨迹
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')

                # 绘制姿态
                for i, tform in enumerate(tforms):
                    position = tform.translation
                    rotation_matrix = tform.rotation
                    # 提取坐标轴方向的向量
                    x_axis = rotation_matrix[:, 0]
                    y_axis = rotation_matrix[:, 1]
                    z_axis = rotation_matrix[:, 2]
                    # 绘制坐标轴向量
                    ax.quiver(position[0], position[1], position[2],
                            x_axis[0], x_axis[1], x_axis[2], color='r', length=0.01)
                    ax.quiver(position[0], position[1], position[2],
                            y_axis[0], y_axis[1], y_axis[2], color='g', length=0.01)
                    ax.quiver(position[0], position[1], position[2],
                            z_axis[0], z_axis[1], z_axis[2], color='b', length=0.01)

                # 设置坐标轴标签
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # 显示图形
                plt.show(block=False)
                plt.pause(0.001)
                break
        
        self.index = 0
        

    def random_valid_state(self):
        return get_random_collision_free_state(
            self.model_roboplan, self.collision_model, distance_padding=0.01
        )

    def runFunc(self):
        self.data.qpos[:7] = self.q_vec[:7, self.index]
        self.index += 1
        if self.index >= self.q_vec.shape[1]:
            self.index = 0
        time.sleep(0.01)

if __name__ == "__main__":
    test = Test("model/franka_emika_panda/scene.xml")
    test.run_loop()

    