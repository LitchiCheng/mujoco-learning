import src.mujoco_viewer as mujoco_viewer
import mujoco,time,threading
import numpy as np
import pinocchio
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import (
    joint_limit_nullspace_component,
    collision_avoidance_nullspace_component,
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

        # Set up the IK solver
        options = DifferentialIkOptions(
            max_iters=200,
            max_retries=10,
            damping=0.0001,
            min_step_size=0.05,
            max_step_size=0.1,
            ignore_joint_indices=ignore_joint_indices,
            rng_seed=None,
        )
        self.ik = DifferentialIk(
            self.model_roboplan,
            data=data,
            collision_model=self.collision_model,
            options=options,
            visualizer=None,
        )
        self.nullspace_components = [
            lambda model_roboplan, q: collision_avoidance_nullspace_component(
                model_roboplan,
                data,
                self.collision_model,
                collision_data,
                q,
                gain=1.0,
                dist_padding=0.05,
            ),
            lambda model_roboplan, q: joint_limit_nullspace_component(
                model_roboplan, q, gain=1.0, padding=0.025
            ),
        ]
        
        theta = np.pi
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        q_start = self.getIk(self.random_valid_state(), rotation_matrix, [0.4, 0.0, 0.4])
        q_goal = self.getIk(self.random_valid_state(), rotation_matrix, [0.3, 0.0, 0.5])

        while True:
            # Search for a path
            options = RRTPlannerOptions(
                max_step_size=0.05,
                max_connection_dist=0.5,
                rrt_connect=False,
                bidirectional_rrt=False,
                rrt_star=True,
                max_rewire_dist=0.5,
                max_planning_time=5.0,
                fast_return=True,
                goal_biasing_probability=0.25,
                collision_distance_padding=0.0001,
            )
            print("")
            print(f"Planning a path...")
            planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)

            q_path = planner.plan(q_start, q_goal)
            print(f"Path found with {len(q_path)} waypoints")
            if q_path is not None and len(q_path) > 0:
                print(f"Got a path with {len(q_path)} waypoints")
                if len(q_path) > 50:
                    print("Path is too long, skipping...")
                    continue
            else:
                print("Failed to plan.")

            # Perform trajectory optimization.
            dt = 0.025
            options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=1,
                min_segment_time=0.5,
                max_segment_time=10.0,
                min_vel=-1.5,
                max_vel=1.5,
                min_accel=-0.75,
                max_accel=0.75,
                min_jerk=-1.0,
                max_jerk=1.0,
                max_planning_time=1.0,
                check_collisions=False,
                min_collision_dist=0.001,
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
                self.tforms = extract_cartesian_poses(self.model_roboplan, "panda_hand", self.q_vec.T)
                self.plot(self.tforms)
                break
        self.index = 0

    def plot(self, tfs):
        positions = []
        for tform in tfs:
            position = tform.translation
            positions.append(position)
        positions = np.array(positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show(block=False)
        plt.pause(0.001)

    def getIk(self, init_state, rotation_matrix, pose):
        while True:
            self.init_state = init_state
            target_tform = pinocchio.SE3(rotation_matrix, np.array(pose))
            q_sol = self.ik.solve(
                self.target_frame,
                target_tform,
                init_state=self.init_state,
                nullspace_components=self.nullspace_components,
                verbose=True,
            )
            if q_sol is not None:
                print("IK solution found")
                return q_sol

    def random_valid_state(self):
        return get_random_collision_free_state(
            self.model_roboplan, self.collision_model, distance_padding=0.001
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

    