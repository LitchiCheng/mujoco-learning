import mujoco_viewer as mujoco_viewer
import mujoco,time
import numpy as np
import pinocchio

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    get_random_collision_free_transform,
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

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        # Create models and data
        self.model_roboplan, self.collision_model, visual_model = load_models()
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
                model_roboplan, q, gain=0.1, padding=0.025
            ),
        ]  
        self.x = 0.29
        self.init_state = self.data.qpos.copy()

    def runFunc(self):
        self.init_state = get_random_collision_free_state(self.model_roboplan, self.collision_model)
        # target_tform = get_random_collision_free_transform(
        #     self.model_roboplan,
        #     self.collision_model,
        #     self.target_frame,
        #     joint_padding=0.05,
        # )

        theta = np.pi
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        # quat = [0.29, 0.79, -0.50, 0.13]
        # rotation_matrix = pinocchio.Quaternion(*quat).matrix()
        
        target_tform = pinocchio.SE3(rotation_matrix, np.array([self.x, -0.0, 0.69]))

        # print(target_tform)
        q_sol = self.ik.solve(
            self.target_frame,
            target_tform,
            init_state=self.init_state,
            nullspace_components=self.nullspace_components,
            verbose=True,
        )
        # self.init_state = self.data.qpos.copy()
        if q_sol is not None:
            self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
            print(f"End effector position: {self.data.body(self.end_effector_id).xpos}")
            print(f"q_sol: {q_sol}")
            self.data.qpos[:7] = q_sol[:7]
            self.x += 0.001
        else:
            print("No solution found.")
        time.sleep(0.01)

if __name__ == "__main__":
    test = Test("./model/franka_emika_panda/scene.xml")
    test.run_loop()

    