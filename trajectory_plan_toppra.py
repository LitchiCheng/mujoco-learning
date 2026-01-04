from scipy.optimize import minimize
import numpy as np
from numpy.linalg import norm, solve
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import pinocchio
import time
import mujoco_viewer as mujoco_viewer

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
    
    def runBefor(self):
        robot = pinocchio.buildModelFromUrdf("model/franka_panda_urdf/robots/panda_arm.urdf")
        print('robot name: ' + robot.name)

        # <key qpos='-1.09146e-23 0.00126288 -3.32926e-07 -0.0696243 -2.28695e-05 0.192135 0.00080101 -5.53841e-09 2.91266e-07'/>
        # <key qpos='0.00296359 0.0163993 0.00368401 -0.0788281 0.259307 0.192303 -0.00312336 -4.3278e-08 1.45579e-07'/>
        # <key qpos='0.00498913 0.246686 0.00381545 -0.0800148 0.415234 0.193705 -0.00425587 3.30553e-07 -3.30357e-07'/>
        # <key qpos='0.00602759 0.424817 0.00377697 -0.0799391 0.371875 0.193076 -0.00368281 1.0024e-06 -1.11103e-07'/>
        # <key qpos='0.00773196 0.822049 0.00373852 -0.0797594 0.315405 0.192995 -0.00319569 1.50177e-06 -9.53418e-07'/>
        # <key qpos='0.00840017 1.08506 0.00374512 -0.0796342 0.304862 0.193412 -0.00331262 2.46441e-06 2.2996e-08'/>
        way_pts = [
            [-1.09146e-23, 0.00126288, -3.32926e-07, -0.0696243, -2.28695e-05, 0.192135, 0.00080101, -5.53841e-09, 2.91266e-07],
            [0.00296359, 0.0163993, 0.00368401, -0.0788281, 0.259307, 0.192303, -0.00312336, -4.3278e-08, 1.45579e-07],
            [0.00498913, 0.246686, 0.00381545, -0.0800148, 0.415234, 0.193705, -0.00425587, 3.30553e-07, -3.30357e-07],
            [0.00602759, 0.424817, 0.00377697, -0.0799391, 0.371875, 0.193076, -0.00368281, 1.0024e-06, -1.11103e-07],
            [0.00773196, 0.822049, 0.00373852, -0.0797594, 0.315405, 0.192995, -0.00319569, 1.50177e-06, -9.53418e-07],
            [0.00840017, 1.08506, 0.00374512, -0.0796342, 0.304862, 0.193412, -0.00331262, 2.46441e-06, 2.2996e-08]
        ]
           
        path_scalars = np.linspace(0, 1, len(way_pts))
        path = ta.SplineInterpolator(path_scalars, way_pts)
        vlim = np.vstack([-robot.velocityLimit, robot.velocityLimit]).T
        al = np.array([2,] * robot.nv)
        alim = np.vstack([-al, al]).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)
        
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc],path,solver_wrapper="seidel")
        jnt_traj = instance.compute_trajectory(0, 0)
        ts_sample = np.linspace(0, jnt_traj.get_duration(), 1000)
        self.qs_sample = jnt_traj.eval(ts_sample)
        self.index = 0
    
    def runFunc(self):
        if self.index < len(self.qs_sample):
            self.data.qpos[:7] = self.qs_sample[self.index][:7]
            self.index += 1
        else:
            self.data.qpos[:7] = self.qs_sample[-1][:7]
        time.sleep(0.01)

test = Test("model/franka_emika_panda/scene.xml")
test.run_loop()