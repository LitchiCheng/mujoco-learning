import mujoco
import ompl.base as ob
import ompl.geometric as og
import time
import mujoco_viewer
import src.casadi_ik as casadi_ik
import src.kdl_ik as kdl_ik
import src.key_listener as key_listener
import utils
import numpy as np
from pynput import keyboard

key_states = {
    keyboard.Key.down: False
}

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, rendor_path, arm_path):
        super().__init__(rendor_path, 3, azimuth=-90, elevation=-30)
        self.arm_path = arm_path
        
        self.obstacles_size = []
        self.obstacles_pos = []
        self.obstacles_rgba = []
        self.obstacles_type = []

        # self.obstacles_type.append("box")
        # self.obstacles_size.append([0.07, 0.07, 0.07])
        # self.obstacles_pos.append([0.25, 0.22, 0.5])
        # self.obstacles_rgba.append([0.4, 0.3, 0.3, 0.8])

        self.obstacles_type.append("sphere")
        self.obstacles_size.append([0.06])
        self.obstacles_pos.append([0.3, 0.2, 0.5])
        self.obstacles_rgba.append([0.3, 0.3, 0.3, 0.8])
        self.addObstacles(self.obstacles_pos, self.obstacles_type, self.obstacles_size, self.obstacles_rgba)
        
        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
    
    def getBipolarJoints(self):
        self.initial_pos = self.model.key_qpos[0]  
        print("start dof", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        self.start_dof = self.data.qpos[:9].copy()

        euler = np.random.uniform(0, 2*np.pi, 3)
        tf = utils.transform2mat(self.goal_x, self.goal_y, self.goal_z, euler[0], euler[1], euler[2])
        self.solver = casadi_ik.Kinematics(self.ee_body_name)
        self.solver.buildFromMJCF(self.arm_path)
        self.dof, info = self.solver.ik(tf, current_arm_motor_q=self.start_dof)
        print("goal dof", self.dof)
        self.goal_dof = self.dof
        if len(self.goal_dof) < 9:
            self.goal_dof = np.concatenate((self.goal_dof, self.start_dof[7:]))
        
    def pathPlaning(self):
        state_space = ob.RealVectorStateSpace(self.model.nq)
        bounds = ob.RealVectorBounds(self.model.nq)
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            bounds.setLow(i, self.model.jnt_range[i, 0])
            bounds.setHigh(i, self.model.jnt_range[i, 1])
        state_space.setBounds(bounds)
        si = ob.SpaceInformation(state_space)

        def is_state_valid(state):
            self.data.qpos[:7] = [state[i] for i in range(7)]
            mujoco.mj_step(self.model, self.data)
            return self.data.ncon == 0

        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        si.setup()
        start = ob.State(state_space)
        goal = ob.State(state_space)
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            start[i] = self.start_dof[i]
            goal[i] = self.goal_dof[i]

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        opt = ob.PathLengthOptimizationObjective(si)
        pdef.setOptimizationObjective(opt)
        planner = og.RRTConnect(si)
        self.planning_range = 0.01
        planner.setRange(self.planning_range)
        planner.setIntermediateStates(True)
        planner.setProblemDefinition(pdef)
        planner.setup()
        self.planning_timeout = 3.0
        solved = planner.solve(self.planning_timeout)
        self.path_states = []
        if solved:
            self.path = pdef.getSolutionPath()
            for i in range(self.path.getStateCount()):
                state = self.path.getState(i)
                state_values = [state[i] for i in range(self.model.nq)]
                self.path_states.append(state_values)
                # print(state_values)
        else:
            print("No solution found.")
        self.index = 0
        return solved
    
    def createTask(self):
        try_cnt = 10
        self.success = False
        for i in range(try_cnt):
            self.getBipolarJoints()
            self.success = self.pathPlaning()
            if self.success:
                break
            print("Try again... cnt ", i)
    
    def runBefore(self):
        self.model.opt.timestep = 0.005
        self.ee_body_name = "ee_center_body"
        self.goal_x = 0.4
        self.goal_y = 0.3
        self.goal_z = 0.4
        self.usr_geom_size = []
        self.usr_geom_pos = []
        self.usr_geom_rgba = []
        self.usr_geom_type = []
        self.usr_geom_pos.append([self.goal_x, self.goal_y, self.goal_z])
        self.usr_geom_type.append("sphere")
        self.usr_geom_size.append([0.02])
        self.usr_geom_rgba.append([0.1, 0.3, 0.3, 0.8])
        self.addVisuGeom(self.usr_geom_pos, self.usr_geom_type, self.usr_geom_size, self.usr_geom_rgba)
        self.createTask()

    def runFunc(self):
        if not self.success:
            return 
        if(len(self.path_states) == 0):
            return
        if self.index < len(self.path_states):
            self.data.qpos[:7] = self.path_states[self.index][:7]
            self.index += 1
        else:
            self.data.qpos[:7] = self.path_states[-1][:7]
            if key_states[keyboard.Key.down]:
                print("re create task")
                self.index = 0
                self.createTask()
        
if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_pos.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda_pos.xml'
    test = Test(SCENE_XML_PATH, ARM_XML_PATH)
    test.run_loop()