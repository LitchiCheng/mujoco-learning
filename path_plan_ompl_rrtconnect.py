import mujoco
import ompl.base as ob
import ompl.geometric as og
import time
import src.mujoco_viewer as mujoco_viewer

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
    
    def runBefore(self):
        state_space = ob.RealVectorStateSpace(self.model.nq)
        bounds = ob.RealVectorBounds(self.model.nq)
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            bounds.setLow(i, self.model.jnt_range[i, 0])
            bounds.setHigh(i, self.model.jnt_range[i, 1])
            print(self.model.jnt_range[i, 0], self.model.jnt_range[i, 1])
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
        start_js = [-2.80558, 0.575492, -0.331959, -0.0709803, 1.21621, 1.03666, 2.21675, 0.0148011, 0.0147989]
        goal_js = [-0.40558, 0.375492, -0.231959, -0.0709803, 1.21621, 1.03666, 1.21675, 0.0148011, 0.0247989]
        for i in range(min(self.model.nq, self.model.jnt_range.shape[0])):
            start[i] = start_js[i]
            goal[i] = goal_js[i]

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = og.RRTConnect(si)
        planner.setRange(0.01)
        # planner.setIntermediateStates(True)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(10.0)
        self.path_states = []
        if solved:
            self.path = pdef.getSolutionPath()
            for i in range(self.path.getStateCount()):
                state = self.path.getState(i)
                state_values = [state[i] for i in range(self.model.nq)]
                self.path_states.append(state_values)
                print(state_values)
        else:
            print("No solution found.")
        self.index = 0

    def runFunc(self):
        if self.index < len(self.path_states):
            self.data.qpos[:7] = self.path_states[self.index][:7]
            self.index += 1
        else:
            self.data.qpos[:7] = self.path_states[-1][:7]
        time.sleep(0.01)

test = Test("model/franka_emika_panda/scene_pos.xml")
test.run_loop()