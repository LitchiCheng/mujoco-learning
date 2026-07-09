from pathlib import Path
from sys import argv
 
import pinocchio as pin
import numpy as np  

def test_urdf_match_with_mjcf(model):
    print("model name: " + model.name)
    print("lowerLimits: " + str(model.lowerPositionLimit))
    print("upperLimits: " + str(model.upperPositionLimit))
    
    data = model.createData()
    q_list = [ 1.97125175, -0.37236355, 1.64044676, -0.67488302, 2.38533178, 0.72726866, -0.95481822]
    q = np.array(q_list, dtype=np.float64)
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(model, data, q)
    # Print out the placement of each joint of the kinematic tree
    for name, oMi in zip(model.names, data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

if __name__ == '__main__':
    print("mjcf test")
    model_mjcf = pin.RobotWrapper.BuildFromMJCF("model/franka_emika_panda/panda_remove_finger.xml").model
    test_urdf_match_with_mjcf(model_mjcf)
    print("")
    print("urdf test")
    model_urdf = pin.buildModelFromUrdf("model/franka_panda_urdf/robots/panda_arm.urdf")
    test_urdf_match_with_mjcf(model_urdf)
