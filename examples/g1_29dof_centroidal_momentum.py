import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

def cm_func(urdf_path:str):
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    print("model name: " + model.name)

    # Create casadi model and data
    cmodel = cpin.Model(model)
    for j in range(cmodel.njoints):
        print(j, cmodel.names[j], cmodel.joints[j].nq, cmodel.joints[j].nv)
    cdata = cmodel.createData()
    
    # make symbolic inputs
    q_sym = casadi.SX.sym("q_sym", model.nq)
    q_dot_sym = casadi.SX.sym("q_dot_sym", model.nv)
    
    CM = cpin.computeCentroidalMomentum(cmodel, cdata, q_sym, q_dot_sym)
    
    lin_momentum_func = casadi.Function("g1_29dof_lin_momentum_func", [q_sym, q_dot_sym], [CM.linear])
    ang_momentum_func = casadi.Function("g1_29dof_ang_momentum_func", [q_sym, q_dot_sym], [CM.angular])
    
    return lin_momentum_func, ang_momentum_func

if __name__ == "__main__":
    urdf_path = "../unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
    lin_momentum_func, ang_momentum_func = cm_func(urdf_path)
    print(lin_momentum_func)
    print(ang_momentum_func)
    print("num instructions lin momentum func: ", lin_momentum_func.n_instructions())
    print("num instructions ang momentum func: ", ang_momentum_func.n_instructions())
    
    lin_momentum_func.save("g1_29dof_lin_momentum_func.casadi")
    ang_momentum_func.save("g1_29dof_ang_momentum_func.casadi")
    
    # eg
    import numpy as np
    joint_pos = np.zeros(29 + 7)
    joint_vel = np.random.randn(29 + 6)
    lin_momentum = lin_momentum_func(joint_pos, joint_vel)
    ang_momentum = ang_momentum_func(joint_pos, joint_vel)
    print("lin momentum: ", lin_momentum)
    print("ang momentum: ", ang_momentum)
