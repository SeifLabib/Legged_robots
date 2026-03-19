# Inverse Kinematics practical
from env.leg_gym_env import LegGymEnv
import numpy as np
from Lab02 import jacobian_rel


def pseudoInverse(A,lam=0.001):
    """ Pseudo inverse of matrix A. 
        Make sure to take into account dimensions of A
            i.e. if A is mxn, what dimensions should pseudoInv(A) be if m>n 
        Also take into account potential singularities
    """
    m,n = np.shape(A)
    pinvA = None
    if m >= n:
        # left pseudoinverse
        pinvA = np.linalg.inv( A.T @ A + lam**2 * np.eye(n) ) @  A.T
    elif m < n:
        # right pseudoinverse
        pinvA = A.T @ np.linalg.inv( A @ A.T + lam**2 * np.eye(m) )
    return pinvA

def ik_geometrical(xz,angleMode="<",l1=0.209,l2=0.195):
    """ Inverse kinematics based on geometrical reasoning.
        Input: Desired foot xz position (array) 
               angleMode (whether leg should look like > or <) 
               link lengths
        return: joint angles
    """
    q = np.zeros(2)

    sideSign = -1
    if angleMode == ">":
        sideSign = 1

    q[1] = sideSign * np.arccos( ((xz[0])**2 + (xz[1])**2 - l1**2 - l2**2) / (2*l1*l2) )
    q[0] = np.arctan2(-xz[0],-xz[1]) - np.arctan2( l2 * np.sin(q[1]), l1 + l2*np.cos(q[1]) )
    return q

def ik_numerical(q0,des_x,tol=1e-4):
    """ Numerical inverse kinematics
        Input: initial joint angle guess, desired end effector, tolerance
        return: joint angles
    """
    i = 0
    max_i = 100 # max iterations
    alpha = 0.5 # convergence factor
    lam = 0.001 # damping factor for pseudoInverse
    joint_angles = q0

    # Condition to iterate: while fewer than max iterations, and while error is greater than tolerance
    while( i < max_i and 0 ):
        # Evaluate Jacobian based on current joint angles
        # J, ee = 0, 0
        J, ee = jacobian_rel(joint_angles)

        # Compute pseudoinverse
        # J_pinv = 0
        J_pinv = pseudoInverse(J,lam)

        # Find end effector error vector
        # ee_error = 0
        ee_error = des_x - ee

        # update joint_angles
        # joint_angles += 0
        joint_angles += alpha * J_pinv @ ee_error

        # update iteration counter
        i += 1

    return joint_angles


if __name__ == "__main__": 
    env = LegGymEnv(render=True, 
                    on_rack=False,    # set True to debug 
                    motor_control_mode='TORQUE',
                    action_repeat=1,
                    )

    NUM_STEPS = 5*1000   # simulate 5 seconds (sim dt is 0.001)
    tau = np.zeros(2) # either torques or motor angles, depending on mode

    IK_mode = "GEOMETRICAL"

    # sample joint PD gains
    kpJoint = np.array([55,55])
    kdJoint = np.array([0.8,0.8])

    # desired foot position (sample)
    des_foot_pos = np.array([0.1,-0.2]) 
    
    # counter = 0

    # while True:
    for counter in range(NUM_STEPS):
        # Compute inverse kinematics in leg frame 
        if IK_mode == "GEOMETRICAL":
            # geometrical
            qdes = env._robot_config.INIT_MOTOR_ANGLES # ik_geometrical
        else:
            # numerical
            qdes = env._robot_config.INIT_MOTOR_ANGLES # ik_numerical
        
        # print 
        # counter += 1
        if counter % 500 == 0:
            J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
            print('---------------', counter)
            print('q ik',qdes,'q real',env.robot.GetMotorAngles())
            print('ee pos',ee_pos_legFrame)

        # determine torque with joint PD
        tau = np.zeros(2) 

        # apply control, simulate
        env.step(tau)
