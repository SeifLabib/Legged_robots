# Jacobian practical
from env.leg_gym_env import LegGymEnv
import numpy as np

def jacobian_abs(q,l1=0.209,l2=0.195):
    """ Jacobian based on absolute angles (like double pendulum)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    # [TODO]
    J[0, 0] = l1 * np.cos(q[0])
    J[1, 0] = l1 * np.sin(q[0])
    J[0, 1] = l2 * np.cos(q[1])
    J[1, 1] = l2 * np.sin(q[1])

    # foot pos
    pos = np.zeros(2)
    # [TODO]
    pos[0] = l1 * np.sin(q[0]) + l2 * np.sin(q[1])
    pos[1] = -l1 * np.cos(q[0]) - l2 * np.cos(q[1])

    return J, pos

def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    # [TODO]
    J[0, 0] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])
    J[1, 0] =  l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    J[0, 1] = -l2 * np.cos(q[0] + q[1])
    J[1, 1] =  l2 * np.sin(q[0] + q[1])

    # foot pos
    pos = np.zeros(2)
    # [TODO]
    pos[0] = -l1 * np.sin(q[0]) - l2 * np.sin(q[0]+q[1])
    pos[1] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0]+q[1])

    return J, pos

if __name__ == "__main__":
    env = LegGymEnv(render=True, 
                    on_rack=False,    # set True to hang up robot in air 
                    motor_control_mode='TORQUE',
                    action_repeat=1,
                    )

    NUM_STEPS = 5*1000 # simulate 5 seconds (sim dt is 0.001)

    env._robot_config.INIT_MOTOR_ANGLES = np.array([-np.pi/4 , np.pi/2]) # test different initial motor angles
    obs = env.reset() # reset environment if changing initial configuration 

    action = np.zeros(2) # either torques or motor angles, depending on mode

    # Test different Cartesian gains! How important are these? 
    kpCartesian = np.diag([500]*2)
    kdCartesian = np.diag([30]*2)

    # test different desired foot positions
    des_foot_pos = np.array([0.05,-0.3]) 

    # for _ in range(NUM_STEPS):
    while True:
        # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
        motor_ang = env.robot.GetMotorAngles()
        # J, foot_pos = np.zeros((2,2)), np.zeros(2) # [TODO]
        J, foot_pos = jacobian_rel(motor_ang)
        # print("foot pos: ", foot_pos)

        # Get foot velocity in leg frame (use GetMotorVelocities() )
        motor_vel = env.robot.GetMotorVelocities()
        # foot_vel = np.zeros(2) # [TODO]
        foot_vel = J @ motor_vel
        # print("foot vel: ", foot_vel)

        # Calculate torque (Cartesian PD, and/or desired force)
        # tau = np.zeros(2) # [TODO]
        tau = J.T @ (kpCartesian @ (des_foot_pos - foot_pos) + kdCartesian @ (-foot_vel))
        
        # add gravity compensation (Force), (get mass with env.robot.total_mass)
        # [TODO]
        tau += J.T @ np.array([0, -9.8 * env.robot.total_mass])

        action = tau
        # apply control, simulate
        env.step(action)


    # make plots of joint positions, foot positions, torques, etc.
    # [TODO]
