# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Inverse Kinematics practical solutions. """ 

from env.leg_gym_env import LegGymEnv
import numpy as np

def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian 
    J = np.zeros((2,2))
    J[0,0] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])
    J[1,0] =  l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    J[0,1] = -l2 * np.cos(q[0] + q[1]) 
    J[1,1] =  l2 * np.sin(q[0] + q[1])
    
    # foot pos
    pos = np.zeros(2)
    pos[0] =  -l1 * np.sin(q[0]) - l2 * np.sin(q[0]+q[1])
    pos[1] =  -l1 * np.cos(q[0]) - l2 * np.cos(q[0]+q[1])

    return J, pos

def pseudoInverse(A,lam=0.001):
    """ Pseudo inverse of matrix A. 
        Make sure to take into account dimensions of A
            i.e. if A is mxn, what dimensions should pseudoInv(A) be if m>n 
    """
    m,n = np.shape(A)
    pinvA = None

    if m >= n:
        # left pseudoinverse
        pinvA = np.linalg.inv( A.T @ A + lam**2 * np.eye(n) ) @  A.T 
    else:
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
    sideSign = -1
    if angleMode == ">":
        sideSign = 1

    q = np.zeros(2)
    q[1] = sideSign * np.arccos( ((xz[0])**2 + (xz[1])**2 - l1**2 - l2**2) / (2*l1*l2) )
    q[0] = np.arctan2(-xz[0],-xz[1]) - np.arctan2( l2 * np.sin(q[1]), (l1 + l2*np.cos(q[1])) )

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
    while( i < max_i and abs(sum(jacobian_rel(joint_angles)[1] - des_x)) > tol ):
        # Evaluate Jacobian based on current joint angles
        J, ee = jacobian_rel(joint_angles)

        # Compute pseudoinverse
        J_pinv = pseudoInverse(J,lam)
        # J_pinv = J.T # as an aside, this also works (think about why)

        # Find end effector error vector
        ee_error = des_x - ee

        # update joint_angles
        joint_angles += alpha * J_pinv @ ee_error

        # update iteration counter
        i += 1

    return joint_angles


env = LegGymEnv(render=True, 
                on_rack=True,    # set True to debug 
                motor_control_mode='TORQUE',
                action_repeat=1,
                )

NUM_STEPS = 5*1000   # simulate 5 seconds (sim dt is 0.001)
tau = np.zeros(2)    # either torques or motor angles, depending on mode

IK_mode = "GEOMETRICAL"
# IK_mode = "NUMERICAL"

# sample joint PD gains
kpJoint = np.array([55,55])
kdJoint = np.array([0.8,0.8])

# desired foot position
des_foot_pos = np.array([0.1,-0.2]) 

for counter in range(NUM_STEPS):
    # Compute inverse kinematics in leg frame 
    if IK_mode == "GEOMETRICAL":
        # geometrical
        qdes = ik_geometrical(des_foot_pos,angleMode="<")
    else:
        # numerical (switch positive/negative to get < or > configuration)
        qdes = ik_numerical(env._robot_config.INIT_MOTOR_ANGLES,des_foot_pos)
    
    # print 
    if counter % 500 == 0:
        J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
        print('---------------', counter)
        print('q ik',qdes,'q real',env.robot.GetMotorAngles())
        print('ee pos',ee_pos_legFrame)

    # determine torque with joint PD
    tau = kpJoint * (qdes - env.robot.GetMotorAngles()) + kdJoint * (-env.robot.GetMotorVelocities())

    # apply control, simulate
    env.step(tau)