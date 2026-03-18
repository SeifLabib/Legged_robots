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

""" Jacobian practical solutions. Note the gains here are not optimal!! """ 

from env.leg_gym_env import LegGymEnv
import numpy as np

def jacobian_abs(q,l1=0.209,l2=0.195):
    """ Jacobian based on absolute angles (like double pendulum)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2,2))
    J[0, 0] = l1 * np.cos(q[0]) 
    J[1, 0] = l1 * np.sin(q[0])
    J[0, 1] = l2 * np.cos(q[1]) 
    J[1, 1] = l2 * np.sin(q[1])

    # foot pos
    pos = np.zeros(2)
    pos[0] =  l1 * np.sin(q[0]) + l2 * np.sin(q[1])
    pos[1] = -l1 * np.cos(q[0]) - l2 * np.cos(q[1]) 

    return J, pos

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


env = LegGymEnv(render=True, 
                on_rack=False,    # set True to debug 
                motor_control_mode='TORQUE',
                action_repeat=1,
                )

COMPENSATE_GRAVITY = True

tau  = np.zeros(2) # either torques or motor angles, depending on mode

# note these gains may not be optimal
kpCartesian = np.diag([500,500])
kdCartesian = np.diag([30,30])

if COMPENSATE_GRAVITY:
    kpCartesian = np.diag([200,200])
    kdCartesian = np.diag([10,10])

des_foot_pos = np.array([0.0,-0.3]) 

while True:
    # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
    J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())
    print('ee pos',ee_pos_legFrame)

    # foot velocity in leg frame (use GetMotorVelocities() )
    motor_vel = env.robot.GetMotorVelocities()
    foot_linvel = J @ motor_vel

    # calculate torque
    tau = J.T @ ( kpCartesian @ (des_foot_pos-ee_pos_legFrame) + kdCartesian @ (-foot_linvel))

    # compensate weak Cartesian PD gains
    if COMPENSATE_GRAVITY:
        tau += J.T @ np.array([0,-9.8*env.robot.total_mass])

    # apply control, simulate
    env.step(tau)
