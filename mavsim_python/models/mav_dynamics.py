"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
        7/13/2023 - RWB
        1/17/2024 - RWB
"""
import numpy as np
# load message types
from message_types.msg_state import MsgState
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()

        # initialize physical parameters
        self.mass = 11.0  # kg
        self.Jx = 0.824  # kg-m^2
        self.Jy = 1.135  # kg-m^2
        self.Jz = 1.759  # kg-m^2
        self.Jxz = 0.120  # kg-m^2

        tau = self.Jx*self.Jz - self.Jxz**2
        self.T1 = (1/tau)*self.Jxz*(self.Jx-self.Jy+self.Jz)
        self.T2 = (1/tau)*self.Jz*(self.Jz-self.Jy)+self.Jxz**2
        self.T3 = self.Jz/tau
        self.T4 = self.Jxz/tau
        self.T5 = (self.Jz - self.Jx)/self.Jy
        self.T6 = self.Jxz/self.Jy
        self.T7 = (1/tau)*(self.Jx - self.Jy)*self.Jx + self.Jxz**2
        self.T8 = self.Jx/tau


    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

    def _f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        ##### TODO #####
        
        # Extract the States
        north = state.item(0)
        east = state.item(1)
        down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(9)
        q = state.item(10)
        r = state.item(11)

        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        Mx = forces_moments.item(3)
        My = forces_moments.item(4)
        Mz = forces_moments.item(5)

        # Position Kinematics
        arr1 = np.array([[e1**2 + e0**2 - e2**2 -e3**2, 2*(e1*e2 - e3*e0), 2*(e1*e3 + e2*e0)], 
                        [2*(e1*e2 + e3*e0), e2**2 + e0**2 - e1**2 - e3**2, 2*(e2*e3 - e1*e0)],
                        [2*(e1*e3 - e2*e0), 2*(e2*e3 + e1*e0), e3**2 + e0**2 - e1**2 - e2**2]])
        arr2 = np.array([[u, v, w]]).T
        pos_dot = np.dot(arr1,arr2)

        # Position Dynamics
        arr3 = np.array([[r*v - q*w],
                        [p*w - r*u], 
                        [q*u - p*v]])
        arr4 = (1/self.mass* np.array([[fx, fy, fz]]).T)
        u_dot = arr3 + arr4

        # rotational kinematics
        arr5 = np.array([[0, -p, -q, -r],
                        [p, 0, r, -q], 
                        [q, -r, 0, p], 
                        [r, q, -p, 0]])
        arr6 = np.array([[e0, e1, e2, e3]]).T
        e0_dot = 0.5*np.dot(arr5,arr6)

        # rotatonal dynamics
        arr7 = np.array([[self.T1*p*q - self.T2*q*r],
                        [self.T5*p*r - self.T6*(p**2 - r**2)], 
                        [self.T7*p*q - self.T1*q*r]])
        arr8 = np.array([[self.T3*Mx + self.T4*Mz], 
                        [1/self.Jy*My], 
                        [self.T4*Mx + self.T8*Mz]])
        p_dot = arr7 + arr8

        # collect the derivative of the states
        x_dot = np.array([[pos_dot.item(0), 
                        pos_dot.item(1), 
                        pos_dot.item(2), 
                        u_dot.item(0), 
                        u_dot.item(1), 
                        u_dot.item(2), 
                        e0_dot.item(0), 
                        e0_dot.item(1), 
                        e0_dot.item(2), 
                        e0_dot.item(3), 
                        p_dot.item(0), 
                        p_dot.item(1), 
                        p_dot.item(2)]]).T
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0