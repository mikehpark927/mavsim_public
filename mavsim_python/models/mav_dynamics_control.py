"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        ##### TODO #####
        # convert steady-state wind vector from world to body frame
        R_v2b = quaternion_to_rotation(self._state[6:10])
        wind_body = R_v2b @ steady_state

        # add the gust 
        self._wind - wind_body + gust

        # convert total wind to world frame
        V = self._state[3:6]

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        Vr = V - self._wind
        
        # compute airspeed (self._Va = ?)
        self._Va = np.linalg.norm(Vr)

        # compute angle of attack (self._alpha = ?)
        self._alpha = np.arctan2(Vr.item(2), Vr.item(0))

        # compute sideslip angle (self._beta = ?)
        self._beta = np.asin(Vr.item(1)/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta.aileron, delta.elevator, delta.rudder, delta.throttle)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        
        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle
        
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        phi = self.true_state.phi
        theta = self.true_state.theta
        psi = self.true_state.psi
        p = self.true_state.p
        q = self.true_state.q
        r = self.true_state.r

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fb_grav = quaternion_to_rotation(self._state[6:10]).T @ np.array([[0, 0, MAV.mass * MAV.gravity]]).T

        # compute Lift and Drag coefficients (CL, CD)
        def sigma(alpha):
            num = 1. + np.exp(-MAV.M*(alpha-MAV.alpha0)) + np.exp(MAV.M*(alpha+MAV.alpha0))
            den = (1. + np.exp(-MAV.M*(alpha-MAV.alpha0))) * (1. + np.exp(MAV.M*(alpha+MAV.alpha0)))
            return num / den
        CL = (1. - sigma(self._alpha)) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) \
            + sigma(self._alpha) * (2 * np.sign(self._alpha) * (np.sin(self._alpha)**2) * np.cos(self._alpha))
            
        # CL = (np.pi * MAV.AR) / (1 + np.sqrt(1 + (MAV.AR/2)**2))
        CD = MAV.C_D_p + ((MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2) / (np.pi * MAV.e * MAV.AR)

        # compute Lift and Drag Forces (F_lift, F_drag)
        dynamic_pressure = MAV.S_wing * 0.5 * MAV.rho * self._Va**2
        F_lift = dynamic_pressure*(CL + (MAV.C_L_q*MAV.c*q/(2*self._Va)) + MAV.C_L_delta_e*delta_e)
        F_drag = dynamic_pressure*(CD + (MAV.C_D_q*MAV.c*q/(2*self._Va)) + MAV.C_D_delta_e*delta_e)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # compute longitudinal forces in body frame (fx, fz)
        Rb_s = np.array([[np.cos(self._alpha), -np.sin(self._alpha)],
                         [np.sin(self._alpha), np.cos(self._alpha)]])
        fx_fz = Rb_s @ np.array([[-F_drag, -F_lift]]).T
        
        # compute lateral forces in body frame (fy)
        fy = dynamic_pressure*(MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*MAV.b*p/(2*self._Va) + MAV.C_Y_r*MAV.b*r/(2*self._Va)+ MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)

        # compute logitudinal torque in body frame (My)
        My = dynamic_pressure*MAV.c*(MAV.C_m_0 + MAV.C_m_alpha*self._alpha + MAV.C_m_q*MAV.c*q/(2*self._Va) + MAV.C_m_delta_e*delta_e)

        # compute lateral torques in body frame (Mx, Mz)
        Mx = dynamic_pressure*MAV.b*(MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p*MAV.b*p/(2*self._Va) + MAV.C_ell_r*MAV.b*r/(2*self._Va) + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        Mz = dynamic_pressure*MAV.b*(MAV.C_n_0 + MAV.C_n_beta*self._beta + MAV.C_n_p*MAV.b*p/(2*self._Va) + MAV.C_n_r*MAV.b*r/(2*self._Va) + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)
        
        # compute full forces
        Fx = fb_grav.item(0) + thrust_prop + fx_fz.item(0)
        Fy = fb_grav.item(1) + fy
        Fz = fb_grav.item(2) + fx_fz.item(1)
        Mx = Mx + torque_prop
        
        forces_moments = np.array([[Fx, Fy, Fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        ##### TODO #####
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max*delta_t

        # Angular speed of propeller (omega_p = ?)
        a = MAV.C_Q0 * MAV.rho * MAV.D_prop**5 / ((2*np.pi)**2)
        b = (MAV.C_Q1 * MAV.rho * MAV.D_prop**4 * Va / (2*np.pi)) + (MAV.KQ**2 / MAV.R_motor)
        c = (MAV.C_Q2 * MAV.rho * MAV.D_prop**3 * Va**2) - (MAV.KQ * v_in) / MAV.R_motor + MAV.KQ * MAV.i0
        
        omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        
        # Advance ratio
        J_op = (2 * np.pi * Va) / (omega_p* MAV.D_prop)
        
        # thrust and torque due to propeller
        C_T = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0
        C_Q = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0
        
        n =  omega_p / (2 * np.pi)
        thrust_prop = MAV.rho * n**2 * np.power(MAV.D_prop,4) * C_T
        torque_prop = -MAV.rho * n**2 * np.power(MAV.D_prop,5) * C_Q

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
