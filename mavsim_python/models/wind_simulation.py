"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####

        #   Dryden gust model parameters (pg 56 UAV book)
        #   sigma_u, sigma_v, sigma_w = intensities of the turbulance along the body frame axes
        #   L_u, L_v, L_w = spatial wavelengths
        #   V_a = velocity of the airframe relative to surrounding air mass
        
        self.sigma_u = 1.06 # m/s
        self._Lu = 200 # m
        airspeed = 25 # m/s
        self.sigma_v = self.sigma_u # m/s
        self._Lv = self._Lu # m/s
        self.sigma_w = 0.7 # m/s
        self._Lw = 50 # m
        
        # self.H_u = self.sigma_u * np.sqrt(2*self.V_a / np.pi*self.L_u) * 1 / (self._steady_state + (self.V_a / self.L_u))
        # self.H_v = self.sigma_v * np.sqrt(3*self.V_a / np.pi*self.L_v) * \
        #             ((self._steady_state + (self.V_a / np.sqrt(3*self.L_v)))/ (self._steady_state + (self.V_a / self.L_v))**2)
        # self.H_w = self.sigma_w * np.sqrt(3*self.V_a / np.pi*self.L_v) * \
        #             ((self._steady_state + (self.V_a / np.sqrt(3*self.L_w)))/ (self._steady_state + (self.V_a / self.L_w))**2)

        # # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        # self.u_w = TransferFunction(num=np.array([[self.H_u]]), den=np.array([[1,1]]),Ts=Ts)
        # self.v_w = TransferFunction(num=np.array([[self.H_v,self.H_v]]), den=np.array([[1,1,1]]),Ts=Ts)
        # self.w_w = TransferFunction(num=np.array([[self.H_w,self.H_w]]), den=np.array([[1,1,1]]),Ts=Ts)
        # self._Ts = Ts
        self._Ts = Ts
        airspeed_div_Lu = airspeed / self._Lu
        u_scalar = self.sigma_u * np.sqrt(2. * airspeed_div_Lu)
        self.u_w = TransferFunction(num=np.array([[u_scalar]]), den=np.array([[1, airspeed_div_Lu]]), Ts=self._Ts)
        airspeed_div_Lv = airspeed / self._Lv
        v_scalar = self.sigma_v * np.sqrt(3. * airspeed_div_Lv)
        self.v_w = TransferFunction(num=np.array([[v_scalar, v_scalar * airspeed_div_Lv / np.sqrt(3)]]), den=np.array([[1, 2. * airspeed_div_Lv, airspeed_div_Lv**2]]), Ts=self._Ts)
        airspeed_div_Lw = airspeed / self._Lv
        w_scalar = self.sigma_w * np.sqrt(3. * airspeed_div_Lw)
        self.w_w = TransferFunction(num=np.array([[w_scalar, w_scalar * airspeed_div_Lw / np.sqrt(3)]]), den=np.array([[1, 2. * airspeed_div_Lw, airspeed_div_Lw**2]]), Ts=self._Ts)
        
    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

