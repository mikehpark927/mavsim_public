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
        self.Lu = 200 # m
        Va = 25 # m/s
        self.sigma_v = self.sigma_u # m/s
        self.Lv = self.Lu # m/s
        self.sigma_w = 0.7 # m/s
        self.Lw = 50 # m
        
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
        self._A = np.array([[-Va/self.Lu, 0, 0, 0, 0],
                            [0, -2*(Va/self.Lv), -(Va/self.Lv)**2, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, -2*(Va/self.Lw), -(Va/self.Lw)**2],
                            [0, 0, 0, 1, 0]])
        self._B = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        self._C = np.array([[self.sigma_u * np.sqrt((2*Va)/self.Lu), 0, 0, 0, 0],
                            [0, self.sigma_v * np.sqrt((3*Va)/self.Lv), np.sqrt((Va/self.Lv)**3), 0, 0],
                            [0, 0, 0, self.sigma_w * np.sqrt((3*Va)/self.Lv), np.sqrt((Va/self.Lw)**3)]])
        self._gust_state = np.zeros((5, 1))
        self._Ts = Ts
        
    def update(self, Va):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
         return np.concatenate(( self._steady_state, self._gust(Va) ))

    def _gust(self, Va):
        self._A = np.array([[-Va/self.Lu, 0, 0, 0, 0],
                            [0, -2*(Va/self.Lv), -(Va/self.Lv)**2, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, -2*(Va/self.Lw), -(Va/self.Lw)**2],
                            [0, 0, 0, 1, 0]])
        self._C = np.array([[self.sigma_u * np.sqrt((2*Va)/self.Lu), 0, 0, 0, 0],
                            [0, self.sigma_v * np.sqrt((3*Va)/self.Lv), np.sqrt((Va/self.Lv)**3), 0, 0],
                            [0, 0, 0, self.sigma_w * np.sqrt((3*Va)/self.Lv), np.sqrt((Va/self.Lw)**3)]])
        # calculate wind gust using Dryden model.  Gust is defined in the body frame
        w = np.random.randn(3, 1)  # zero mean unit variance Gaussian (white noise)
        # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
        self._gust_state += self._Ts * (self._A @ self._gust_state + self._B @ w)
        # output the current gust: y[k] = C x[k]
        return self._C @ self._gust_state

