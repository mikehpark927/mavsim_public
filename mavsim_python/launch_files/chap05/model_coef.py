import numpy as np
x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.969395, 0.000000, 1.236657, 0.998046, 0.000000, -0.062482, 0.000000, 0.000000, 0.000000, 0.000000]]).T
u_trim = np.array([[-0.123326, -0.007776, 0.001281, 0.399628]]).T
Va_trim = 25.000000
alpha_trim = 0.049486
theta_trim = -0.125046
a_phi1 = 22.628851
a_phi2 = 130.883678
a_theta1 = 5.294738
a_theta2 = 99.947422
a_theta3 = -36.112390
a_V1 = 0.261468
a_V2 = 4.141829
a_V3 = 9.660965
A_lon = np.array([
    [-0.186636, 0.496063, -1.207055, -9.733241, -0.000000],
    [-0.558558, -4.463598, 24.371686, 1.223485, -0.000000],
    [0.197761, -3.993003, -5.294738, 0.000000, -0.000000],
    [0.000000, 0.000000, 1.000071, 0.000000, -0.000000],
    [-0.124720, -0.992192, -0.000000, 24.619786, -0.000000]])
B_lon = np.array([
    [-0.139757, 4.141829],
    [-2.586110, 0.000000],
    [-36.112390, 0.000000],
    [0.000000, 0.000000],
    [-0.000000, -0.000000]])
A_lat = np.array([
    [-0.776773, 1.236657, -24.969395, 9.733241, 0.000000],
    [-3.866748, -22.628851, 10.905041, 0.000000, 0.000000],
    [0.783075, -0.115092, -1.227655, 0.000000, 0.000000],
    [0.000000, 0.999966, -0.125693, 0.000000, 0.000000],
    [0.000000, 0.000004, 1.007835, 0.000000, 0.000000]])
B_lat = np.array([
    [1.486172, 3.764969],
    [130.883678, -1.796374],
    [5.011735, -24.881341],
    [0.000000, 0.000000],
    [0.000000, 0.000000]])
Ts = 0.010000