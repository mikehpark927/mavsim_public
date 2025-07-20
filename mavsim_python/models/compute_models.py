"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])
    
    ###### TODO ######
    # define transfer function constants

    # Lateral dynamics
    a_phi1 = -1/2 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_p * MAV.b / (2 * Va_trim)
    a_phi2 = 1/2 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    
    # Longitudinal dynamics
    a_theta1 = -1/(2*MAV.Jy) * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_q * MAV.c / (2 * Va_trim)
    a_theta2 = -1/(2*MAV.Jy) * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_alpha
    a_theta3 = 1/(2*MAV.Jy) * MAV.rho * Va_trim**2 * MAV.c * MAV.S_wing * MAV.C_m_delta_e
    
    # Compute transfer function coefficients using new propulsion model
    # Transfer function from throttle and pitch angle to airspeed
    deriv1 = dT_ddelta_t(mav, Va_trim, trim_input.throttle)
    deriv2 = dT_dVa(mav, Va_trim, trim_input.throttle)
    a_V1 = (MAV.rho * Va_trim * MAV.S_wing) / MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * trim_input.elevator)\
         - 1/MAV.mass * deriv2
    a_V2 = deriv1 / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3

"""
Computes the longitudinal and lateral state-space models from the full 12x12 model.

Args:
    mav: The MAV dynamics model.
    trim_state: Trimmed state (13x1).
    trim_input: Trimmed control input (MsgDelta).

Returns:
    A_lon, B_lon, A_lat, B_lat: State-space matrices for longitudinal and lateral dynamics.
"""
def compute_ss_model(mav, trim_state, trim_input):
    ##### TODO #####
    # Convert quaternion state to Euler state
    x_euler = euler_state(trim_state)

    # **1. Compute full-state Jacobians**
    A = df_dx(mav, x_euler, trim_input)  # 12x12
    B = df_du(mav, x_euler, trim_input)  # 12x4
    
    # **2. Extract Longitudinal System Matrices**
    # Longitudinal states: [u, w, q, theta, h] (5x1)
    lon_idx = [3, 5, 10, 7, 2]  # Indices of longitudinal states
    A_lon = A[np.ix_([3, 5, 10, 7, 2], [3, 5, 10, 7, 2])]  # Extract relevant rows & columns (5x5)
    B_lon = B[np.ix_(lon_idx, [0, 3])]  # Inputs: [delta_elevator, delta_throttle] (5x2)

    for i in range(0,5):
        A_lon[i,4] = -A_lon[i,4]
        A_lon[4,i] = -A_lon[4,i]
    for i in range(0,2):
        B_lon[4,i] = -B_lon[4,i] 

    # **3. Extract Lateral System Matrices**
    # Lateral states: [v, p, r, phi, psi] (5x1)
    lat_idx = [4, 9, 11, 6, 8]  # Indices of lateral states
    A_lat = A[np.ix_(lat_idx, lat_idx)]  # Extract relevant rows & columns (5x5)
    B_lat = B[np.ix_(lat_idx, [1, 2])]  # Inputs: [delta_aileron, delta_rudder] (5x2)

    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
    ##### TODO #####
    phi,theta,psi = quaternion_to_euler(x_quat[6:10])
    x_euler = np.zeros((12,1))
    x_euler[:6] = x_quat[:6]
    x_euler[6:9] = np.array([[phi, theta, psi]]).T
    x_euler[9:] = x_quat[10:]
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    ##### TODO #####
    x_quat = np.zeros((13,1))
    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)
    e = euler_to_quaternion(phi, theta, psi)
    x_quat[:6] = x_euler[:6]
    x_quat[6:10] = e
    x_quat[10:] = x_euler[9:]
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of quaternion_to_euler(quat) with respect to quat
    # compute partial quaternion_to_euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    ##### TODO #####
    f_euler_ = np.zeros((12,1))
    f = mav._f(x_quat, mav._forces_moments(delta))
    f_euler_ = euler_state(f)

    # correct the attitude states
    e = x_quat[6:10]
    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)
    dTheta_dquat = np.zeros((3,4))
    for j in range(0,4):
        tmp = np.zeros((4, 1))
        tmp[j][0] = .001
        e_eps = (e +tmp) / np.linalg.norm(e + tmp)
        phi_eps, theta_eps, psi_eps = quaternion_to_euler(e_eps)
        dTheta_dquat[0][j] = (theta_eps - theta) / 0.001
        dTheta_dquat[1][j] = (phi_eps - phi) / 0.001
        dTheta_dquat[2][j] = (psi_eps - psi) / 0.001

    f_euler_[6:9] = np.copy(dTheta_dquat @ f[6:10])

    return f_euler_


"""
Computes the Jacobian matrix df/dx using finite differences.

Args:
    mav: The MAV dynamics model.
    x_euler: The state vector in Euler representation (12x1).
    delta: Control input vector.

Returns:
    A: Jacobian matrix (12x12), df/dx.
"""
def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation

    ##### TODO #####
    n = 12  # Number of states
    A = np.zeros((n, n))  # Initialize Jacobian matrix

    # Compute f_euler at the nominal state
    f_euler0 = f_euler(mav, x_euler, delta)

    # Loop over each state variable to compute partial derivatives
    for i in range(n):
        x_perturbed = np.copy(x_euler)  # Copy original state
        x_perturbed[i] += eps  # Perturb state i
        f_euler1 = f_euler(mav, x_perturbed, delta)  # Compute f(x + eps)
        # Compute finite difference approximation of partial derivative
        df = (f_euler1.flatten() - f_euler0.flatten()) / eps
        A[:, i] = df

    return A

"""
Computes the Jacobian matrix df/du using finite differences.

Args:
    mav: The MAV dynamics model.
    x_euler: The state vector in Euler representation (12x1).
    delta: Control input vector (MsgDelta).

Returns:
    B: Jacobian matrix (12x4), df/du.
"""
def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # Small perturbation
    n_x = 12  # Number of states
    n_u = 4   # Number of control inputs
    B = np.zeros((n_x, n_u))  # Initialize Jacobian matrix

    # Compute f_euler at the nominal input
    f_euler0 = f_euler(mav, x_euler, delta)

    # Loop over each control input to compute partial derivatives
    for i, key in enumerate(["elevator", "aileron", "rudder", "throttle"]):
        # **1. Create a perturbed control input (MsgDelta object)**
        delta_perturbed = MsgDelta(
            elevator=delta.elevator,
            aileron=delta.aileron,
            rudder=delta.rudder,
            throttle=delta.throttle
        )
        setattr(delta_perturbed, key, getattr(delta, key) + eps)  # Perturb control input i

        # **2. Compute f(x, u+eps)**
        f_euler1 = f_euler(mav, x_euler, delta_perturbed)

        # **3. Compute finite difference approximation**
        B[:, i] = (f_euler1.flatten() - f_euler0.flatten()) / eps

    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01

    ##### TODO #####
    thrust1, torque1 = mav._motor_thrust_torque(Va, delta_t)
    thrust2, torque2 = mav._motor_thrust_torque(Va+ eps, delta_t)
    dT_dVa = (thrust2 - thrust1) / eps   
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01

    ##### TODO #####
    thrust1, torque1 = mav._motor_thrust_torque(Va, delta_t)
    thrust2, torque2 = mav._motor_thrust_torque(Va, delta_t + eps)
    dT_ddelta_t = (thrust2 - thrust1) / eps
    return dT_ddelta_t
