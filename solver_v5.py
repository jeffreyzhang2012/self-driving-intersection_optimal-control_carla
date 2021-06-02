import numpy as np
from scipy.integrate import odeint
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cvx


@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, x, u):
    A = jax.jacfwd(f, argnums=0)(x, u)
    B = jax.jacfwd(f, argnums=1)(x, u)
    c = f(x, u)
    return A, B, c


def intersection(s, u):
    """Compute the state derivative."""
    x1, y1, theta1 = s
    v1, omega1 = u
    sin1, cos1 = jnp.sin(theta1), jnp.cos(theta1)
    ds = jnp.array([
        v1 * cos1,
        v1 * sin1,
        omega1
    ])
    return ds


def scp_formpc(f, Q, R, Q_N, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, Q_d, agent, hist, v0, w0):
    """Outer loop of scp"""
    n = 3  # state dimension, s = [x, y, theta, linear vel, angular vel]
    m = 2  # control dimension, u = [linear acc, angular acc]
    eps = 0.01  # termination threshold for scp

    # initialize reference rollout s_bar,u_bar
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0, :3] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])

    # Compute new state and control via scp.
    s, u, c = scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, Q_d,
                                   agent, hist, v0, w0)
    if u is None:
        if c is None:
            return None, None, 0
        else:
            return None, None, c

    # run scp until u converges
    round = 0
    err = 0
    err = max(err, np.linalg.norm(u - u_bar, np.inf))
    # terminate if (1) converge, or (2) reaches maximum iteration
    while err > eps and round < 15:
        round = round + 1
        s_bar = s
        u_bar = u
        s, u, c = scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB,
                                       Q_d, agent, hist, v0, w0)
        err = 0
        err = max(err, np.linalg.norm(u - u_bar, np.inf))

    return s, u, c


def scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, Q_d, agent,
                         hist, v0, w0):
    """implement one iteration of scp"""
    s = cvx.Variable(s_bar.shape)
    u = cvx.Variable(u_bar.shape)
    cost_terms = []
    constraints = [s[0, :3] == s0]
    constraints.append(u[0, 0] == v0)
    constraints.append(u[0, 1] == w0)
    A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    for k in range(N):
        # objective functions
        cost_terms.append(cvx.quad_form(s[k, :3] - s_star[:3], Q))
        if (s_bar[k, :2]).all != 0:
            # dk = np.linalg.norm(hist + np.array([0, 5.]) - s_bar[k, :2])
            # unit_vec = (hist + np.array([0, 5.]) - s_bar[k, :2]) / dk
            dk = np.linalg.norm(hist - s_bar[k, :2])
            unit_vec = (hist - s_bar[k, :2]) / dk
            # constraints.append(cvx.sum(unit_vec @ (hist + np.array([0, 3.]) - s[N, :2])) >= 8)
            # cost_terms.append(1000 - 50 * cvx.sum(unit_vec @ (hist + np.array([0, 1.]) - s[k, :2])))
            s_push = hist - unit_vec * 20
            cost_terms.append(cvx.quad_form(s[k, :2] - s_push, Q_d))
            # cost_terms.append(cvx.quad_form(s[k, :2] - (hist[:2] + np.array([0, -3])), Q_d))
        cost_terms.append(cvx.quad_form(u[k], R))
        if k >= 1:
            cost_terms.append(cvx.norm(u[k,0]-u[k-1,0])*100)
            # constraints.append(u[k, 0] -u[k-1, 0]<= aUB/dt)
            # constraints.append(u[k, 0] -u[k-1, 0]>= -aUB/dt)
        # constraints.append(u[k, 0] <= aUB)
        # constraints.append(u[k, 0] >= -aUB)
        constraints.append(u[k, 0] <= vUB)
        constraints.append(u[k, 0] >= -vUB)
        constraints.append(u[k, 1] <= omegaUB)
        constraints.append(u[k, 1] >= -omegaUB)
        # constraints.append(s[k, 3] >= 0)
        # constraints.append(s[k, 3] <= vUB)
        constraints.append(A[k] @ (s[k] - s_bar[k]) + B[k] @ (u[k] - u_bar[k]) + c[k] == s[k + 1])
    # Cost and Constraint for k = N
    cost_terms.append(cvx.quad_form(s[N, 0:3] - s_star[0:3], Q_N))
    # if np.linalg.norm(hist - s_bar[N, :2]) > 0:
    if (s_bar[N, :2]).all != 0:
        # dk = np.linalg.norm(hist + np.array([0, 5.]) - s_bar[N, :2])
        # unit_vec = (hist + np.array([0, 5.]) - s_bar[N, :2]) / dk
        dk = np.linalg.norm(hist - s_bar[N, :2])
        unit_vec = (hist - s_bar[N, :2]) / dk
        # constraints.append(cvx.sum(unit_vec @ (hist + np.array([0, 3.]) - s[N, :2])) >= 8)
        # cost_terms.append(1000 - 50 * cvx.sum(unit_vec @ (hist + np.array([0, 3.]) - s[N, :2])))
        s_push = hist - unit_vec * 20
        # print('push=', s_push)
        cost_terms.append(cvx.quad_form(s[N, :2] - s_push, Q_d))
    # if np.linalg.norm(hist - s_bar[N, :2]) > 0:
    #     print('unit_vec=', unit_vec @ (hist - s[N, :2]))
    # cost_terms.append(10 - cvx.norm(s[N, 0:2] - hist[0:2]))
    # if hist[1] <= 254:
    #     cost_terms.append(cvx.quad_form(s[N, :2] - (hist[:2] + jnp.array([-5, 0])), Q_d))
    # else:
    # cost_terms.append(cvx.quad_form(s[N, :2] - (hist[:2] + jnp.array([0, -3])), Q_d))

    objective = cvx.Minimize(cvx.sum(cost_terms))
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    s = s.value
    # if np.linalg.norm(hist - s_bar[N, :2]) > 0:
    #     print(s[N], '\ndir=', unit_vec @ (hist - s[N, :2]))
    u = u.value

    return s, u, cvx.sum(cost_terms).value


def get_hist(filename):
    """
    Import trajectory for the ego agent to follow
    Output: np.array in shape (len x n), where len is how many time steps the trajectory has,
            and n is dimension of state variable (n=3 now)
    """
    file = open(filename, 'r')
    x_hist = []
    y_hist = []
    theta_hist = []
    idx = 0
    x_data, y_data, theta_data = None, None, None
    for z in file:
        if idx == 0:
            x_data = z.split(',')
        if idx == 1:
            y_data = z.split(',')
        if idx == 2:
            theta_data = z.split(',')
        idx += 1
    for i in range(len(x_data)):
        if float(y_data[i]) < 1e-3:
            continue
        x_hist.append(float(x_data[i]))
        y_hist.append(float(y_data[i]))
        theta_hist.append(float(theta_data[i]) / 180 * np.pi)
    # hist is now [[x], [y], [theta]]
    hist = np.vstack((np.array(x_hist), np.array(y_hist), np.array(theta_hist))).T
    # append two columns of zeros to match n=5
    # hist = np.hstack((hist, np.zeros((len(hist), 2))))
    return hist


def mpc(hist=None, traj_11=None, traj_10=None, traj_2=None, tick=None, goal=None, dt = 0.1, horizon= 20, v0 = 0, w0 = 0):
    # simulation parameters
    n = 3
    m = 2
    agent = 1
    # print(v0,w0,"!!!!!!!!!!!")
    # dt = 0.1  # 0.01
    # horizon = 20  # 30

    # cost function
    Qf = np.diag(np.array([100, 100, 100]))  # , 0, 0])) # 1000.*np.eye(n)
    Q = np.diag(np.array([10, 10, 10]))  # , 0, 0]))
    Q_d = np.diag(np.array([7, 7]))
    R = np.diag(np.array([1, 1]))

    # specify dynamics
    f = jax.jit(intersection)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    # scp parameters
    UB, LB = None, None  # variables labeled as None are rebundant and will be removed later
    aUB = 4.  # acceleration upper bound
    vUB = 15.  # velocity upper bound
    omegaUB = np.pi / 17  # angular acceleration upper bound
    rho = None  # trust region constraint, currently not used to guarantee solvability

    '''MPC Process: In total time step length N, perform SCP iteration with some time steps (e.g. 30)'''
    start_state = traj_2
    # setting the goal state according to specified reference trajectory
    goal = np.array([goal[1], goal[0], 0.])
    dist = np.linalg.norm(goal[:2] - traj_2[:2])
    goal_dir = (goal[:2] - traj_2[:2]) / dist
    goal_angle = np.arctan(goal_dir[1] / goal_dir[0])
    goal_state = np.array([traj_2[0] + goal_dir[0] * 5, traj_2[1] + goal_dir[1] * 5, max(traj_2[2] - goal_angle, 0)])
    # solve with scp
    # predict traj_12
    if traj_10 is not None:
        traj_1_d = np.linalg.norm(traj_11 - traj_10)
        # print(traj_11, traj_10)
        if traj_1_d > 0:
            traj_1_dir = (traj_11 - traj_10) / np.linalg.norm(traj_11 - traj_10)
            traj_1 = traj_11 + traj_1_dir * 10
        else:
            traj_1 = traj_11
    else:
        traj_1 = traj_11
    # print('start:', start_state, 'goal:', goal_state, 'traj_1:', traj_11, 'traj_target:', traj_1)
    s_mpc, u_mpc, c_mpc = scp_formpc(f_discrete, Q, R, Qf, goal_state, start_state,
                                     horizon, dt, rho, UB, LB, aUB, vUB, omegaUB, Q_d, agent, traj_1, v0, w0)
    sn_mpc = s_mpc[-1, :2]
    '''
    if s_mpc is not None:
        if np.linalg.norm(s_mpc[-1, 0:2] - traj_1[0:2]) <= 10 and traj_1[0] - s_mpc[-1, 1] <= 3:
            kh -= 1
            goal_state = np.array([hist[kh, 0], hist[kh, 1], hist[kh, 2]])
            s_mpc, u_mpc = scp_formpc(f_discrete, Q, R, Qf, goal_state, start_state,
                                      5, dt, rho, UB, LB, aUB, vUB, omegaUB, Q_d, agent, hist)
                                      '''
    if u_mpc is None:
        '''
        u1, u2 = None, None
        if traj_2[mpc_t, 3] > vUB or control_2[mpc_t - 1, 0] > aUB:
            u1 = -10.
        elif traj_2[mpc_t, 3] < 0 or control_2[mpc_t - 1, 0] < -aUB:
            u1 = 10.
        else:
            u1 = 0.
        if control_2[mpc_t - 1, 1] > omegaUB:
            u2 = -0.1
        elif control_2[mpc_t - 1, 1] < -omegaUB:
            u2 = 0.1
        else:
            u2 = 0.
        '''
        s_mpc = np.array([[0., 0., 0.]])
        # u_mpc = np.array([[-2., 0.]])
        u_mpc = np.zeros((horizon, m))
        sn_mpc = np.array([0, 0])
    # print(np.linalg.norm(start_state[:2] - traj_1[:2]))
    # print(c_mpc)
    # return [s_mpc[0, 0], s_mpc[0, 1], s_mpc[0, 2]], sn_mpc, [u_mpc[0, 0], u_mpc[0, 1]], c_mpc, traj_1
    return [s_mpc[0, 0], s_mpc[0, 1], s_mpc[0, 2]], sn_mpc, u_mpc, c_mpc, traj_1


    # break
