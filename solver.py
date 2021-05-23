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
    x1, y1, theta1, v1, omega1 = s
    a1, alpha1 = u
    sin1, cos1 = jnp.sin(theta1), jnp.cos(theta1)
    ds = jnp.array([
        v1 * cos1,
        v1 * sin1,
        omega1,
        a1,
        alpha1,
    ])
    return ds


def scp_formpc(f, Q, R, Q_N, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, case, agent, hist):
    # Outer loop of scp.

    n = 5  # state dimension
    m = 2  # control dimension
    eps = 0.01  # termination threshold for scp

    # initialize reference rollout s_bar,u_bar
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])

    # Compute new state and control via scp.
    s, u = scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, case,
                                agent, hist)

    # run scp until u converges
    round = 0
    err = 0
    err = max(err, np.linalg.norm(u - u_bar, np.inf))
    while err > eps and round < 15:
        # print("round: %s, u update: %s" % (round, err))
        # print("u=", u_bar, "y=", s_bar[:, 1])
        round = round + 1
        s_bar = s
        u_bar = u
        s, u = scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, case,
                                    agent, hist)
        err = 0
        err = max(err, np.linalg.norm(u - u_bar, np.inf))

    return s, u


def scp_iteration_formpc(f, Q, R, Q_N, s_bar, u_bar, s_star, s0, N, dt, rho, UB, LB, aUB, vUB, omegaUB, case, agent,
                         hist):
    ###########################################################################
    # implement one iteration of scp

    for a in range(agent):
        s = cvx.Variable(s_bar.shape)
        u = cvx.Variable(u_bar.shape)
        cost_terms = []
        constraints = [s[0] == s0]
        A, B, c = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B, c = np.array(A), np.array(B), np.array(c)
        for k in range(N):
            # objective functions
            cost_terms.append(cvx.quad_form(s[k, 0:3] - s_star[0:3], Q))
            cost_terms.append(cvx.quad_form(u[k], R))
            constraints.append(s[k, 3] <= vUB)
            constraints.append(s[k, 3] >= 0)
            constraints.append(u[k, 0] <= aUB)
            constraints.append(u[k, 0] >= -aUB)
            constraints.append(u[k, 1] <= omegaUB)
            constraints.append(u[k, 1] >= -omegaUB)
            constraints.append(A[k] @ (s[k] - s_bar[k]) + B[k] @ (u[k] - u_bar[k]) + c[k] == s[k + 1])
        # Cost and Constraint for k = N
        cost_terms.append(cvx.quad_form(s[N, 0:3] - s_star[0:3], Q_N))

        objective = cvx.Minimize(cvx.sum(cost_terms))
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        s = s.value
        u = u.value

    ###########################################################################
    return s, u


def mpc():
    # import trajectory of agent 2
    file = open('vehicle2_traj.txt', 'r')
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
        if float(y_data[i]) < 1e-3 or i % 10 != 0:
            continue
        x_hist.append(float(x_data[i]))
        y_hist.append(float(y_data[i]))
        theta_hist.append(float(theta_data[i]) / 180 * np.pi)
    hist = np.vstack((np.array(x_hist), np.array(y_hist), np.array(theta_hist))).T
    hist = np.hstack((hist, np.zeros((len(hist), 1)), np.zeros((len(hist), 1))))

    # simulation parameters
    n = 5
    m = 2
    agent = 1

    dt = 0.01
    T = 4

    # cost function
    Qf = np.diag(np.array([1000, 1000, 1000]))  # , 0, 0])) # 1000.*np.eye(n)
    Q = np.diag(np.array([10, 10, 10]))  # , 0, 0]))
    R = np.diag(np.array([5, 5]))

    # specify dynamics
    f = jax.jit(intersection)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    # scp parameters
    UB, LB = 1., -1.
    aUB = 20.  # 1.
    vUB = 20  # 1.589
    omegaUB = np.pi  # np.pi/20 # ???
    rho = 1
    case = np.array(['straightright', 'upright'])  # np.array(['straight', 'upleft'])# 'straight' # 'right'

    t = np.arange(0., T, dt)
    N = t.size - 1
    traj_2 = np.zeros((N + 1, n))
    traj_2[0, :] = np.array([hist[0, 0], hist[0, 1], hist[0, 2], 0, 0])
    control_2 = np.zeros((N, m))

    kh = 1
    for mpc_t in range(N):
        if mpc_t == 0:
            start_state = np.array([hist[0, 0], hist[0, 1], hist[0, 2], 0, 0])
        else:
            start_state = traj_2[mpc_t - 1]
            if kh < len(hist) - 2 and kh != -1:
                kh += 1
            else:
                kh = -1
        # kh = -1
        goal_state = np.array([hist[kh, 0], hist[kh, 1], hist[kh, 2], 0, 0])
        # goal_state = np.array([hist[0, 0], hist[0, 1] + 10, hist[0, 2], 0, 0])
        print('start:', start_state)
        print('goal:', goal_state)
        # solve with scp
        # print('Computing SCP solution ... ', end='')
        s_mpc, u_mpc = scp_formpc(f_discrete, Q, R, Qf, goal_state, start_state, 30, dt, rho, UB, LB, aUB, vUB, omegaUB,
                                  case, agent, hist)
        '''
        if np.linalg.norm(s_mpc[-1, 0:2] - traj_1[mpc_t, 0:2]) <= 1:
            # print('conflict')
            kh -= 2
            mpc_t -= 1
            continue
            '''
        control_2[mpc_t] = u_mpc[0]
        # print('u=', u_mpc)
        # print('s=', s_mpc)
        traj_2[mpc_t + 1] = f_discrete(traj_2[mpc_t], control_2[mpc_t])
        print('traj_2:\n', traj_2[mpc_t + 1])
        print('-----------------step done!')
        # print(len(hist))
        # break

    print('done!')
