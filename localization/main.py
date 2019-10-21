import numpy as np
import matplotlib.pyplot as plt
import car_params as params
import scipy.io as sio
from ukf import UKF
from ukf import unwrap
from extractdata import *


if __name__ == "__main__":
    ukf = UKF()

    x_hist = truth_data['x_truth']
    y_hist = truth_data['y_truth']
    theta_hist = truth_data['th_truth']
    t_truth = truth_data['t_truth']
    mu_hist = []

    x0 = params.x0
    y0 = params.y0
    phi0 = params.theta0
    mu = np.array([x0, y0, phi0])
    Sigma = np.eye(3)

    t_prev = 0.0
    meas_index = 0
    for i in range(odom_t.size):
        #stuff for plotting
        mu_hist.append(mu)
        dt = odom_t.item(i) - t_prev
        t_prev = odom_t.item(i)

        #find which measurements to use
        while l_time.item(meas_index) < odom_t.item(i):
            meas_index += 1
            if meas_index >= l_time.size:
                break
        meas_index -= 1
        
        lm_ind = np.argwhere(np.isfinite(l_depth[:,meas_index]))
        r = l_depth[lm_ind, meas_index]
        phi = l_bearing[lm_ind, meas_index]
        zt = np.vstack((r, phi))

        mu, Sigma, K = ukf.update(mu, Sigma, zt, vel_odom[0,i], vel_odom[1,i], dt)

    # fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True)
    # x_hist = np.array(x_hist).T
    # mu_hist = np.array(mu_hist).T
    # ax1[0].plot(t, x_hist[0,:], label="Truth")
    # ax1[0].plot(t, mu_hist[0,:], label="Est")
    # ax1[0].set_ylabel("x (m)")
    # ax1[0].legend()
    # ax1[1].plot(t, x_hist[1,:], label="Truth")
    # ax1[1].plot(t, mu_hist[1,:], label="Est")
    # ax1[1].set_ylabel("y (m)")
    # ax1[1].legend()
    # ax1[2].plot(t, x_hist[2,:], label="Truth")
    # ax1[2].plot(t, mu_hist[2,:], label="Est")
    # ax1[2].set_xlabel("Time (s)")
    # ax1[2].set_ylabel("$\psi$ (rad)")
    # ax1[2].legend()
    # ax1[0].set_title("Estimate vs Truth")

    # fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
    # err_hist = np.array(err_hist).T
    # x_err_bnd = np.sqrt(np.array(x_covar_hist)) * 2
    # y_err_bnd = np.sqrt(np.array(y_covar_hist)) * 2
    # psi_err_bnd = np.sqrt(np.array(psi_covar_hist)) * 2
    # ax2[0].plot(t, err_hist[0,:], label="Err")
    # ax2[0].plot(t, x_err_bnd, 'r', label="2 $\sigma$")
    # ax2[0].plot(t, -x_err_bnd, 'r')
    # ax2[0].set_ylabel("Err (m)")
    # ax2[0].legend()
    # ax2[1].plot(t, err_hist[1,:], label="Err")
    # ax2[1].plot(t, y_err_bnd, 'r', label="2 $\sigma$")
    # ax2[1].plot(t, -y_err_bnd, 'r')
    # ax2[1].set_ylabel("Err (m)")
    # ax2[1].legend()
    # ax2[2].plot(t, err_hist[2,:], label="Err")
    # ax2[2].plot(t, psi_err_bnd, 'r', label="2 $\sigma$")
    # ax2[2].plot(t, -psi_err_bnd, 'r')
    # ax2[2].set_ylabel("Err (m)")
    # ax2[2].set_xlabel("Time (s)")
    # ax2[2].legend()
    # ax2[0].set_title("Error vs Time")

    # plt.figure(4)
    # K_hist = np.array(K_hist)
    # plt.plot(t, K_hist[:,0,0])
    # plt.plot(t, K_hist[:,1,0])
    # plt.plot(t, K_hist[:,2,0])
    # plt.plot(t, K_hist[:,0,1])
    # plt.plot(t, K_hist[:,1,1])
    # plt.plot(t, K_hist[:,2,1])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Kalman Gain")
    # plt.title("Kalman Gain vs Time")

    plt.show()
    print("Finished")
    plt.close()
