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
    
    v_interp = np.interp(np.squeeze(l_time), np.squeeze(odom_t), np.squeeze(vel_odom[0,:]))
    w_interp = np.interp(np.squeeze(l_time), np.squeeze(odom_t), np.squeeze(vel_odom[1,:]))

    t_prev = 0.0
    for meas_index in range(l_time.size):
        #stuff for plotting
        mu_hist.append(mu)
        dt = l_time.item(meas_index) - t_prev
        t_prev = l_time.item(meas_index)
        
        lm_ind = np.argwhere(np.isfinite(l_depth[:,meas_index]))
        r = l_depth[lm_ind, meas_index]
        phi = l_bearing[lm_ind, meas_index]
        zt = np.hstack((r, phi)).T

        mu, Sigma, K = ukf.update(mu, Sigma, zt, lm_ind, v_interp[meas_index], w_interp[meas_index], dt)

    fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True)
    mu_hist = np.array(mu_hist).T
    odom_t = odom_t.flatten()
    ax1[0].plot(t_truth, x_truth, label="Truth")
    ax1[0].plot(l_time, mu_hist[0,:], label="Est")
    ax1[0].set_ylabel("x (m)")
    ax1[0].legend()
    ax1[1].plot(t_truth, y_truth, label="Truth")
    ax1[1].plot(l_time, mu_hist[1,:], label="Est")
    ax1[1].set_ylabel("y (m)")
    ax1[1].legend()
    ax1[2].plot(t_truth, th_truth, label="Truth")
    ax1[2].plot(l_time, mu_hist[2,:], label="Est")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].set_ylabel("$\psi$ (rad)")
    ax1[2].legend()
    ax1[0].set_title("Estimate vs Truth")

    plt.figure(2)
    plt.plot(x_truth, y_truth, 'b')
    plt.plot(mu_hist[0,:], mu_hist[1,:], 'r')
    plt.plot(pos_odom_se2[0,:], pos_odom_se2[1,:])

    plt.show()
    print("Finished")
    plt.close()
