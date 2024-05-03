# -----------------------------------------------------------------------------
# PLOT.py
#
# Python script to plot collected data from the PostgreSQL database.
#
# ------------------------------------------------------------------------
#
# Written by Theodore Wilkening with assistance from ChatGPT4, May 2024

import psycopg2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

if sys.platform == "win32":
    sys.path.insert(
        0,
        "C:\\Users\\twilkeni\\AppData\\Local\\anaconda3\\envs\\pytorch-env\\site-packages",
    )
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def GPRegression(meas, meas_new, test_x, model):
    # Perform Regression on the new data
    # NOTE: based on size of the GPRegression kernel, we will need to
    # delete some of the old prediction data and fill it in with the
    # newly processed prediction of the mean now that we have samples

    # columns:
    # time, meas_mean, processed

    # pull the un-processed data from the mean_table
    # if there isn't any,
    #   set train_x & train_y = meas_new only
    # else
    # set train_x & train_y = [un-processed, meas_new]
    # then update the model training data
    model.set_train_data(train_x, train_y)

    # mark the un-processed data as processed now

    # set test_x = test_x

    # run the GPModel prediction by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_new = model.likelihood(model(test_x))
        gp_mean_new = model.likelihood(model(train_x))

    # update mean_table with meas_new data, flag as un-processed

    return gp_mean_new, pred_new


def fetch_and_plot_data(conn, lines, model):
    cur = conn.cursor()
    cur.execute(
        "SELECT time, measured_value FROM daq_table WHERE processed = FALSE"
    )  # noqa
    rows = cur.fetchall()
    # convert to array and transpose to (2, N) so can stack with old data
    meas_new = np.asarray(rows).T
    # plot new data alongside the old data
    line_meas = lines[0]
    meas_old = np.asarray(line_meas.get_data())  # outputs (2, N) array
    meas = np.hstack((meas_old, meas_new))
    t0 = meas[0, 0]
    line_meas.set_data(meas[0, :] - t0, meas[1, :])
    # mark the measured data as processed:
    cur.execute(
        "UPDATE daq_table SET processed = TRUE WHERE processed = FALSE"  # noqa
    )  # noqa
    conn.commit()

    # calculate regression
    # based on the measured data, and the GPModel, compute expected mean
    # for the new times
    test_x = np.linspace(meas[0, -1], meas[0, -1] + 5, 25)
    gp_mean_new, pred_new = GPRegression(meas, meas_new, test_x, model)

    # plot *measured* mean
    line_gp_mean = lines[1]
    gp_mean_old = np.asarray(line_gp_mean.get_data())  # outputs (2, N) array
    gp_mean = np.hstack((gp_mean_old, gp_mean_new))
    line_gp_mean.set_data(gp_mean[0, :] - t0, gp_mean[1, :])

    # plot *predictive* mean
    line_pred_mean = lines[2]
    line_pred_mean.set_data(pred_new[0, :] - t0, pred_new[1, :])

    cur.close()
    plt.draw()

    return [line_meas, line_gp_mean, line_pred_mean]


def scheduled_fetch(model):
    conn = psycopg2.connect("dbname=test user=postgres")
    plt.ion()
    fig, ax = plt.subplots()
    t0 = time.time()
    (line_meas,) = ax.plot(t0, 0, "k*")  # init measured data scatter
    (line_gp_mean,) = ax.plot(t0, 0, "b")  # init GP mean line
    (line_pred_mean,) = ax.plot(t0, 0, "g")  # init prediction line
    lines = [line_meas, line_gp_mean, line_pred_mean]
    try:
        while True:
            lines = fetch_and_plot_data(conn, lines, model)
            time.sleep(2)  # Fetch data every 2 seconds
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        conn.close()
        plt.close()


# load model from off-line training data, will have optimized
# hyperparameters and also comes with training data on it. TODO
# model = load model

scheduled_fetch(model)

# # plotting notes:
# line, = ax.plot(x, y, 'r-')  # The comma is important to unpack the
# list of lines returned by plot
# # Update the data of the line:
# line.set_ydata(new_y)
# can also use line.set_data(*args) for both (x,y) data where
#   *args: (2, N) array or two 1D arrays)
#
# plt.ion()
# Enable interactive mode, which shows / updates the figure after every
# plotting command, so that calling show() is not necessary.
# (don't need to use plt.pause())

# SO. we'll need to track the lines and update them.
