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
        "C:\\Users\\twilkeni\\AppData\\Local"
        + "\\anaconda3\\envs\\pytorch-env\\site-packages",
    )
import gpytorch


def GPRegression(conn, meas, meas_new, test_x, model):
    # Perform Regression on the new data

    # columns of the mean_table:
    # time, meas_mean, processed

    # pull the un-processed data from the mean_table
    cur = conn.cursor()
    cur.execute(
        "SELECT time, meas_mean FROM mean_table WHERE processed = FALSE"
    )  # noqa
    rows = cur.fetchall()
    meas_unprocessed = np.asarray(rows).T
    if len(meas_unprocessed) == 0:
        train_x = meas_new[0, :]
        train_y = meas_new[1, :]
    else:
        meas_both = np.hstack((meas_unprocessed, meas_new))
        train_x = meas_both[0, :]
        train_y = meas_both[1, :]

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)

    # update the model training data
    model.set_train_data(train_x, train_y)

    # mark the un-processed data as processed now
    cur.execute(
        "UPDATE mean_table SET processed = TRUE WHERE processed = FALSE"  # noqa
    )  # noqa
    conn.commit()

    # run the GPModel prediction by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_mean = model.likelihood(model(test_x))
        obs_mean = model.likelihood(model(train_x))

    # combine the actual mean data from the GPModel predictive
    # posteriors with the respective x values
    pred_new = np.vstack(
        (test_x.numpy().reshape(1, -1), pred_mean.mean.numpy().reshape(1, -1))
    )
    gp_mean_new = np.vstack(
        (train_x.numpy().reshape(1, -1), obs_mean.mean.numpy().reshape(1, -1))
    )

    # add to mean_table with meas_new data, flag as un-processed
    processed_flag = np.zeros(meas_new.shape[1], dtype=bool)  # row of False
    data = np.vstack((meas_new, processed_flag)).T  # transpose after stacking
    data = [tuple(row) for row in data]  # convert to tuple for SQL
    cur.executemany(
        " ".join(
            [
                "INSERT INTO mean_table",
                "(time, meas_mean, processed)",
                "VALUES (%s, %s, %s)",
            ]
        ),
        data,
    )
    conn.commit()
    cur.close()

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
    cur.close()

    # calculate regression
    # based on the measured data, and the GPModel, compute expected mean
    # for the new times
    test_x = np.linspace(meas[0, -1], meas[0, -1] + 5, 25)
    gp_mean_new, pred_new = GPRegression(conn, meas, meas_new, test_x, model)

    # plot *measured* mean
    line_gp_mean = lines[1]
    gp_mean_old = np.asarray(line_gp_mean.get_data())  # outputs (2, N) array
    # compare size of gp_mean_new with meas_new
    gp_mean_replace_sz = gp_mean_new.shape[1] - meas_new.shape[1]
    # remove gp_mean_old data that is being replaced by new estimates
    # given the updated set of observations
    if gp_mean_replace_sz > 0:
        gp_mean_old = gp_mean_old[:, 0:-gp_mean_replace_sz]

    gp_mean = np.hstack((gp_mean_old, gp_mean_new))
    line_gp_mean.set_data(gp_mean[0, :] - t0, gp_mean[1, :])

    # plot *predictive* mean
    line_pred_mean = lines[2]
    line_pred_mean.set_data(pred_new[0, :] - t0, pred_new[1, :])

    # update the plot
    plt.xlim([0, 100])
    plt.ylim([-0.1, 0.1])
    plt.draw()

    return [line_meas, line_gp_mean, line_pred_mean]


def scheduled_fetch(model):
    conn = psycopg2.connect("dbname=test user=postgres")
    plt.ion()
    fig, ax = plt.subplots()
    t0 = time.time()
    gp_update_interval = 20  # seconds
    (line_meas,) = ax.plot(t0, 0, "k*")  # init measured data scatter
    (line_gp_mean,) = ax.plot(t0, 0, "b")  # init GP mean line
    (line_pred_mean,) = ax.plot(t0, 0, "g")  # init prediction line
    lines = [line_meas, line_gp_mean, line_pred_mean]
    i = 1
    try:
        while True:
            lines = fetch_and_plot_data(conn, lines, model)
            if (time.time() - t0) / gp_update_interval > i:
                i += 1
                # check to see if new parameters are available
                cur = conn.cursor()
                cur.execute("SELECT gp_update_avail FROM gp_table")
                gp_update_avail = cur.fetchone()
                if gp_update_avail:
                    # update model parameters if update available
                    model.load_state_dict(state_dict)
                    # set the flag to gp update un-available
                    cur.execute("UPDATE gp_table SET gp_update_avail = FALSE")
                cur.close()
            time.sleep(2)  # Fetch data every 2 seconds
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        conn.close()
        plt.close()


# Define ExactGPModel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# load model from off-line training data
train_x = torch.load("data/train_x.pth")
train_y = torch.load("data/train_y.pth")
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
    noise_prior=gpytorch.priors.NormalPrior(0, 0.1),
)
state_dict = torch.load("data/model_state_test.pth")
model = ExactGPModel(train_x, train_y, likelihood)
model.load_state_dict(state_dict)

scheduled_fetch(model)

# TODO: create a function to delete all of the data from SQL server if desired

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
