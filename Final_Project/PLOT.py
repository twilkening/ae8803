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

# from IPython.display import display, clear_output
import logging
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
from config import load_config


logger = logging.getLogger(__name__)
FORMAT = (
    "[%(filename)s:%(lineno)s - %(funcName)20s() ] "
    + "%(asctime)s : %(message)s"  # noqa
)  # noqa
logging.basicConfig(
    filename="logs/plot.log",
    filemode="w",
    format=FORMAT,
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

train_num = 400
torch.set_default_dtype(torch.float64)


def GPRegression(conn, meas, meas_new, test_x, model, likelihood):
    # Perform Regression on the new data

    # get up to the last 'train_num' measurements/observations
    num_new = meas_new.shape[1]
    if meas.shape[1] <= (train_num + num_new):
        train_x = meas[0, :]
        train_y = meas[1, :]
    else:
        train_x = meas[0, -(train_num + num_new) :]  # noqa
        train_y = meas[1, -(train_num + num_new) :]  # noqa

    train_x = torch.from_numpy(train_x.astype(np.float64))
    train_y = torch.from_numpy(train_y.astype(np.float64))
    test_x = torch.from_numpy(test_x.astype(np.float64))
    # normalize the target values to [-1 1]
    max_y = max(abs(train_y))
    train_y = train_y / max_y
    scale = max_y.numpy()

    # update the model training data
    model.set_train_data(train_x, train_y, strict=False)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # run the GPModel prediction by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # y* predictions (likelihood noise included)
        pred_mean = model.likelihood(model(test_x))
        # f* predictions (noise-free)
        obs_mean = model(train_x)

    # combine the actual mean data from the GPModel predictive
    # posteriors with the respective x values
    pred_new = np.vstack(
        (
            test_x.numpy().reshape(1, -1),
            pred_mean.mean.numpy().reshape(1, -1) * scale,  # noqa
        )
    )
    gp_mean_new = np.vstack(
        (
            train_x.numpy().reshape(1, -1),
            obs_mean.mean.numpy().reshape(1, -1) * scale,  # noqa
        )
    )

    # add to mean_table the (possibly re-processed) obs_mean data, flag
    # as un-processed
    mean_new = [
        (
            gp_mean_new[0, -num_new + i].astype(np.double),
            gp_mean_new[1, -num_new + i].astype(np.double),
            False,
        )
        for i in range(num_new)  # noqa
    ]  # noqa
    with conn.cursor() as cur:
        cur.executemany(
            " ".join(
                [
                    "INSERT INTO mean_table",
                    "(time, meas_mean, processed)",
                    "VALUES (%s, %s, %s)",
                ]
            ),
            mean_new,
        )
        conn.commit()

    return gp_mean_new, pred_new


def fetch_and_plot_data(conn, lines, model, likelihood, tstart, fig, ax):
    cur = conn.cursor()
    cur.execute(
        "SELECT time, measured_value FROM daq_table WHERE processed = FALSE"
    )  # noqa
    rows = cur.fetchall()
    # convert to array and transpose to (2, N) so can stack with old data
    meas_new = np.asarray(rows).T
    if len(meas_new) != 0:
        # mark the measured data as processed (first, so no delay)
        cur.execute(
            "UPDATE daq_table SET processed = TRUE WHERE processed = FALSE"  # noqa
        )  # noqa
        conn.commit()
        cur.close()

        # sort new measured data by time
        meas_new = meas_new[:, meas_new[0, :].argsort()]
        meas_new[0, :] = meas_new[0, :] - tstart  # shift x

        # get old data
        line_meas = lines[0]
        meas_old = np.asarray(line_meas.get_data())  # outputs (2, N) array
        # sort old measured data by time
        meas_old = meas_old[:, meas_old[0, :].argsort()]
        # stack old and new together
        meas = np.hstack((meas_old, meas_new))
        # plot new data alongside the old data
        line_meas.set_data(meas[0, :], meas[1, :])

        # calculate regression
        # based on the measured data, and the GPModel, compute expected mean
        # for the new times
        test_x = np.linspace(meas[0, -1], meas[0, -1] + 10, 50)
        gp_mean_new, pred_new = GPRegression(
            conn, meas, meas_new, test_x, model, likelihood
        )

        # plot *measured* mean
        line_gp_mean = lines[1]
        # get old gp_mean plot data, as (2, N) array
        gp_mean_old = np.asarray(line_gp_mean.get_data())
        # remove gp_mean_old data that is being replaced by new estimates
        # given the updated set of observations
        if gp_mean_old.shape[1] <= train_num:
            logger.debug("replacing the entire old mean line")
            gp_mean = gp_mean_new
        else:
            logger.debug(
                f"replacing only the last {train_num} entries to mean line"
            )  # noqa
            logger.debug(f"gp_mean_new length: {gp_mean_new.shape[1]}")
            gp_mean = np.hstack((gp_mean_old[:, :-train_num], gp_mean_new))

        line_gp_mean.set_data(gp_mean[0, :], gp_mean[1, :])

        # plot *predictive* mean
        line_pred_mean = lines[2]
        line_pred_mean.set_data(pred_new[0, :], pred_new[1, :])

        # update the plot
        if __name__ == "__main__":
            plt.xlim([0, 200])
            # plt.ylim([-1.25 * min(meas[1, :]), 1.25 * max(meas[1,
            # :])])
            plt.ylim([-0.02, 0.02])
            plt.draw()
        else:
            ax.draw_artist(ax.patch)
            ax.draw_artist(line_meas)
            ax.draw_artist(line_gp_mean)
            ax.draw_artist(line_pred_mean)
            plt.xlim([0, 200])
            plt.ylim([-0.01, 0.01])
            fig.canvas.draw()
            fig.canvas.flush_events()

        new_lines = [line_meas, line_gp_mean, line_pred_mean]
    else:
        logger.debug("no new data")
        new_lines = lines

    return new_lines


def scheduled_fetch(model, likelihood):
    if __name__ == "__main__":
        plt.ion()

    logger.debug(f"scheduled fetch started at {time.time()}")
    fig, ax = plt.subplots()
    t0 = time.time()
    gp_update_interval = 30  # seconds
    (line_meas,) = ax.plot(0, 0, "k*")  # init measured data scatter
    (line_gp_mean,) = ax.plot(0, 0, "r")  # init GP mean line
    (line_pred_mean,) = ax.plot(0, 0, "g--")  # init prediction line
    plt.title("GP Mean Prediction of Pressure Data")
    plt.xlabel("time [s]")
    plt.ylabel(r"Pressure; $\Delta$P [kPa]")
    lines = [line_meas, line_gp_mean, line_pred_mean]
    i = 1
    try:
        config = load_config()
        conn = psycopg2.connect(**config)
        while True:
            # fetch and plot data:
            lines = fetch_and_plot_data(
                conn, lines, model, likelihood, t0, fig, ax
            )  # noqa

            # update the GP Model every gp_update_interval:
            if (time.time() - t0) / gp_update_interval > i:
                i += 1
                # check to see if new parameters are available
                cur = conn.cursor()
                cur.execute(
                    "SELECT gp_update_avail FROM gp_table WHERE id = 1;"
                )  # noqa
                gp_update_avail = cur.fetchone()
                logger.debug(f"gp_update_avail: {gp_update_avail}")
                if gp_update_avail[0]:
                    # update model parameters if update available
                    state_dict = torch.load("data/model_state_update.pth")
                    model.load_state_dict(state_dict)
                    # set the flag to gp update un-available
                    cur.execute(
                        "UPDATE gp_table SET gp_update_avail = FALSE"
                        + " WHERE id = 1;"  # noqa
                    )
                conn.commit()
                cur.close()

            plt.pause(5)  # only fetch data every n seconds

    except KeyboardInterrupt:
        logger.debug("Stopped by user.")
    finally:
        conn.close()


if __name__ == "__main__":
    # Define ExactGPModel
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()  # noqa
            )  # noqa

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
    state_dict = torch.load("data/model_state_start.pth")
    model = ExactGPModel(train_x, train_y, likelihood)
    model.load_state_dict(state_dict)

    scheduled_fetch(model, likelihood)

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
