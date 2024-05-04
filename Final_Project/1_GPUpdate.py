# -----------------------------------------------------------------------------
# 1_GPUpdate.py
#
# Python script to plot collected data from the PostgreSQL database.
#
# ------------------------------------------------------------------------
#
# Written by Theodore Wilkening with assistance from ChatGPT4, May 2024

import psycopg2
import time
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

# Connect to the PostgreSQL database
conn = psycopg2.connect("dbname=test user=postgres")


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


# initial loading of model from off-line training data
train_x = torch.load("data/train_x.pth")
train_y = torch.load("data/train_y.pth")
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
    noise_prior=gpytorch.priors.NormalPrior(0, 0.1),
)
state_dict = torch.load("data/model_state_test.pth")
model = ExactGPModel(train_x, train_y, likelihood)
model.load_state_dict(state_dict)


def GPUpdate(model, likelihood, conn):
    cur = conn.cursor()
    # collect last 100 observations from daq_table
    cur.execute("SELECT * FROM daq_table ORDER BY time DESC LIMIT 100")
    rows = cur.fetchall()
    # transpose to (3, N) array, since it includes processed flag
    obs = np.asarray(rows).T
    train_x = obs[0, :]
    train_y = obs[1, :]

    # train the model for 5 iterations
    training_iter = 5
    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   th1: %.3f  th2: %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.outputscale.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()

    # save the new model state_dict
    torch.save(model.state_dict(), "data/model_state_test.pth")

    # set the flag for new gp hyperparameters available
    cur.execute("UPDATE gp_table SET gp_update_avail = TRUE WHERE id = 1;")
    cur.close()


# while loop to update the parameters every 20 seconds.
# cancellable by keyboard interrupt
try:
    while True:
        # update parameters every 20 seconds
        t0 = time.time()
        GPUpdate(model, likelihood, conn)
        tsleep = 20 - (time.time() - t0)
        time.sleep(tsleep)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    conn.close()
