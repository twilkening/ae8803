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


def GPRegression():
    # Perform Regression on the new data
    # TODO
    tmp = 1
    return tmp


def fetch_and_plot_data(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT time, measured_value FROM daq_table WHERE processed = FALSE"
    )  # noqa
    rows = cur.fetchall()
    data = np.asarray(rows)
    # plot new data
    plt.scatter(data[:, 0], data[:, 1], "k*")
    plt.draw()
    # plot regression given the new data

    # in order to update the plot so that it clears the previous
    # estimate of where the means would lie, I need to update the entire
    # plot I believe... will have to hold onto all of the data then (or
    # get it every time) and also will need to have a separate table in
    # the database for Regression estimates...
    # I need to somehow save the regression ... or repeat it over all of
    # the data each time?? that doesn't make sense...
    cur.execute(
        "UPDATE daq_table SET processed = TRUE WHERE processed = FALSE"  # noqa
    )  # noqa
    conn.commit()
    cur.close()


def scheduled_fetch():
    conn = psycopg2.connect("dbname=test user=postgres")
    plt.ion()
    fig, ax = plt.subplots()

    try:
        while True:
            fetch_and_plot_data(conn)
            time.sleep(2)  # Fetch data every 2 seconds
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        conn.close()
        plt.close()


scheduled_fetch()
