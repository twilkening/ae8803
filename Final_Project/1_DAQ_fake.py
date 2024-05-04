# -----------------------------------------------------------------------------
# DAQ.py
#
# Python script to collect data into a PostgreSQL database.
#
# ------------------------------------------------------------------------
#
# Written by Theodore Wilkening, May 2024

import psycopg2
from time import sleep
from qwiic_ads1115.qwiic_ads1115 import QwiicAds1115
import time
import numpy as np
from config import load_config


# configure the ADS1115
ads = QwiicAds1115()
ads.os = "0"  # no effect
ads.mux = "100"  # Analog In, channel 0, compare to GND
ads.gain = "000"  # +/- 6.144V
ads.mode = "0"  # continuous
ads.data_rate = "000"  # 8 SPS
ads.comp_que = "11"  # disable comparator
ads.scale = 1  # 1V = 1kPa delta between ports
if ads.is_connected():
    ads.configure()
    time.sleep(0.5)
    ads.calibrate()

    # Confirm calibration set:
    if abs(ads.get_measurement()) > 0.1:
        print("Pressure calibration has likely failed")
    else:
        print("ADS1115 successfully initialized and calibrated")
else:
    print("ADS1115 not connected")

# dictionary of ADS1115 data sampling rates
rates = {
    "000": 8,
    "001": 16,
    "010": 32,
    "011": 64,
    "100": 128,
    "101": 250,
    "110": 475,
    "111": 860,
}

# fake data parameters:
period = 60
amp = 0.04
mean = -0.02
var = amp / 8
omega = 2 * np.pi / period  # frequency [rad/s]


# Define function to get new data
def get_new_data(ads_object, tstart):

    ts = 1 / rates[ads_object.data_rate]  # sampling period
    tcollect = 1  # seconds to collect data
    n = tcollect / ts
    data = []  # time, measurement, processed flag

    i = 0
    t0 = time.time()
    tm = t0

    # noise for fake'd data
    noise = np.random.randn(n) * var

    # ts in the while loop evaluation adds some buffer
    while tm < t0 + tcollect + ts:
        # time of measurement & measurement
        tm = time.time()
        # meas = ads_object.measure
        # fake'd data
        tpassed = tm - tstart
        meas = mean + amp * abs(np.sin(tpassed * omega)) + noise[i]

        # add to the data array
        if i < n:
            data.append((tm, meas, False))

        i += 1

        tdelta = t0 + ts * i - time.time()
        if tdelta < 0:
            continue
        else:
            time.sleep(tdelta)

    return data


def create_tables():
    """Create tables in the PostgreSQL database"""
    drops = (
        """DROP TABLE IF EXISTS daq_table;""",
        """DROP TABLE IF EXISTS gp_table;""",
        """DROP TABLE IF EXISTS mean_table;""",
    )
    commands = (
        """
        CREATE TABLE daq_table (
            time FLOAT PRIMARY KEY,
            measured_value FLOAT NOT NULL,
            processed BOOLEAN NOT NULL DEFAULT FALSE
        )
        """,
        """ CREATE TABLE gp_table (
                id SERIAL PRIMARY KEY,
                gp_update_avail BOOLEAN NOT NULL DEFAULT FALSE
                )
        """,
        """
        CREATE TABLE mean_table (
            time FLOAT PRIMARY KEY,
            meas_mean FLOAT NOT NULL,
            processed BOOLEAN NOT NULL DEFAULT FALSE
        )
        """,
    )
    try:
        config = load_config()  # sets connection to "test" database
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                # execute the DROP TABLE statements
                for drop in drops:
                    cur.execute(drop)
                # execute the CREATE TABLE statements
                for command in commands:
                    cur.execute(command)
                # commit the new tables to database
                conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


# NOTE: before we can connect to the PostgreSQL database, it first has
# to be created from the command line interface via psql. See the
# following webpages:
# https://www.postgresqltutorial.com/postgresql-python/connect/
# https://www.postgresqltutorial.com/postgresql-getting-started/install-postgresql/
# https://www.postgresqltutorial.com/postgresql-getting-started/install-postgresql-linux/

# NOTE: we also need to *start* the PostgreSQL database everytime the
# system is powered on:
# sudo systemctl start postgresql
# sudo systemctl enable postgresql


if __name__ == "__main__":
    create_tables()
    # Connect to the PostgreSQL database
    config = load_config()  # sets connection to "test" database
    with psycopg2.connect(**config) as conn:
        with conn.cursor() as cur:
            # initialize gp_table
            cur.execute(
                "INSERT INTO gp_table (gp_update_avail) VALUES (FALSE)"  # noqa
            )  # noqa
    # initialize the
    # # Insert data continuously
    # try:
    #     while True:
    #         # Generate or receive your data
    #         t0 = time.time()
    #         data = get_new_data(ads, t0)
    #         cur.executemany(
    #             " ".join(
    #                 [
    #                     "INSERT INTO daq_table",
    #                     "(time, measured_value, processed)",
    #                     "VALUES (%s, %s, %s)",
    #                 ]
    #             ),
    #             data,
    #         )
    #         conn.commit()
    #         sleep(1 / rates[ads.data_rate])  # Pause for sampling period (sec)
    # except KeyboardInterrupt:
    #     print("stopped by user.")
    # finally:
    #     cur.close()
    #     conn.close()

# if we only want one measurement per entry
# # Define function to get one set of new data
# def get_new_data(ads_object):

#     meas = ads_object.measure
#     return (time.time(), meas)
