# -----------------------------------------------------------------------------
# qwiic_ads1115.py
#
# Python library for communicating with the ADS1115.
#
# ------------------------------------------------------------------------
#
# Written by Theodore Wilkening, April 2024
#
# This python library supports the SparkFun Electronics qwiic
# sensor/board ecosystem
#
# More information on qwiic is at https:// www.sparkfun.com/qwiic
#
# =============================================================================
# Copyright (c) 2024 Theodore Wilkening
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
#
# This libarary was originally inspired off of qwiic_joystick.py and has
# substantial code derived from that library. The library link is below
# as well as their copyright statement.
#
# https://github.com/sparkfun/Qwiic_Joystick_Py
#
# =============================================================================
# Copyright (c) 2019 SparkFun Electronics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""
qwiic_joystick
===============
Python module for the ADC component ADS1115.

This package can be used in conjunction with the overall
[SparkFun qwiic Python Package](https://github.com/sparkfun/Qwiic_Py)

New to qwiic? Take a look at the entire
[SparkFun qwiic ecosystem](https://www.sparkfun.com/qwiic).

"""
# -----------------------------------------------------------------------------

from __future__ import print_function

import qwiic_i2c
import time

# Define the device name and I2C addresses. These are set in the class
# defintion as class variables, making them avilable without having to
# create a class instance. This allows higher level logic to rapidly
# create a index of qwiic devices at runtine
#
# The name of this device
_DEFAULT_NAME = "ADS1115"

# Some devices have multiple availabel addresses - this is a list of
# these addresses.
# NOTE: The first address in this list is considered the default I2C
# address for the device.
_AVAILABLE_I2C_ADDRESS = [0x48, 0x49, 0x4A, 0x4B]

# Register Map for the ADS1115
CONVERSION = 0x00
CONFIG = 0x01
LOW_THRES = 0x10
HI_THRESH = 0x11

# available gains, matched to the gain configuration bits
GAINS = [6.144, 4.096, 2.048, 1.024, 0.512, 0.256, 0.256, 0.256]

# define the class that encapsulates the device being created. All
# information associated with this device is encapsulated by this class.
# The device class should be the only value exported from this module.


class QwiicAds1115(object):
    """
    QwiicAds1115

        :param address: The I2C address to use for the device.
                        If not provided, the default address is used.
        :param i2c_driver: An existing i2c driver object. If not provided
                        a driver object is created.
        :return: The QwiicAds1115 device object.
        :rtype: Object
    """

    # Class Constructor
    device_name = _DEFAULT_NAME
    available_addresses = _AVAILABLE_I2C_ADDRESS

    # Instance Constructor
    def __init__(self, address=None, i2c_driver=None):

        # Did the user specify an I2C address?
        self.address = address if address is not None else self.available_addresses[0]

        # load the I2C driver if one isn't provided

        if i2c_driver is None:
            self._i2c = qwiic_i2c.getI2CDriver()
            if self._i2c is None:
                print("Unable to load I2C driver for this platform.")
                return
        else:
            self._i2c = i2c_driver

        # define the other instance variables

        # Operational Status or single-shot conversion start
        # When writing:
        # 0 : No effect
        # 1 : Start a single conversion (when in power-down state)
        # When reading:
        # 0 : Device is currently performing a conversion
        # 1 : Device is not currently performing a conversion (default)
        self.os = "1"

        # input multiplexer configuration
        # 000 : AINP = AIN0 and AINN = AIN1 (default)
        # 001 : AINP = AIN0 and AINN = AIN3
        # 010 : AINP = AIN1 and AINN = AIN3
        # 011 : AINP = AIN2 and AINN = AIN3
        # 100 : AINP = AIN0 and AINN = GND
        # 101 : AINP = AIN1 and AINN = GND
        # 110 : AINP = AIN2 and AINN = GND
        # 111 : AINP = AIN3 and AINN = GND
        self.mux = "000"

        # programmable gain amplifier configuration
        # 000 : FSR = ±6.144 V
        # 001 : FSR = ±4.096 V
        # 010 : FSR = ±2.048 V (default)
        # 011 : FSR = ±1.024 V
        # 100 : FSR = ±0.512 V
        # 101 : FSR = ±0.256 V
        # 110 : FSR = ±0.256 V
        # 111 : FSR = ±0.256 V
        self.gain = "010"

        # device operating mode
        # 0: continuous-conversion mode
        # 1: single-shot or power-down state (default)
        self.mode = "1"

        # data rate
        # 000 : 8 SPS
        # 001 : 16 SPS
        # 010 : 32 SPS
        # 011 : 64 SPS
        # 100 : 128 SPS (default)
        # 101 : 250 SPS
        # 110 : 475 SPS
        # 111 : 860 SPS
        self.data_rate = "100"

        # comparator mode
        # 0: traditional comparator (default)
        # 1: window comparator
        self.comp_mode = "0"

        # comparator polarity
        # 0: active low (default)
        # 1: active high
        self.comp_pol = "0"

        # latching comparator
        # 0: non-latching comparator (default)
        # 1: latching comparator
        self.comp_latch = "0"

        # comparator queue and disable
        # 00 : Assert after one conversion
        # 01 : Assert after two conversions
        # 10 : Assert after four conversions
        # 11 : Disable comparator and set ALERT/RDY pin to high-
        # impedance (default)
        self.comp_que = "11"

        # calibration offset
        self.cal_offset = 0

        # scale voltage to engineering values [units/V]
        self.scale = 1

    def is_connected(self):
        """
        Determine if an ADS1115 device is conntected to the system..

        :return: True if the device is connected, otherwise False.
        :rtype: bool

        """
        return qwiic_i2c.isDeviceConnected(self.address)

    connected = property(is_connected)

    def calibrate(self, n=50):
        """
        Calibrate the readings of the ADS1115 based on the current
        configuration by taking n readings, averaging them, and then
        setting an offset value from 2.5V.

        Args:
            n (int, optional): number of measurements to average over.
            Defaults to 50.

        Returns:
            success (int): returns 0 for failure, or 1 for success
        """
        # TODO

    def reset_cal(self):
        """
        Resets the calibration offset to 0.
        """
        self.cal_offset = 0

    def get_measurement(self):
        """
        Requests one measurement from the conversion register on the
        ads1115. In continuous conversion mode, the reading will change
        every request - in single shot mode, the reading will stay the
        same until the next shot is requested.

        This function additionally applies a conversion to engineering
        units based on the value of self.scale:

        measurement =
        register value [DN] / 2^15 [DN] * gain [V/DN] * scale [units/V]

        Additional notes from ads1115 documentation:
        "The 16-bit Conversion register contains the result of the last
        conversion in binary two's complement format. Following
        power-up, the Conversion register is cleared to 0, and remains
        0 until the first conversion is completed."

        Returns:
            measurement (float): the measured value in either [V] or the
            engineering units defined by scale [units/V]
        """

        # get the register data
        # two bytes from the conversion register
        reg = self._i2c.readBlock(self.address, CONVERSION, 2)

        # shift the register data
        val = reg[0] << 8 | reg[1]

        # convert the register data
        gain = GAINS[int(self.gain, base=2)]
        measurement = val / 2**15 * gain * self.scale

        return measurement

    measure = property(get_measurement)

    def configure(self):
        """
        Method for setting the configuration of the ads1115 behavior per
        the configuration variables.

        Returns:
            success (int): 0 = failed to properly configure, 1 = success
        """
        # build the configuration command
        cfg_block = [
            int(self.os + self.mux + self.gain + self.mode, base=2),  # noqa
            int(
                self.data_rate
                + self.comp_mode
                + self.comp_pol
                + self.comp_latch
                + self.comp_que,
                base=2,
            ),
        ]

        # send the configuration command on the i2c bus
        self._i2c.writeBlock(self.address, CONFIG, cfg_block)
        # wait 100ms for ads1115 to process the configuration request
        time.sleep(0.1)
        # read back the 2 bytes of the configuration register
        cfg_set = self._i2c.readBlock(self.address, CONFIG, 2)
        # confirm the configuration matches as expected
        success = cfg_block == cfg_set

        return int(success)

    def reset_config(self):
        """
        Set the configuration back to the default configuration.

        Returns:
            success (int): 0 = failed to properly configure, 1 = success
        """
        # two bytes to reset:
        # [os + mux + gain + mode]
        # [data_rate + comp_mode + comp_pol + comp_latch + comp_que]
        cfg_block = [
            int("1" + "000" + "010" + "1", base=2),
            int("100" + "0" + "0" + "0" + "11", base=2),
        ]
        # send the configuration command on the i2c bus
        self._i2c.writeBlock(self.address, CONFIG, cfg_block)
        # wait 100ms for ads1115 to process the configuration request
        time.sleep(0.1)
        # read back the 2 bytes of the configuration register
        cfg_set = self._i2c.readBlock(self.address, CONFIG, 2)
        # confirm the configuration matches as expected
        success = cfg_block == cfg_set
        return int(success)

    def config(self):
        """
        Gets and prints the current configuration settings of the
        ads1115.
        """
        # TODO
