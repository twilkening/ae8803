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

# Register codes for the Joystick
JOYSTICK_ID = 0x00
JOYSTICK_VERSION1 = 0x01
JOYSTICK_VERSION2 = 0x02
JOYSTICK_X_MSB = 0x03
JOYSTICK_X_LSB = 0x04
JOYSTICK_Y_MSB = 0x05
JOYSTICK_Y_LSB = 0x06
JOYSTICK_BUTTON = 0x07
JOYSTICK_STATUS = 0x08  # 1  -> button clicked
JOYSTICK_I2C_LOCK = 0x09
JOYSTICK_CHANGE_ADDREESS = 0x0A


# define the class that encapsulates the device being created. All
# information associated with this device is encapsulated by this class.
# The device class should be the only value exported from this module.


class QwiicJoystick(object):
    """
    QwiicJoystick

        :param address: The I2C address to use for the device.
                        If not provided, the default address is used.
        :param i2c_driver: An existing i2c driver object. If not provided
                        a driver object is created.
        :return: The QwiicJoystick device object.
        :rtype: Object
    """

    # Constructor
    device_name = _DEFAULT_NAME
    available_addresses = _AVAILABLE_I2C_ADDRESS

    # Constructor
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

    def is_connected(self):
        """
        Determine if an ADS1115 device is conntected to the system..

        :return: True if the device is connected, otherwise False.
        :rtype: bool

        """
        return qwiic_i2c.isDeviceConnected(self.address)

    connected = property(is_connected)

    def get_horizontal(self):
        """
        Returns the 10-bit ADC value of the joystick horizontal position

        :return: The next button value
        :rtype: byte as integer

        """
        msb = self._i2c.readByte(self.address, JOYSTICK_X_MSB)
        lsb = self._i2c.readByte(self.address, JOYSTICK_X_LSB)

        return ((msb << 8) | lsb) >> 6

    horizontal = property(get_horizontal)

    def get_vertical(self):
        """
        Returns the 10-bit ADC value of the joystick vertical position

        :return: The next button value
        :rtype: byte as integer

        """
        msb = self._i2c.readByte(self.address, JOYSTICK_Y_MSB)
        lsb = self._i2c.readByte(self.address, JOYSTICK_Y_LSB)

        return ((msb << 8) | lsb) >> 6

    vertical = property(get_vertical)

    def get_button(self):
        """
        Returns 0 button is currently being pressed.

        :return: button status
        :rtype: integer

        """

        return self._i2c.readByte(self.address, JOYSTICK_BUTTON)

    button = property(get_button)

    def check_button(self):
        """
        Returns 1 if button was pressed between reads of .getButton()
        or .checkButton() the register is then cleared after read.

        :return: button status
        :rtype: integer

        """

        status = self._i2c.readByte(self.address, JOYSTICK_STATUS)

        # We've read this status bit, now clear it
        self._i2c.writeByte(self.address, JOYSTICK_STATUS, 0x00)

        return status

    def get_version(self):
        """
        Returns a string of the firmware version number

        :return: The firmware version
        :rtype: string
        """
        vMajor = self._i2c.readByte(self.address, JOYSTICK_VERSION1)
        vMinor = self._i2c.readByte(self.address, JOYSTICK_VERSION2)

        return "v %d.%d" % (vMajor, vMinor)

    version = property(get_version)
