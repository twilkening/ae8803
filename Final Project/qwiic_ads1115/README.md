Qwiic_Ads1115_Py
==================

Python module for the ADS1115 analog-to-digital converter.

New to qwiic? Take a look at the entire [SparkFun qwiic ecosystem](https://www.sparkfun.com/qwiic).

## Contents

- [Qwiic\_Ads1115\_Py](#qwiic_ads1115_py)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [**TODO: update the remaining sections**](#todo-update-the-remaining-sections)
  - [Installation](#installation)
    - [PyPi Installation](#pypi-installation)
    - [Local Installation](#local-installation)
  - [Example Use](#example-use)

Dependencies
---------------

This driver package depends on the qwiic I2C driver:
[Qwiic_I2C_Py](https://github.com/sparkfun/Qwiic_I2C_Py)

**TODO: update the remaining sections**
-------------

Installation
-------------

### PyPi Installation

This repository is hosted on PyPi as the [sparkfun-qwiic-joystick](https://pypi.org/project/sparkfun-qwiic-joystick/) package. On systems that support PyPi installation via pip, this library is installed using the following commands

For all users (note: the user must have sudo privileges):

```sh
sudo pip install sparkfun-qwiic-joystick
```

For the current user:

```sh
pip install sparkfun-qwiic-joystick
```

### Local Installation
To install, make sure the setuptools package is installed on the system.

Direct installation at the command line:

```sh
python setup.py install
```

To build a package for use with pip:

```sh
python setup.py sdist
 ```

A package file is built and placed in a subdirectory called dist. This package file can be installed using pip.

```sh
cd dist
pip install sparkfun_qwiic_joystick-<version>.tar.gz

```

Example Use
 ---------------
See the examples directory for more detailed use examples.

```python
from __future__ import print_function
import qwiic_joystick
import time
import sys

def runExample():

    print("\nSparkFun qwiic Joystick   Example 1\n")
    myJoystick = qwiic_joystick.QwiicJoystick()

    if myJoystick.isConnected() == False:
        print("The Qwiic Joystick device isn't connected to the system. Please check your connection", \
            file=sys.stderr)
        return

    myJoystick.begin()

    print("Initialized. Firmware Version: %s" % myJoystick.getVersion())

    while True:

        print("X: %d, Y: %d, Button: %d" % ( \
                    myJoystick.getHorizontal(), \
                    myJoystick.getVertical(), \
                    myJoystick.getButton()))

        time.sleep(.5)

if __name__ == '__main__':
    try:
        runExample()
    except (KeyboardInterrupt, SystemExit) as exErr:
        print("\nEnding Example 1")
        sys.exit(0)

```
<p align="center">
<a href="https://www.sparkfun.com" alt="SparkFun">
<img src="https://cdn.sparkfun.com/assets/custom_pages/3/3/4/dark-logo-red-flame.png" alt="SparkFun - Start Something"></a>
</p>
