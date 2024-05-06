# -----------------------------------------------------------------------------
# MOTION.py
#
# Python script to move the Jetbot back and forth as it collects data.
#
# ------------------------------------------------------------------------
#
# Written by Theodore Wilkening, May 2024

from time import sleep
from qwiic_ads1115.qwiic_ads1115 import QwiicAds1115
import logging
from jetbot import Robot

logger = logging.getLogger(__name__)
FORMAT = (
    "[%(filename)s:%(lineno)s - %(funcName)20s() ] "
    + "%(asctime)s : %(message)s"  # noqa
)  # noqa
logging.basicConfig(
    filename="logs/motion.log",
    filemode="w",
    format=FORMAT,
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

robot = Robot()

if __name__ == "__main__":
    try:
        forward = 1
        while True:
            # travel back and forth
            if forward == 1:
                logger.debug("forward motion set.")
                robot.forward(0.3)
                sleep(5)
                forward = 0
            elif forward == 0:
                logger.debug("backward motion set.")
                robot.backward(0.3)
                sleep(5)
                forward = 1
            else:
                logger.debug("something weird happened.")
                robot.stop()
                break
            
    except KeyboardInterrupt:
        print("stopped by user.")
        logger.debug("stopped by user.")
    finally:
        robot.stop()