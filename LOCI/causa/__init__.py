import os

# PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
# ROOT = '/'.join(PACKAGE_DIR.split('/')[:-1])
# DATA_DIR = ROOT + '/data'

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))  # This points to the `causa` folder
ROOT = os.path.join(PACKAGE_DIR, "..")  # Go up one level to `ROCHE`
DATA_DIR = os.path.join(PACKAGE_DIR, "../../DATA")  # Point to `data` directory within `ROCHE`