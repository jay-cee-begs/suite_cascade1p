#Manual Mode
import os, warnings
import sys
import glob
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')
#%matplotlib widget # can be commented back in to make plots interactive
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import pickle
## import functions ##
from functions_general import *
from CASCADE_functions import *
from functions_plots import *
from functions_data_transformation import *


## import configurations ##
import configurations
ops_path

"""Activate cascade env"""
from configurations import *
import run_cascade
from functions_plots import *
if __name__ == "__main__":
    run_cascade.main()
groups
