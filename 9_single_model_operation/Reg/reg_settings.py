# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
import os

MODEL_SAVE_PATH = "Reg_Model"
MODEL_NAME = "reg_model"
SUMMARY_PATH = "Reg_Logs"
TRAINING_STEPS = 100

# 随机生成若干个点，围绕在y=0.1x+0.3的直线周围
num_points = 100