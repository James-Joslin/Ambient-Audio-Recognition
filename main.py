import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm
from glob import glob
import pandas as pd
import tensorflow as tf

import external as ext

AUTOTUNE = tf.data.AUTOTUNE

if __name__ == "__main__":
    pass