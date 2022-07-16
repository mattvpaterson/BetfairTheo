import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime as datetime
import time


class Variance_Fit:

    def __init__(self, base_frame):
        self.base = base_frame

    def associate_constant_variance(self,std):
        #build this out to enable custom mapping
        self.base.loc[:, 'std'] = std

