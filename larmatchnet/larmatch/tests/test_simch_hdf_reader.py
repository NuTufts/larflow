import os,sys
import numpy as np
import larmatch
from larmatch.data import LArMatchSimChHDF5Dataset

input_file = "out_test.h5"
reader = LArMatchSimChHDF5Dataset(input_file)

entry = reader[0]
for k,v in entry.items():
    if type(v) is np.ndarray:
        print(k,v.shape)
    else:
        print(k,v)
