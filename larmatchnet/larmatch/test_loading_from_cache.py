import os,sys
import larmatch
from larmatch.data.larmatch_hdf5_reader import LArMatchHDF5Dataset

dataset = LArMatchHDF5Dataset(load_from_cachefile="cache_list_larmatch_training_dataset.txt")
