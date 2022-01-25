import numpy as np

import torch
import torch.nn as nn

from lightmodelnet import LightModelNet

def main():
    
    model = LightModelNet(3)
    print model

if __name__ == '__main__':
    main()
