"""
MAE (Masked Auto-Encoder) Training Module for Spacepoint Encodings

This module implements semi-supervised masked auto-encoding for learning
robust representations of energy deposition spacepoints in LArTPC data.
"""

from . import models
from . import loss
from . import data
from . import utils
