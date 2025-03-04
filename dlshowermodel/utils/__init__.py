# Import the scheduler modules to ensure registration happens
from . import schedulers

# Expose the get_lr_scheduler function
from .lr_scheduler import get_lr_scheduler