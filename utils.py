"""
utils.py
================================================================================
Provides helper utilities for data scaling and error metrics:
- Defines `DataScaler`, a helper used to standardize numerical input data via 
  `fwd()` in "dataset.py", and to invert model predictions back to the original 
  scale via `inv()` in "model.py".
- Defines `WAPE(Y_hat, Y)` to compute Weighted Average Percentage Error; used as
  the `ModelCheckpoint` monitor in "train.py", and as a validation metric in 
  "model.py".
- Defines `MAE(Y_hat, Y)` as a wrapper around `nn.L1Loss` (i.e., Mean Absolute 
  Error); used as a validation metric in "model.py".
"""


import torch
import torch.nn as nn


class DataScaler: 
    # Standardizes (i.e., z-scores) numeric input data
    def __init__(self, 
                 mean, 
                 std,
                 ):
        
        self.mean, self.std = mean, std

    def fwd(self, x):
        # Standardize raw numerical inputs in the `DataProcessor` (see 
        # .. "dataset.py")
        return (x - self.mean) / self.std

    def inv(self, x):
        # Invert `y_hat` to original scale in in the model (see `Jimmy` in 
        # "model.py").
        return x * self.std + self.mean


# METRICS
# ----------------------------------------------------------------------------------------------------------------------  
# NOTE: `y_hat_rollout` and `ts_y` are model output tensors of shape (batch_size , timestep_horizon). In "model.py", 
# .. `on_validation_epoch_end`, these per-batch tensors are stacked into `val_Y_hat_epoch`, `val_ts_Y_epoch` to compute
# .. epoch-level validation metrics (not a mean of individual batch-level metrics).

def WAPE(Y_hat, Y):
    # Computes Weighted Average Percentage Error: WAPE = ( sum( |Y - Y_hat| ) / sum( |Y| ) ) * 100
    return (torch.sum(torch.abs(Y - Y_hat)) / torch.sum(torch.abs(Y))).item() * 100

def MAE(Y_hat, Y):
    mae_fn = nn.L1Loss() # Mean Absolute Error = sum( |Y - Y_hat| ) / (num_elements = sample_num * timestep_horizon)
    mae    = mae_fn(Y_hat, Y).item() # Returns a 0-dim tensor containing the MAE; `.item()` extracts it as a float
    return mae
