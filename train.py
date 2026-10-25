"""
train.py
================================================================================
Defines `fit(args: config.ArgConfig)` as the single entry point for training the 
model. In particular:
1. Uses `config.ArgConfig` (see "config.py") which is a centralized experiment 
   hyperparameters and paths object.
2. Instantiates `model.Jimmy` (see "model.py") and `DatasetManager` 
   (a `LightningDataModule` object; see "dataset.py").
3. Configures a TensorBoard logger under 
   "logs_ckpts/<forecast_mode>/<version_comment>".
4. Configures a `ModelCheckpoint` callback that monitors `val_WAPE_epoch` and 
   saves the best-epoch ckpt; see "utils.py" for WAPE defn.
5. Builds a Lightning `Trainer` w/settings retrieved from `args`; see 
   `help(config.ArgConfig)` for more info. 


USAGE (see "run.py" for runtime setup)
--------------------------------------------------------------------------------
    from config import ArgConfig
    from train import fit
    
    # `ArgConfig` is not a bag of settings; it is the architectural contract for
    # .. an experiment. A decision such as `timestep_horizon` is made once and 
    # .. propagates through data construction, allowing the entire system to 
    # .. reconfigure coherently around a single definition of time.
    
    args_next_week = ArgConfig(forecast_mode        = "next-week",
                               train_val_idxs_fpath = os.path.join(PROCESSED_DIR, "train_val_i_time-split.csv"),
                               cpu_workers          = CPU_NUM,
                               device               = DEVICE,
                               sales_fpath          = SALES_FPATH,
                               price_discount_fpath = PRICE_DISCOUNT_FPATH,
                               holidays_fpath       = HOLIDAYS_FPATH,
                               img_dir              = IMG_DIR,
                               logs_ckpts_dir       = LOGS_CKPTS_DIR,
                               )

    print(args_next_week)
    
    fit(args = args_next_week) # owns assembly and execution
"""

from lightning.pytorch import seed_everything # To seed Python, Torch, NumPy (inc. Pandas)
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

import model
import dataset

from importlib import reload # Re-imports the module so the latest edits are reflected w/o restarting the Jupyter kernel
reload(model) 
reload(dataset)

from dataset import DatasetManager


def fit(args, ckpt_path = None): # `args` is an experiment-wide hyper-parameter and paths config object, `ArgConfig`; 
                                 # .. see "config.py"

    # Seeds Python, Torch, NumPy (inc. Pandas) in all downstream modules at the start of EACH `fit()` call so every run 
    # .. starts from the same RNG state; w/o per-run seeding, run 2 in a long-lived Jupyter session would continue from 
    # .. the RNG state advanced by run 1
    if args.deterministic: seed_everything(args.user_seed, workers = True) 
    
    model_instance = model.Jimmy(args) # Instantiate the forecasting model; affectionately called `Jimmy`
    datamodule     = DatasetManager(args) # `datamodule` is a `DatasetManager` instance (which inherits from 
                                          # .. `LightningDataModule`); see "dataset.py"           
                                          
    if args.debug: # Disable experiment logging and checkpointing in debug mode
        logger_val    = False
        callbacks_val = None
        enable_checkpointing_val = False # Avoids setting the default `ModelCheckpoint`
    else:    
        logger_val   = pl_loggers.TensorBoardLogger(save_dir = args.logs_ckpts_dir,
                                                    name     = args.forecast_mode, 
                                                    version  = args.version_comment,
                                                    ) # Experiment logger; experiment subdirectory is constructed from 
                                                      # .. args: "`logs_ckpts_dir`/`forecast_mode`/`version_comment`"
    
        # Custom callback to save model parameters (.ckpt) if latest monitored metric is less than the best-so-far 
        # .. (per `mode = "min"`); logged using `log_dict` in "model.py"; written to disk under the directory implied by 
        # .. `Trainer(logger = , ...)`. See "utils.py" for WAPE defn. NOTE: `ModelCheckpoint` runs last among callbacks, 
        # .. so `val_WAPE_epoch` value, which is logged `on_validation_epoch_end` in `model.Jimmy`, is read at the 
        # .. `on_validation_end` hook.
        callbacks_val = ModelCheckpoint(monitor  = "val_WAPE_epoch", # key tracked by `log_dict`
                                        mode     = "min",
                                        filename = f"ckpt-{args.version_comment}-"+"{epoch:03d}-{train_loss_epoch:.6f}-{val_loss_epoch:.6f}-{val_WAPE_epoch:.6f}", # `{}_loss_epoch` is MSE
                                        ) # Optimize on MSE but monitor WAPE
        
        enable_checkpointing_val = True
    
    trainer = Trainer(deterministic = args.deterministic, # Paired w/`seed_everything` for experiment reproducibility
                      accelerator   = args.device, # NOTE: Any new tensors you create inside the model, a 
                                                   # .. `pl.LightningModule`, (e.g., in `forward()` or in hooks) must be 
                                                   # .. moved manually to device via `.to(self.device)`
                      devices       = 1, # Number of GPUs
                      max_epochs    = args.epochs,
                      logger               = logger_val, 
                      callbacks            = callbacks_val,
                      enable_checkpointing = enable_checkpointing_val, 
                      )                       
                                                                                  
        
    trainer.fit(model      = model_instance,
                datamodule = datamodule,
                ckpt_path  = ckpt_path,
                )
    
    return None