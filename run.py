"""
run.py
================================================================================
Terminal entry point for `fit()` in "train.py". Run from "codebase" directory 
in project's env.


USAGE
--------------------------------------------------------------------------------
    python -u run.py --help
    
    # Next-week forecasting
    python -u run.py --forecast_mode next-week --train_val_idxs_fpath train_val_i_time-split.csv

    # Horizon forecasting
    python -u run.py --forecast_mode horizon --train_val_idxs_fpath train_val_i_time-split-no-overlap.csv


WALKTHR0UGH
--------------------------------------------------------------------------------
1. `-u`: Python flag; disables output buffering so training output is written
   as it is produced.
   
2. `forecast_mode` and the split filename, `train_val_idxs_fpath`, are required. 
   Split filename path is resolved under "data/visuelle/processed"; absolute
   split paths are also accepted.

3. Append other settings to either command, e.g.:

     `--epochs 30 --batch_size 128 --version_comment baseline`
   
   Unspecified settings use `ArgConfig` defaults, except hardware and standard 
   paths, whose defaults are detected/computed in `main()` below.
    
4. The resolved config is printed before training; enter `y` to start.


REP0 STRUCTURE
--------------------------------------------------------------------------------
..
    repository
    ├── codebase
    │   ├── config.py
    │   ├── dataset.py
    │   ├── model.py
    │   ├── run.py
    │   ├── train.py
    │   └── utils.py
    ├── data
    │   └── visuelle
    │       ├── processed
    │       │   ├── train_val_i_time-split-no-overlap.csv
    │       │   └── train_val_i_time-split.csv
    │       └── raw
    │           ├── images
    │           ├── holidays.csv
    │           ├── price_discount_series.csv
    │           └── sales.csv
    └── logs-ckpts
"""


import sys
from config import ArgConfig


def main():
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(ArgConfig.__doc__)
        return None
    
    import argparse
    import multiprocessing
    import os

    import lightning.pytorch
    import torch

    from train import fit

    print(f"Lightning: {lightning.pytorch.__version__}")

    
    # C0NFIG PATHS
    # ------------------------------------------------------------------------------------------------------------------
    WORK_DIR       = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR       = os.path.dirname(WORK_DIR)
    DATA_DIR       = os.path.join(BASE_DIR, "data", "visuelle")
    RAW_DIR        = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
    LOGS_CKPTS_DIR = os.path.join(BASE_DIR, "logs-ckpts")

    SALES_FPATH          = os.path.join(RAW_DIR, "sales.csv")
    PRICE_DISCOUNT_FPATH = os.path.join(RAW_DIR, "price_discount_series.csv")
    HOLIDAYS_FPATH       = os.path.join(RAW_DIR, "holidays.csv")
    IMG_DIR              = os.path.join(RAW_DIR, "images")

    print("\n"
          f"WORK_DIR: {WORK_DIR}\n"
          f"BASE_DIR: {BASE_DIR}\n"
          )
    
    
    # HARDWARE CHECKS
    # ------------------------------------------------------------------------------------------------------------------
    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available():       DEVICE = "cuda"
    else:                                 DEVICE = "cpu"

    CPU_NUM = multiprocessing.cpu_count()
    
    print(f"CPU_NUM: {CPU_NUM}\n"
          f"DEVICE: {DEVICE}\n"
          )


    # ARG PARSER
    # ------------------------------------------------------------------------------------------------------------------
    # `argparse.SUPPRESS`:  If arg not specified in  CLI, omit passing None; the ArgConfig default takes effect. 
    # .. Explicit argparse defaults still take precedence.
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    
    parser.add_argument("--debug", action = argparse.BooleanOptionalAction)
    
    parser.add_argument("--deterministic", action = argparse.BooleanOptionalAction)
    parser.add_argument("--user_seed",     type   = int)
    
    parser.add_argument("--forecast_mode")
    parser.add_argument("--train_val_idxs_fpath", type = lambda value: os.path.join(PROCESSED_DIR, value))
    parser.add_argument("--timestep_horizon",     type = int)
    parser.add_argument("--teacher_forcing_P",    type = lambda value: None if value.lower() == "none" else float(value))
    
    parser.add_argument("--fusion_img_dim", type = int)
    parser.add_argument("--h_t_dim",        type = int)
    
    parser.add_argument("--epochs",     type = int)
    parser.add_argument("--batch_size", type = int)
    parser.add_argument("--lr",         type = float)
    parser.add_argument("--dropout_P",  type = float)
    
    parser.add_argument("--device",                  default = DEVICE)
    parser.add_argument("--cpu_workers", type = int, default = CPU_NUM)
    
    parser.add_argument("--sales_fpath",          default = SALES_FPATH)
    parser.add_argument("--price_discount_fpath", default = PRICE_DISCOUNT_FPATH)
    parser.add_argument("--holidays_fpath",       default = HOLIDAYS_FPATH)
    parser.add_argument("--img_dir",              default = IMG_DIR)
    parser.add_argument("--logs_ckpts_dir",       default = LOGS_CKPTS_DIR)
    
    parser.add_argument("--version_comment")

    args = ArgConfig(**vars(parser.parse_args()))

    print(args)
    
    if input("Start training? y/n: ") == "y": 
        fit(args = args, ckpt_path = None)

if __name__ == "__main__":
    main()
