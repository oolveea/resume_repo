"""
config.py
================================================================================
Defines the `utils.ArgConfig` object which centralizes experiment-wide 
hyperparameters and file paths into a single `args` object. These settings are 
distributed to components throughout the training pipeline. For full args info, 
see `help(ArgConfig)`.


USAGE
--------------------------------------------------------------------------------
    from config import ArgConfig
    
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
    
    help(ArgConfig)
"""

import textwrap # To wrap print long strings 


class ArgConfig: # An experiment-wide hyper-parameter and paths config object
    def __init__(self,
                 debug = False,
                 deterministic = True,
                 user_seed     = 42,
                 forecast_mode        = None,
                 train_val_idxs_fpath = None,
                 timestep_horizon     = 12,
                 teacher_forcing_P    = 1/3,
                 fusion_img_dim = 128,
                 h_t_dim        = 128,
                 epochs     = 20,
                 batch_size = 128,
                 lr         = 1e-3,
                 dropout_P  = 0.2,
                 device      = "cpu",
                 cpu_workers = 12,
                 sales_fpath          = None,
                 price_discount_fpath = None,
                 holidays_fpath       = None,
                 img_dir              = None,
                 logs_ckpts_dir  = None,
                 version_comment = None,
                 ):
        
        
        # PATHS CHECK
        # --------------------------------------------------------------------------------------------------------------
        if any(path is None for path in (sales_fpath, price_discount_fpath, holidays_fpath, train_val_idxs_fpath, img_dir)):
            raise ValueError("Missing path(s). Please provide full paths to data files and the image dir path:\n"
                             "- sales_fpath\n"
                             "- price_discount_fpath\n"
                             "- holidays_fpath\n"
                             "- train_val_idxs_fpath\n"
                             "- img_dir\n"
                             )
            
        self.sales_fpath          = sales_fpath
        self.price_discount_fpath = price_discount_fpath
        self.holidays_fpath       = holidays_fpath
        self.train_val_idxs_fpath = train_val_idxs_fpath
        self.img_dir              = img_dir


        # F0RECAST SPECIFICATI0N
        # --------------------------------------------------------------------------------------------------------------
        if forecast_mode not in ["horizon", "next-week"]:
            raise ValueError(f"Invalid `forecast_mode = {forecast_mode}`: expected 'horizon' or 'next-week'.") 
        self.forecast_mode = forecast_mode
        
        if isinstance(timestep_horizon, int) and (1 <= timestep_horizon <= 12):
            self.timestep_horizon = timestep_horizon
        else:
            raise ValueError(f"Invalid `horizon = {timestep_horizon}`: expected an integer in [1, 12] interval.")
        
        if self.forecast_mode == "horizon":
            if teacher_forcing_P is None or isinstance(teacher_forcing_P, float) and (0.0 < teacher_forcing_P <= 1.0):
                self.teacher_forcing_P = teacher_forcing_P
            else: 
                raise ValueError(f"Invalid `teacher_forcing_P = {teacher_forcing_P}`: expected a float in (0.0, 1.0] interval or None (i.e., teacher forcing OFF).")
        elif self.forecast_mode == "next-week":
            self.teacher_forcing_P = None # In “next-week” mode, `Jimmy` auto-regressively passes ground truth `y` at 
                                          # .. every timestep in training and val, so teacher forcing does not apply 
                                          # .. i.e., set to `None`
            print('Teacher forcing is OFF in `forecast_mode == "next-week"` (i.e., `teacher_forcing_P = None`).\n')
        
        # NOTE: `teacher_forcing_P = 1.0` in "horizon" mode matches "next-week" behavior during TRAINING only; during 
        # .. val these modes differ because "horizon" always uses its own previous prediction while "next-week" always 
        # .. uses ground truth `y`.
        
        
        # EXPERIMENT REPR0DUCIBILITY
        # --------------------------------------------------------------------------------------------------------------
        self.deterministic = bool(deterministic)
        self.user_seed     = user_seed
        
        if self.deterministic:
            if not isinstance(self.user_seed, int):
                raise ValueError(f"Invalid `seed = {self.user_seed!r}`: expected an integer when `deterministic = True`.")
        else: # User seed has no effect in non-deterministic mode
            seed_note = f": `user_seed = {self.user_seed}` is ignored" if self.user_seed is not None else "" 
            print(f"The run is not deterministic{seed_note}. To enable, re-init `ArgConfig(deterministic = True)`.\n")
        
        
        # DR0P0UT
        # --------------------------------------------------------------------------------------------------------------
        # Mirrors `nn.Dropout(p = ...)` input semantics; this check is required by custom `LockedDropout` in "model.py"
        if isinstance(dropout_P, float) and (0.0 <= dropout_P <= 1.0):
            self.dropout_P = dropout_P
        else:
            raise ValueError(f"Invalid `dropout_P = {dropout_P}`: expected a float in [0.0, 1.0] interval.")
        

        # 0THER ARGS
        # --------------------------------------------------------------------------------------------------------------
        self.debug = bool(debug)
        
        self.fusion_img_dim = fusion_img_dim
        self.h_t_dim        = h_t_dim
        
        self.device      = device
        self.cpu_workers = cpu_workers
        
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        
        self.logs_ckpts_dir  = logs_ckpts_dir
        self.version_comment = version_comment


    def __str__(self):
        
        def _format_path__str__(path):
            # Wrap a path string to 80 chars for readable printing
            if path == None: path = "None"
            return textwrap.fill(path, width = 80, subsequent_indent = ' ' * 25)
            
        args_str = (f"{'debug':<25}{self.debug}\n"
                    f"{'deterministic':<25}{self.deterministic}\n"
                    f"{'user_seed':<25}{self.user_seed}\n"
                    "\n"
                    f"{'train_val_idxs_fpath':<25}{_format_path__str__(self.train_val_idxs_fpath)}\n"
                    "\n"
                    f"{'forecast_mode':<25}{self.forecast_mode}\n"
                    f"{'timestep_horizon':<25}{self.timestep_horizon}\n"
                    f"{'teacher_forcing_P':<25}{self.teacher_forcing_P}\n"
                    "\n"
                    f"{'fusion_img_dim':<25}{self.fusion_img_dim}\n"
                    f"{'h_t_dim':<25}{self.h_t_dim}\n"
                    "\n"
                    f"{'epochs':<25}{self.epochs}\n"
                    f"{'batch_size':<25}{self.batch_size}\n"
                    f"{'lr':<25}{self.lr}\n"
                    f"{'dropout_P':<25}{self.dropout_P}\n"
                    "\n"
                    f"{'device':<25}{self.device}\n"
                    f"{'cpu_workers':<25}{self.cpu_workers}\n"
                    "\n"
                    f"{'sales_fpath':<25}{_format_path__str__(self.sales_fpath)}\n\n"
                    f"{'price_discount_fpath':<25}{_format_path__str__(self.price_discount_fpath)}\n\n"
                    f"{'holidays_fpath':<25}{_format_path__str__(self.holidays_fpath)}\n\n"
                    f"{'img_dir':<25}{_format_path__str__(self.img_dir)}\n"
                    "\n"
                    f"{'logs_ckpts_dir':<25}{_format_path__str__(self.logs_ckpts_dir)}\n\n"
                    f"{'version_comment':<25}{_format_path__str__(self.version_comment)}\n"
                    )

        return args_str


ArgConfig.__doc__ = """
See the "F0RECAST SPECIFICATI0N" section below for how the experiment means.

EXPERIMENT REPR0DUCIBILITY AND DEBUG M0DE
================================================================================
debug : bool = False
    Enables debug mode: disables logging (no TensorBoard artifacts, no 
    checkpointing) in "train.py": truncates train/val indices to `[:256]` (two 
    128-size mini-batches) in "dataset.py" so train/val pipelines run fast.
    
deterministic : bool = True    
    In "train.py", calls `seed_everything(user_seed, workers=True)` and sets 
    `Trainer(deterministic = True, ...)` for identical arithmetic between 
    experiments; tested and confirmed arithmetic determinism (and inter-hardware
    consistency) across "cpu", "mps", and "cuda".
    
user_seed : int or None = 42
    RNG seed when `deterministic = True`; ignored otherwise.


F0RECAST SPECIFICATI0N
================================================================================
timestep_horizon : int = 12
   Number of predicted decoder time-steps (i.e., weeks). Must satisfy 
   (1 ≤ `timestep_horizon` ≤ 12) where 12 is the maximum horizon supported by 
   the dataset.
   
forecast_mode : {'horizon', 'next-week'}
    Choose multi-step "horizon" or single-step "next-week" forecast (see 
    `train_val_idxs_fpath` arg below for explanation). Also passed to 
    `pl_loggers.TensorBoardLogger(name = , ...)` in "train.py" (see "VERSI0N 
    C0NTR0L" section below). 
    
teacher_forcing_P : float or None = 1/3
    Enables probabilistic ground-truth injection during training in "horizon" 
    mode; sets the probability of passing true `y`. When set to `None` in 
    "horizon" mode, teacher forcing is OFF: after the initial delimeted step, 
    always feed back the model’s own prediction. In "next-week" mode, this 
    control is unused and is auto-configured to `None`; observed sales are 
    always fed back.
    
    NOTE: `teacher_forcing_P = 1.0` in "horizon" mode matches "next-week" 
    behavior during TRAINING only; during val these modes differ because 
    "horizon" always uses its own previous prediction while "next-week" always 
    uses ground truth `y`.

train_val_idxs_fpath : str
    Full path to CSV containing precomputed train/validation index splits;
    split semantics are described below:

    "train_val_i_time-split-no-overlap.csv" | forecast_mode = "horizon"
    ----------------------------------------------------------------------------
    This train-validation split simulates a pre-season new product introduction 
    (NPI) forecast i.e., predicting demand for new SKUs (no historical sales) 
    for the entirety of a future season. The model is trained to predict  
    "y_hat_t+1" from "y_hat_t" input over 12 steps.

    To build this split, we iteratively select a sample with the latest release 
    date in each of the final SS and AW seasons and move all samples sharing its
    `ext_code` (i.e., SKU  number), regardless of the samples' own release 
    dates, into the validation set. We stop when validation reaches ~10% of the 
    full dataset. This confines each SKU to either train or val split and 
    ensures the model never sees any validation SKUs during training. 

    However, the split separates SKUs without guaranteeing chronological 
    separation. Selecting a SKU for its latest release also moves its earlier
    releases into validation. Some samples left in training therefore have
    release dates later than those early validation samples—even though no SKU
    appears in both sets.

    "train_val_i_time-split.csv" | forecast_mode = "next-week" OR "horizon"
    ----------------------------------------------------------------------------
    This train–validation split simulates activating the forecasting system 
    mid-season. At mid-season, "t" weeks (i.e., time-steps) of real sales "y_t" 
    for each active SKU have been observed. From these "t" weeks the model can:

    - "next-week": The model trains to predict "y_hat_t+1" from "y_t" input 
      over 12 time-steps. IRL inference, the model would forecast demand for 
      just the next week (i.e., "t + 1") to guide SKU markdown or transfer. 

    - "horizon": The model trains to predict "y_hat_t+1" from "y_hat_t"  
      input over 12 time-steps (identical to the NPI forecast). IRL inference, 
      the model would re-forecast demand for the remainder of the season (weeks 
      "t + 1" through 12; or from "t = 0" for a mid-season NPI) to guide SKU 
      replenishment.

    This split slices by release date, moving the latest 10% of samples from the 
    full dataset to the validation set. It does not slice a single SKU’s sales 
    curve at the 10% release date cut-off: the entire curve falls in train or 
    validation. Since identical `ext_code`s (i.e., SKUs can launch at different 
    retail locations, they may have different release dates that fall on either 
    side of the 10% release date cut-off. Hence, some `ext_code`s appear in both
    sets. 


ARCHITECTURE C0NTR0LS
================================================================================
fusion_img_dim : int = 128
    Number of dimensions in `fusion_cat_t` (input to `fusion_mlp`) allocated
    to the image (vs. attrs + time-series). Does NOT affect: `ImageEncoder`
    outputs, `AdditiveAttention` capacity (`attn_dim`), or `decoder_rnn`
    hidden size (`h_t_dim`).
    
h_t_dim : int = 128
    Hidden layer size of the `decoder_rnn`. 


EXPERIMENT SPECIFICATI0N
================================================================================
epochs : int = 20
    Number of training epochs; passed to `Trainer(max_epochs = , ...)` in 
    "train.py".
    
batch_size : int = 128
    Mini-batch size for both training and validation; passed to 
    `DataLoader(batch_size = , ...)` in "dataset.py".
    
lr : float = 1e-3
    Learning rate for the optimizer; see "model.py".
    
dropout_P : float = 0.2
    Dropout rate applied in the decoder; see `LockedDropout` 
    in "model.py".
    
    
HARDWARE C0NTR0LS
================================================================================
device : str = "cpu"
    Sets `Trainer(accelerator = device, ...)` in "train.py"; places registered 
    model parameters and DataLoader batches on the specified device (typically 
    "cpu", "cuda" or "mps"). Tensors created manually inside model methods must 
    be created on, or explicitly moved to, `self.device`.
    
cpu_workers : int = 12
    Sets the number of `DataLoader` subprocesses in "dataset.py". 
    
    NOTE: Non-image tensors are preprocessed once and stored in memory. To save 
    RAM, images are loaded from disk and transformed on the CPU on demand 
    inside `DataProcessor.__getitem__` (see "dataset.py"). To make this assembly
    faster s.t. the GPU does not wait, set `cpu_workers` to CPU core count; this
    spawns that many CPU subprocesses that assemble batches in parallel.
    

DATASET PATHS
================================================================================
sales_fpath : str
    Full path to "sales.csv".
    
price_discount_fpath : str
    Full path to "price_discount_series.csv".
    
holidays_fpath : str
    Full path to "holidays.csv".
    
img_dir : str
    Path to directory containing SKU images.


VERSI0N C0NTR0L
================================================================================
logs_ckpts_dir : str or None = None
    Path to directory for saving TensorBoard logs and Lightning checkpoints; 
    passed to `pl_loggers.TensorBoardLogger(save_dir = , ...)` (see 
    `version_comment` arg below). Required when `debug = False`; ignored when 
    `debug = True`.
    
version_comment : str or None = None
    Experiment label passed to `pl_loggers.TensorBoardLogger(version = , ...)`. 
    If set, experiment subdirectory is: 
    "`logs_ckpts_dir`/`forecast_mode`/`version_comment`"; else: 
    "`logs_ckpts_dir`/`forecast_mode`/'version_N'"
""".strip()