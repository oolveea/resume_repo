"""
dataset.py
================================================================================
Defines the data pipeline as two cooperating components: `DatasetManager` and 
`DataProcessor`.

`DatasetManager` is a `LightningDataModule` object that _orchestrates_ the
raw-to-tensor pipeline:
  1. Slices raw data using a precomputed train/val index split (see the “RAW
     DATA” section below).    
  2. Instantiates one DataProcessor for each slice (train vs. val), wraps each 
     processor in its own `DataLoader`, and exposes the two loaders to `Trainer` 
     in "train.py" through the datamodule’s `train_dataloader()` and 
     `val_dataloader()` hooks.
  3. Computes and stores train-set metadata (categorical vocabularies;  
     mean/std), which is passed to `DataProcessor` and retrieved by 
     `Jimmy.setup()` (see "model.py"). 

`DataProcessor` is a custom dataset object (`torch.utils.data.Dataset`) that 
owns slice tensorization:
  1. Builds and applies categorical {token -> integer id} maps and z-score  
     transforms using train-set metadata from `DatasetManager`.
  2. Processes and tensorizes one slice per `DataProcessor` instance, exposing 
     indexed samples through the map-style `Dataset` contract (`__len__` and 
     `__getitem__`) consumed by a `DataLoader` (see the “PR0CESSED DATA” section 
     below).
  3. Stores non‐image tensors in memory; images are loaded and transformed 
     on‐the‐fly in `__getitem__` for efficient RAM use.


RAW DATA
--------------------------------------------------------------------------------
1. "sales.csv" columns: 
   [image_path, release_date, retail, category, color, fabric,
   "0", "1", ... , "11"] # Weekly sales series       
2. "price_discount_series.csv" columns: 
   [price, "0", "1", ... , "11" ] # Weekly discount series                   
3. "holidays.csv" column: 
   [date]
4. "train_val_i_<split_type>.csv" columns: 
   [train_i, val_i] # Raw data split by index


PR0CESSED DATA
--------------------------------------------------------------------------------
`DataProcessor.__getitem__(idx)` returns an n-tuple matching the unpack order in
`Jimmy.forward(...)` (see "model.py"):

1. `img`: SKU image tensor, processed on-the-fly in `DataProcessor.__getitem__`; 
   resized, center-padded, and z-scored to pixel-value distributions expected by 
   ResNet weights.
  
2. `ts_y`: Sales curve over `timestep_horizon`; z-scored w/train-set mean/std.

3. `y_star_zero`: A cold-start input; zeros of shape ( : , 1) passed as 
   y_star_{t = 0} (for defn., see `model.Jimmy._on_for_loop_mechanics_`) to 
   predict y_hat_{t = 1}, as y_{t = 0} and y_hat_{t = 0} do not exist. 
  
3. `ts_y_star_mask`: Delimiter tensor of shape ( : , `timestep_horizon`) with
   `mask[:, 0] = 1` and zeros elsewhere; disambiguates synthetic `y_star_zero` 
   from ordinary observed or predicted values.
  
4. `ts_in_market_week`: Time-step position (index) within the forecast horizon; 
   peak normalized; identical across samples. 

5. `ts_calendar_week`: ISO week-of-year, from "release_date" and through the 
   `timestep_horizon`; mapped onto the unit circle via (cos, sin). 

6. `ts_holiday`: Peak normalized holiday count on a given ISO week-of-year. 

7. `ts_discount`: Discount curve over `timestep_horizon`; z-scored using 
   train-set mean/std.
  
8. `attr_price`: Static numeric attribute; z-scored using train-set mean/std.

9. `attr_retail`, `attr_category`, `attr_color`, `attr_fabric`: Static 
   categorical SKU attributes mapped to integer IDs using train-only vocab. IDs
   start at 1; index 0 is reserved for values unseen during training.


USAGE
--------------------------------------------------------------------------------
# `fit()` (see "train.py") creates a `datamodule` instance for training; to 
# .. manually create and inspect the `datamodule`: 

    from config import ArgConfig
    from dataset import DatasetManager
    
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
    
    datamodule = DatasetManager(args_next_week)
    datamodule.setup()

"""


import os
import numpy as np
import pandas as pd

import torch                                           # Template for `datamodule` object; a single container for all 
from lightning.pytorch.core import LightningDataModule # .. data logic; complies w/`Trainer` procedures
from torch.utils.data import DataLoader, Dataset # Template for constructing a custom dataset class
import torchvision.transforms.v2 as img_transforms # To build the image processing pipeline

from PIL import Image, ImageFile # To manipulate image files 
ImageFile.LOAD_TRUNCATED_IMAGES = True # Error raised if loaded image is corrupted; lets the image load and pads missing
                                       # .. parts w/black pixels; user warning included in `DatasetManager.setup()`

from utils import DataScaler # Helper to standardize numerical input data; instantiated in `DatasetManager`


# DATASET MANAGER CLASS
# ======================================================================================================================
class DatasetManager(LightningDataModule):
    def __init__(self, 
                 args, # Experiment-wide hyper-parameter and paths config object, `ArgConfig`; see "config.py"
                 ):
        super().__init__() # Must run parent `init` to setup internal registries
        
        self.args = args
       
          
    # TRAINER C0NTRACT METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    # `DatasetManager` is passed to `Trainer` as a `datamodule` (see "train.py"). `Trainer` expects the following 
    # .. methods to setup the datasets and build the train/val DataLoaders at the _correct_ point in its lifecycle
    def setup(self, stage = None):        
        # `Trainer` exposes `fit`, `validate`, `test`, and `predict` methods; each calls `setup(stage = "<method>")`; 
        # .. we call `trainer.fit()` only (see "train.py"), which contains both train and val loops, both belonging to 
        # .. `stage = "fit"`. We retain `stage` arg to uphold `Trainer` contract but do not branch on `stage` since 
        # .. we've got only one train/val build; `stage = None` applies only when `setup()` is called manually.

        # User warning for the import-time `PIL` flag at the top of this file
        if ImageFile.LOAD_TRUNCATED_IMAGES:
                print("WARNING: Corrupted images will be padded and loaded (see `PIL` in 'dataset.py').")
                
        self.sales_df, self.pnd_df, self.holidays_df = self.load_data()
    
        self.train_i, self.val_i = self.load_split_indices() # Load train/val index splits from disk
        
        # Partition raw data into train/val slices
        # NOTE: Sample rows line up one‐to‐one between raw datasets; index slicing retains this alignment
        self.sales_train_df = self.sales_df.iloc[self.train_i].reset_index() # Reset for `__getitem__` indexing; see
        self.sales_val_df   = self.sales_df.iloc[self.val_i].reset_index()   # .. `DataProcessor` below
        self.pnd_train_df   = self.pnd_df.iloc[self.train_i].reset_index()
        self.pnd_val_df     = self.pnd_df.iloc[self.val_i].reset_index()

        # NOTE: Compute train‐only vocabularies and statistics to avoid leaking val-set numerical stats (e.g., price 
        # .. mean/std) and categorical values (e.g., a new retail store) into training; both vocab and stats are dicts 
        # .. and (1) are retrieved by the model from `self.trainer.datamodule`, and (2) are saved in .ckpt s.t. at 
        # .. "predict" stage we would process input data and invert `y_hat` output to original scale using train-set 
        # .. transforms (see `Jimmy.setup()` in "model.py" for both)
        self.train_categ_attr_spec  = self.compute_categ_attr_spec(self.sales_train_df)
        self.train_numer_data_stats = self.compute_numer_stats(self.sales_train_df, self.pnd_train_df)

        if self.args.debug: self.debug_mode() # Truncates train/val indices to [:256] (two 128-size mini-batches) so
                                              # .. `Trainer` pipelines run fast

        self.train_processed = self.process_data(self.sales_train_df, self.pnd_train_df) # Runs `DataProcessor`
        print(f"Processed train set (`ts_Y.shape`: {self.train_processed.ts_Y.shape}).")                                         
        self.train_dloader   = self.build_dataloader(self.train_processed, 
                                                     shuffle = True, # Samples are ordered by "release_date", so batch
                                                     )               # .. samples belong to adjacent time periods; 
        print("Built train DataLoader.\n")                           # .. shuffle samples each epoch to mix periods, so
                                                                     # .. batch's loss gradient better approximates that
                                                                     # .. of the full dataset
        
        self.val_processed = self.process_data(self.sales_val_df, self.pnd_val_df)
        print(f"Processed validation set (`ts_Y.shape`: {self.val_processed.ts_Y.shape}).") 
        self.val_dloader   = self.build_dataloader(self.val_processed)
        print("Built validation DataLoader.\n")
        
        return None
            
            
    # `Trainer` calls `datamodule.train_dataloader()` and `datamodule.val_dataloader()`; this prevents having to 
    # .. manually pass distinct dataloaders to `trainer.fit` in "train.py"
    def train_dataloader(self):
        return self.train_dloader

    def val_dataloader(self):
        return self.val_dloader


    # CLASS METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    def load_data(self):
        # Loads "sales", "holidays", "price_discount_series" CSVs from paths defined in `config.ArgConfig`
        # NOTE: If file not found, `pd.read_csv` raises "FileNotFoundError"
        sales_df    = pd.read_csv(self.args.sales_fpath, 
                                  parse_dates = ["release_date"],
                                  index_col   = 'Unnamed: 0',     # Sets 'Unnamed: 0' col of presaved indices as 
                                  )                               # .. DataFrame index; verified row-wise `ext_code` 
        pnd_df      = pd.read_csv(self.args.price_discount_fpath) # .. alignment w/"price_discount_series"
        holidays_df = pd.read_csv(self.args.holidays_fpath,
                                  parse_dates = ["date"]
                                  ) 
        
        return sales_df, pnd_df, holidays_df
    

    def load_split_indices(self):
        # Loads CSV with pre-computed train/val split indices from paths defined in `config.ArgConfig`
        if not os.path.isfile(self.args.train_val_idxs_fpath):
            raise FileNotFoundError("CSV with train/val slice indices not found or path missing; see `help(ArgConfig)` for slice info and set path via `ArgConfig(train_val_split_idxs_fpath = FULL_PATH)`.\n")
            
        train_val_idxs_df = pd.read_csv(self.args.train_val_idxs_fpath)    
        train_i           = train_val_idxs_df.loc[:, "train_i"].dropna()
        val_i             = train_val_idxs_df.loc[:, "val_i"].dropna()
        print("\nTrain and validation set indices loaded.\n")

        return train_i, val_i
    
    
    def compute_categ_attr_spec(self, dataset):
        # For each categorical attribute, build a train‐only vocab and compute `num_embeddings` (passed to 
        # .. `AttributeEncoder` as `datamodule.train_categ_attr_spec` in "model.py"). The actual mapping from 
        # .. categorical values to integers takes place in `DataProcessor` below
        categ_attr_spec = {}
        for attr_name in ["retail", "category", "color", "fabric"]:
            train_uniques = dataset[attr_name].unique().tolist() # List of distinct categ values in an attr
            categ_attr_spec[attr_name] = {"uniques": train_uniques,
                                          "num_embeddings": len(train_uniques) + 1,
                                          } # `+1` is a placeholder for extra vocab element to which val-set 
                                            # .. categorical values unknown at model training are mapped
                                            
        return categ_attr_spec
    

    def compute_numer_stats(self, sales_df, pnd_df):
        # Compute train‐only `mean`, `std` to be passed to `DataScaler` (see "utils.py")
        # NOTE: Prices in dataset are normalized; still we standardize s.t. all input features are on a comparable scale
        
        # Get data
        end_col = str(self.args.timestep_horizon - 1)             # Each sales/discounts curve comprises 12 weeks w/cols 
        sales_data    = sales_df.loc[:, "0" : end_col].to_numpy() # .. labeled from "0" through "11"; end col is  
        discount_data = pnd_df.loc[:, "0" : end_col].to_numpy()   # .. `timestep_horizon` aware
        price_data    = pnd_df.loc[:, "price"].to_numpy()
        
        numer_data_stats = {}
        numer_data_stats["sales"]    = {"mean": sales_data.mean(),    "std": sales_data.std(ddof = 1)}      
        numer_data_stats["discount"] = {"mean": discount_data.mean(), "std": discount_data.std(ddof = 1)}
        numer_data_stats["price"]    = {"mean": price_data.mean(),    "std": price_data.std(ddof = 1)}
        # NOTE: `ddof = 1` to compute sample and not population std
       
        return numer_data_stats


    def debug_mode(self):
        self.sales_train_df = self.sales_train_df.iloc[:256].reset_index()
        self.sales_val_df   = self.sales_val_df.iloc[:256].reset_index()
        self.pnd_train_df   = self.pnd_train_df.iloc[:256].reset_index()
        self.pnd_val_df     = self.pnd_val_df.iloc[:256].reset_index()
        
        
    def process_data(self, sales_slice_df, pnd_slice_df):
        processed_data = DataProcessor(self.args,
                                       sales_slice_df, 
                                       pnd_slice_df,
                                       self.holidays_df,
                                       self.train_categ_attr_spec,  # Train-set transforms applied to both train and
                                       self.train_numer_data_stats, # .. validation sets
                                       )
        
        return processed_data


    def build_dataloader(self, processed_dataset, shuffle = False):
        # `persistent_workers`: Retains workers across epochs to cut per-epoch startup overhead (by default workers are 
        # .. re-spawned each epoch); requires `num_workers > 0`, else raises `ValueError`. NOTE: If `True`, worker RNG 
        # .. state persists across epochs so potential stochastic augments _inside_ workers would continue from prior 
        # .. state; this does not affect the sampler’s per-epoch data shuffle
        dataloader = DataLoader(processed_dataset,                   
                                batch_size         = self.args.batch_size,
                                shuffle            = shuffle,
                                num_workers        = self.args.cpu_workers,
                                persistent_workers = self.args.cpu_workers > 0, # Boolean; see above
                                )
        
        return dataloader
    
   
# DATA PR0CESS0R CLASS
# ======================================================================================================================
class DataProcessor(Dataset):
    def __init__ (self,
                  args, 
                  sales_slice_df,
                  pnd_slice_df,
                  holidays_df,
                  categ_attr_spec,
                  numer_data_stats,
                  ): 
        super().__init__() # Defensive as contract simply requires implemented `__len__` and `__getitem__` methods
        
        self.sales_slice_df = sales_slice_df # Passed to `__len__` via `self` as `__len__` can’t accept extra arguments
        
        # NOTE: Processed non-image tensors are stored below and looked up directly by `__getitem__(idx)`; images, which 
        # .. use more RAM, are loaded and transformed on-the-fly as a batch is assembled in `__getitem__`
        self.img_dir       = args.img_dir # `img_dir` is used in `process_img_raw` method; `process_img_raw` is called 
                                          # .. within `__getitem__`; since `__getitem__(idx)` can’t accept extra     
                                          # .. arguments, `img_dir` is passed to `process_img_raw` via `self`
        self.transform_img = self.build_transform_img_pipeline() # Assign `Compose` callable to reuse in 
                                                                 # .. `process_img_raw` w/o rebuilding
                                                                 
        # From "release_date" col in `sales_slice_df`, for each sample, build per-timestep ISO week and ISO year grids 
        # .. of shape (Epoch Size, `timestep_horizon`); used by `process_calendar_weeks` and `process_holidays` methods
        calendar_weeks_grid_wrapped, calendar_years_grid_offset = self.build_week_year_grids(args)

        # Create `utils.DataScaler` instances from train‐only `mean, std` to standardize raw data
        sales_scaler     = DataScaler(numer_data_stats["sales"]["mean"],    numer_data_stats["sales"]["std"])
        discounts_scaler = DataScaler(numer_data_stats["discount"]["mean"], numer_data_stats["discount"]["std"])
        prices_scaler    = DataScaler(numer_data_stats["price"]["mean"],    numer_data_stats["price"]["std"])
        
        
        # PR0CESSED N0N-IMAGE TENS0RS
        # --------------------------------------------------------------------------------------------------------------
        # NOTE: "ES" stands for epoch size (sample num in a data slice) and "BS" for batch size
        # NOTE: `Y_star_zero` and `ts_Y_star_mask` below are model-logic artifacts constructed at dataset processing,  
        # .. deliberately violating model/data separation to prevent on-the-fly batch assembly in `Jimmy.forward` and 
        # .. to confine it to the DataLoader; for model-logic, see for-loop counter step "j = 0" in `Jimmy.__doc__`
        self.ts_Y               = self.process_sales(args, sales_scaler) # (ES, 12)
        self.Y_star_zero        = torch.zeros(self.ts_Y.shape[0], 1) # (ES, 1); See above
        self.ts_Y_star_mask     = self.build_ts_Y_star_mask(args) # (ES, 12); See above
        
        self.ts_in_market_weeks = self.build_in_market_week(args) # (ES, 12)
        self.ts_calendar_weeks  = self.process_calendar_weeks(calendar_weeks_grid_wrapped) # (ES, 12, 2)
        self.ts_holidays        = self.process_holidays(holidays_df, 
                                                        calendar_weeks_grid_wrapped, 
                                                        calendar_years_grid_offset,
                                                        ) # (ES, 12)
        self.ts_discounts       = self.process_discounts(args, pnd_slice_df, discounts_scaler) # (ES, 12)
        
        self.attr_prices     = self.process_attr_prices(pnd_slice_df, prices_scaler) # (ES)
        
        self.attr_categories = self.process_categ_attr("category", categ_attr_spec) # (ES)
        self.attr_colors     = self.process_categ_attr("color",    categ_attr_spec) # (ES)
        self.attr_fabrics    = self.process_categ_attr("fabric",   categ_attr_spec) # (ES)
        self.attr_retails    = self.process_categ_attr("retail",   categ_attr_spec) # (ES)
        
    
    # IMAGE METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    def center_Pad(self, img_RGB):
        # Center pad to 224 x 224
        width, height = img_RGB.size # PIL method returns a `(width, height)`
        
        if width == 224 and height == 224: # Already 224 x 224
            return img_RGB
        
        width_delta  = 224 - width
        height_delta = 224 - height
        
        left           = width_delta // 2 # If odd delta, extra pixel goes to `right` (and `bottom`) via subtraction
        right          = width_delta - left
        top            = height_delta // 2
        bottom         = height_delta - top

        pad_transform  = img_transforms.Pad((left, top, right, bottom)) # No-signal (background) value of alpha = 0 in 
        img_RGB_padded = pad_transform(img_RGB)                         # .. SKU images converts to RGB black (0, 0, 0) 
                                                                        # .. upon `Image.open`; this matches the default 
                                                                        # .. `Pad` fill of (0, 0, 0)
        return img_RGB_padded
    
    
    def build_transform_img_pipeline(self):
        # A frozen pre-trained ResNet101 encodes SKU images (see `Jimmy` in "model.py"). We use a custom `Compose` image
        # .. transform pipeline (i.e., not `ResNet101_Weights.IMAGENET1K_V2.transforms`; e.g. no `CenterCrop` as SKUs 
        # .. reach edges) that adapts Visuelle images to ResNet expected input shape and distribution:
        # .. 1. `Compose` is a callable object that stores a list of transform callables; accepts `img` as an argument, 
        # .. .. `__call__(img)`, and sequentially passes it to stored callables;
        # .. 2. `Resize` first resizes the short side to `size`, then `size` is overruled s.t. the long side is resized
        # .. .. to `max_size`; this results in the short side <= `size`; `max_size` only supported if `size` is passed;
        # .. .. aspect ratio (AR) is preserved; default `InterpolationMode.BILINEAR`, as expected by ResNet;
        # .. .. # NOTE: `Resize` requires strict `max_size` > `size`, hence, `size` = `max_size` - 1: for most ARs,
        # .. .. final long edge = 224 and short is scaled to AR (i.e., step 2 overrides step 1); for square input, final 
        # .. .. Resize is 223 × 223 (0.9% loss), then center-paded to 224×224.
        # .. 3. `center_Pad` computes left, top, right, bottom to center pad to 224 x 224
        # .. 4. `ToImage` converts and reshapes (Height, Width, Channels) PIL object to (C, H, W) tensor;
        # .. 5. `ToDtype(torch.float32, scale = True)` rescales [0, 255] RGB interval to a [0.0, 1.0];
        # .. 6. `Normalize` in fact standardizes (an affine tranform; no distortion) SKU image tensors of shape 
        # .. .. (3, 224, 224) to the pixel-value distributions expected by ResNet weights.
        transform_img = img_transforms.Compose([img_transforms.Resize(size = 223, max_size = 224), # 223: see NOTE above
                                                self.center_Pad, # Assigned as a callable
                                                img_transforms.ToImage(), 
                                                img_transforms.ToDtype(torch.float32, scale = True), 
                                                img_transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                         std  = [0.229, 0.224, 0.225],
                                                                         ),
                                                ])
        return transform_img # (3, 224, 224)
    
    
    def process_img_raw(self, idx):
        img_path   = self.sales_slice_df['image_path'][idx] # Retrieve image path
        img_fpath  = os.path.join(self.img_dir, img_path) # Compile image fullpath
        img_RGB    = Image.open(img_fpath).convert("RGB") # Open ".png"; convert to RGB
        img_tensor = self.transform_img(img_RGB) # Apply the image pipeline (see `build_transform_img_pipeline`); 
                                                 # .. `Compose` takes img as arg: `__call__(img)`
        return img_tensor # (3, 224, 224); Return one processed image tensor per call


    # ATTRIBUTE METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    def process_attr_prices(self, pnd_slice_df, prices_scaler):
        # NOTE: In the dataset, the "price" column has been transformed to obscure the original values
        attr_prices = pnd_slice_df.loc[:, "price"]
        attr_prices = torch.Tensor(attr_prices)
        attr_prices = prices_scaler.fwd(attr_prices)
        
        return attr_prices # (ES)
   
   
    def process_categ_attr(self, attr_name, categ_attr_spec):
        attr_col    = self.sales_slice_df.loc[:, attr_name] # Retrieve attribute column
        uniques     = categ_attr_spec[attr_name]["uniques"]
        key_val_map = {key: (val + 1) for val, key in enumerate(uniques)} # Assign int to each token, starting at 1; as
        attr        = attr_col.map(key_val_map).fillna(0).astype(int)     # .. such, 0 is reserved for validation set  
        attr        = torch.LongTensor(attr)                              # .. tokens UNK during training: `fillna(0)`
        
        return attr
    
    
    # TIME-SERIES METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    def process_sales(self, args, sales_scaler):
        # Each sales curve comprises 12 weeks w/cols labeled from "0" through "11"; end col is `timestep_horizon` aware
        ts_Y = self.sales_slice_df.loc[:, "0" : str(args.timestep_horizon - 1)].to_numpy()
        ts_Y = torch.Tensor(ts_Y)
        ts_Y = sales_scaler.fwd(ts_Y)

        return ts_Y # (ES, 12)    
    

    def build_ts_Y_star_mask(self, args):
        # Binary delimiter tensor of shape (ES, timestep_horizon): {i = 0} =  1.0,  {i > 0} = 0.0; see for-loop counter  
        # .. step "j = 0" in `Jimmy.__doc__`. NOTE: In future iterations, `ts_Y_star_mask` could mark in-stock versus 
        # .. out-of-stock periods, distinguishing zeros due to no demand from zeros due to lack of inventory
        ts_Y_star_mask       = torch.zeros((self.ts_Y.shape[0], args.timestep_horizon)) # Outputs float by default
        ts_Y_star_mask[:, 0] = 1.0
        
        return ts_Y_star_mask
    

    def build_in_market_week(self, args):
        # Commercial week-age of a SKU at the end of the predicted week; week-age in the week of release is 1
        a = torch.arange(1, args.timestep_horizon + 1).unsqueeze(0) # (1, 12); [[1, 2, ... , timestep_horizon]]
        b = torch.zeros((self.ts_Y.shape[0], 1)) # (ES, 1)
        in_market_weeks = (a + b) / args.timestep_horizon # (ES, 12); broadcast and peak normalize
        
        return in_market_weeks 
    

    def build_week_year_grids(self, args):
        # Build shape (Epoch Size, `timestep_horizon`) week and year time-series grids:
        # .. `calendar_weeks_grid_wrapped`: ISO week-of-year for each time-step, starting at the SKU’s "release_date"
        # .. .. week and rolling forward `timestep_horizon` steps, wrapped into the 1–52 ISO week-of-year range.
        # .. `calendar_years_grid_offset`: ISO year aligned to those weeks.
        # Summary of transformations:
        # .. Suppose `release_years[i] = 2019`, `release_weeks[i] = 51` and `timestep_horizon = 4` for sample "i", then:
        # .. .. `release_years[i, None]            = [2019]`
        # .. .. `release_weeks[i, None]            = [51]`
        # .. .. `weeks_offset                      = [[ 0,  1,  2,  3]]` (to roll `release_weeks[:]` forward)
        # .. .. `calendar_weeks_grid_offset[i, :]  = [51, 52, 53, 54]` (broadcast + offset; spills past week 52)
        # .. .. `years_offset[i, :]                = [ 0,  0,  1,  1]` (weeks >52 imply next ISO year)
        # .. .. `calendar_years_grid_offset[i, :]  = [2019, 2019, 2020, 2020]` (broadcast + offset `release_years[i]`)
        # .. .. `calendar_weeks_grid_wrapped[i, :] = [51, 52,  1,  2]` (wrap into 1–52)
        
        release_dates = self.sales_slice_df.loc[:, "release_date"]
        
        # Convert "release_date" to ISO week-of-year and year; no day info as step is weekly
        release_weeks = release_dates.dt.isocalendar().week.values.to_numpy() # Cast to `np.array` to enable 
        release_years = release_dates.dt.isocalendar().year.values.to_numpy() # .. broadcasting
        
        if np.any(release_weeks == 53): 
            print("WARNING: ISO calendar week 53 (e.g. 2020-12-31) detected in release dates. This case is rare and not present in the original dataset, so the current implementation wraps week 53 to week 1. Code update recommended.")
            
        # WEEKS
        # --------------------------------------------------------------------------------------------------------------
        # Build a shape (1, `timestep_horizon`) array of week offsets: [[0, 1, ... , `timestep_horizon` - 1]]
        weeks_offset                = np.expand_dims(np.arange(args.timestep_horizon), axis = 0)
        
        # Broadcast `release_weeks` through `timestep_horizon` and offset to form the unwrapped week grid: 
        # .. (ES, 1) + (1, `timestep_horizon`) -> (ES, `timestep_horizon`); 
        # .. Row `i`: [release_weeks[i] + 0, ..., release_weeks[i] + (`timestep_horizon` - 1)]
        calendar_weeks_grid_offset  = np.expand_dims(release_weeks, axis = 1) + weeks_offset

        # Wrap >52 week values into the 1–52 range (see "process_release_weeks+holidays.ipynb" for arithmetic details)
        calendar_weeks_grid_wrapped = (calendar_weeks_grid_offset - 1) % 52 + 1 
        
        # YEARS
        # --------------------------------------------------------------------------------------------------------------
        # Detect year rollover from the unwrapped week grid: vals in 1–52 map to 0 (same year); >52 to 1 (next year)
        years_offset               = (calendar_weeks_grid_offset - 1) // 52
        
        # Broadcast `release_years` through `timestep_horizon` and apply the year offsets:
        # .. (ES, 1) + (ES, `timestep_horizon`) -> (ES, `timestep_horizon`)
        calendar_years_grid_offset = np.expand_dims(release_years, axis = 1) + years_offset
        # --------------------------------------------------------------------------------------------------------------
        
        return calendar_weeks_grid_wrapped, calendar_years_grid_offset
    

    def process_calendar_weeks(self, calendar_weeks_grid_wrapped):
        # Map each of 52 ISO weeks onto the unit circle and compute each week's polar angle (r = 1), so weeks 52 and 1 
        # .. are adjacent in the cycle; year info omitted since it carries no insight into sales seasonality; day info
        # .. omitted as forecast is weekly
        ISO_week_theta     = 2 * np.pi * calendar_weeks_grid_wrapped / 52
        
        # Convert polar to a cartesian coordinate pair; range: [-1, 1]
        cos_week, sin_week = np.cos(ISO_week_theta), np.sin(ISO_week_theta)
        
        # (ES, `timestep_horizon`, 2); stack the cartesian coordinate pair s.t. each forecast week has two channels
        ts_calendar_weeks  = np.stack([cos_week, sin_week], axis = 2)
            
        return torch.Tensor(ts_calendar_weeks)
        

    def process_holidays(self, holidays_df, calendar_weeks_grid_wrapped, calendar_years_grid_offset):
        # Build a weekly holiday intensity curve (peak normalized holiday count in a given week) for each sample
        
        holiday_dates = holidays_df.loc[:, "date"] # (46, 1); List of holiday dates in the train/val time interval
        holiday_weeks = holiday_dates.dt.isocalendar().week.values # Convert date to ISO week-of-year
        holiday_years = holiday_dates.dt.isocalendar().year.values # To ISO year; no day info as time-steps are weekly
        
        # Encode each holiday date as a YYYYWW key (e.g., 2019-W52 -> 201952)
        holiday_YYYYWW         = holiday_years * 100 + holiday_weeks
        
        # NOTE: Forecast granularity is weekly, so in a given year, multiple distinct holiday dates can fall in the same
        # .. week. As a result, these holiday dates share the same YYYYWW key, so YYYYWW duplicates are expected
        holiday_YYYYWW_freq    = holiday_YYYYWW.value_counts() # Frequency table of YYYYWW keys
        holiday_YYYYWW_weights = (holiday_YYYYWW_freq / holiday_YYYYWW_freq.max()) # Peak normalize YYYYWW frequencies
        

        # Combine ISO year and week grids into a single YYYYWW key grid; for sample `i` w/`release_years[i] = 2019`, 
        # .. `release_weeks[i] = 51`, and `timestep_horizon = 4` we get:
        # .. .. `calendar_years_grid_offset[i, :]  = [2019, 2019, 2020, 2020]`
        # .. .. `calendar_weeks_grid_wrapped[i, :] = [  51,   52,    1,    2]`
        # .. .. `calendar_YYYYWW_grid[i, :]        = [201951, 201952, 202001, 202002]`
        calendar_YYYYWW_grid  = (calendar_years_grid_offset * 100 + calendar_weeks_grid_wrapped)
        
        #  Flatten `calendar_YYYYWW_grid`; apply `holiday_YYYYWW_weights` to `calendar_YYYYWW_grid` keys ; key N/A means
        # .. no holiday that week i.e., apply weight of 0 (`fillna(0)`) 
        holidays_weights_flat = holiday_YYYYWW_weights.reindex(calendar_YYYYWW_grid.ravel()).fillna(0)
        holidays_weights_grid = holidays_weights_flat.to_numpy().reshape(calendar_YYYYWW_grid.shape) # Restore grid’s 
        ts_holidays           = torch.Tensor(holidays_weights_grid)                                  # .. original shape
        
        return ts_holidays
    
    
    def process_discounts(self, args, pnd_slice_df, discounts_scaler):
        # Each discounts curve comprises 12 weeks w/cols labeled from "0" through "11"; end col is `timestep_horizon` 
        # .. aware
        ts_discounts = pnd_slice_df.loc[:, "0" : str(args.timestep_horizon - 1)].to_numpy()
        ts_discounts = torch.Tensor(ts_discounts)
        ts_discounts = discounts_scaler.fwd(ts_discounts) 
        
        return ts_discounts # (ES, 12)
    

    # SUPERCLASS C0NTRACT METH0DS 
    # ------------------------------------------------------------------------------------------------------------------
    # `DataProcessor` inherits from `torch.utils.data.Dataset`, which defines `__len__` and  `__getitem__(idx)` methods 
    # .. that must be implemented by the subclass that builds the custom dataset
    def __len__(self): # Returns row num in the train/val data slice
        return len(self.sales_slice_df)  

    def __getitem__(self, idx): 
        # Returns processed non-image tensors (stored in `__init_`) for a single sample at position `idx`; to save RAM, 
        # .. associated raw image is loaded and processed on-the-fly as the batch assembles
        
        img = self.process_img_raw(idx) # (3, 224, 224)
        
        ts_y              = self.ts_Y[idx] # (12)
        y_star_zero       = self.Y_star_zero[idx] # (1)
        ts_y_star_mask    = self.ts_Y_star_mask[idx] # (12)
        
        ts_in_market_week = self.ts_in_market_weeks[idx] # (12)
        ts_calendar_week  = self.ts_calendar_weeks[idx] # (12, 2)
        ts_holiday        = self.ts_holidays[idx] # (12)
        ts_discount       = self.ts_discounts[idx] # (12)
        
        attr_price    = self.attr_prices[idx] # ()
        
        attr_retail   = self.attr_retails[idx] # ()
        attr_category = self.attr_categories[idx] # ()
        attr_color    = self.attr_colors[idx] # ()
        attr_fabric   = self.attr_fabrics[idx] # ()
        
        # Samples are emitted in the exact tuple order expected by `Jimmy.forward()` (see "model.py")
        return (img, 
                ts_y, y_star_zero, ts_y_star_mask, 
                ts_in_market_week, ts_calendar_week, ts_holiday, ts_discount,
                attr_price,
                attr_retail, attr_category, attr_color, attr_fabric,
                )