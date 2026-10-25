"""
model.py
================================================================================
Defines the model, `Jimmy`, and its modules: `AttributeEncoder`, `ImageEncoder`,
`AdditiveAttention`, `LockedDropout`. For more info, see `help(Jimmy)`, 
`help(AdditiveAttention)`, and `help(LockedDropout)`.


`Jimmy` (the model)
--------------------------------------------------------------------------------
* This model constitutes a multimodal nonlinear autoregressive forecaster 
  w/eXogenous covariates, implemented w/GRU. At every decoder step, the model 
  fuses:
  - Attended SKU image features.
  - Prior-week sales: observed or predicted; first step uses a synthetic zero
    because no prior-week sales value is available.
    - Cold-start delimiter: `y_star_mask_t`; 1 for that synthetic input,
      0 thereafter.
  - Forecast-week covariates: `ts_in_market_week` (time in market),
    `ts_calendar_week` (cyclical week of year), `ts_holiday` (holiday 
    intensity), and `ts_discount` (price discount).
  - Categorical attributes: "retail", "category" (garment type), "color", 
    "fabric".
  - Numeric attribute: "price".
* Returns: sales predictions, aligned targets, and batch size.

1. Rollout is organized around distinct variable of forecast time (`t`), tensor
   position (`i`), and loop execution (`j`); See 
   `print(Jimmy._on_loop_mechanics_)`.
   
2. Static image encoded once per forward pass; channel attention recomputed
   each step using the previous hidden state and descriptive attributes,
   allowing visual emphasis to change throughout the rollout.

3. All static attributes enter fusion at every step; the descriptive subset
   (category, color, fabric) also guides image attention.

4. Probabilistic teacher forcing selects observed vs predicted prior-week sales
   as autoregressive input. We sever the extra gradient path created by
   autoregressively consuming predictions. See `print(Jimmy._on_detach_)`.

 
`AttributeEncoder`
--------------------------------------------------------------------------------
* Encodes time-invariant SKU attributes: 
  - Categorical: "retail", "category" (garment type), "color", and "fabric";
  - Numeric: "price".
* Returns: 5-tuple of tensor encodings.

1. Exposes `attr_embedding_dims` so `Jimmy.setup()` can size downstream modules.
2. `padding_idx = 0`: UNK (index 0) embedding injects noise; hence zeroed. 
3. `scale_grad_by_freq = True`: Embeddings of frequent vs infrequent attribute 
    tokens "drift" unevenly through loss landscape; hence embedding weights 
    update using the mean (not sum) of the attribute token's gradients.
4. `attr_embedding_dims` adjusted s.t. downstream vector sizes align to 
    multiples of 8 for GPU-friendly matrix multiply.


`ImageEncoder`
--------------------------------------------------------------------------------
* Extracts SKU image features using a frozen, pretrained ResNet101 backbone that
  is truncated before the global spatial average pool (C, H, W) -> (C, 1, 1) and 
  the 1000-class FC head;
* Returns: (BS, C, H x W) tensor (flattened spatial grid).

1. Probes encoder output dims using a processed training image to automatically
   size downstream attention and image-projection layers.
  

`AdditiveAttention`
--------------------------------------------------------------------------------
* Bahdanau-style channel attention; reweights static `img_embedding` at each
  decoder step, conditioned on static descriptive SKU attributes
  (`attr_IMG_DESCRIPTOR`) and previous hidden state (`h_t_minus_1`).
* Returns: (BS, C, H x W) channel-weighted image features.

1. Standard ResNet's `AdaptiveAvgPool2d` collapses each channel's spatial grid,
   (BS, 2048, 7, 7), to one average value, (BS, 2048, 1, 1); this constitutes
   channel readout. We insert attention before this averaging to reweight
   the channels based on within-channel spatial significance. The choice
   of channel attention, rather than the spatial attention canonically used
   in image encoding, is motivated by standard ResNet's architectural
   emphasis on channel features.
   - For mathematical exposition see `help(AdditiveAttention)`.
    
2. We adapt standard ResNet readout structure:

   - Spatial pooling:
       ResNet: `AdaptiveAvgPool2d` + `Flatten`;
               (BS, 2048, 7, 7) -> (BS, 2048, 1, 1) -> (BS, 2048)
       Jimmy:  `img_avg_pool`; (BS, 2048, 49) -> (BS, 2048)
     
   - Linear projection:
       ResNet: 1000-class FC; (BS, 2048) -> (BS, 1000)
       Jimmy:  `img_fc_reduction`; (BS, 2048) -> (BS, 128)
               Produces image features rather than class scores.
            
               
`LockedDropout`
--------------------------------------------------------------------------------
* Problem: Standard `nn.Dropout()` resamples its mask on each decoder step `j`.
  This per-step mask re-sampling stochastically censors sample features 
  (activations) on every step `j`: the decoder then must ~model not only the 
  sequence dynamics, but also a time-varying pattern of missing coordinates, 
  i.e., noise. 
* Solution: By contrast, `LockedDropout` creates a single mask at `j = 0` and 
  reuses it on each step, until reset in `Jimmy.forward` for the next batch. 
  This way, each sample retains a consistent _thinned_ feature subspace 
  throughout its rollout.
  
  For mechanics see `help(LockedDropout)`.


USAGE (see "run.py" for runtime setup)
--------------------------------------------------------------------------------
    import multiprocessing
    import torch
    
    from config import ArgConfig
    from train import fit
    
    CPU_NUM = multiprocessing.cpu_count()

    if torch.backends.mps.is_available(): DEVICE = "mps"
    elif torch.cuda.is_available(): DEVICE = "cuda"
    else: DEVICE = "cpu"

    args_next_week = ArgConfig(forecast_mode     = "next-week",
                               train_val_idxs_fpath = os.path.join(PROCESSED_DIR, "train_val_i_time-split.csv"),
                               cpu_workers          = CPU_NUM,
                               device               = DEVICE,
                               sales_fpath          = SALES_FPATH,
                               price_discount_fpath = PRICE_DISCOUNT_FPATH,
                               holidays_fpath       = HOLIDAYS_FPATH,
                               img_dir              = IMG_DIR,
                               logs_ckpts_dir       = LOGS_CKPTS_DIR,
                               )

    fit(args = args_next_week)
"""


import random # For teacher forcing decision (`tf_decision`) draw

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision.models import resnet101, ResNet101_Weights

import utils as u


# HARDC0DED VARS
# ----------------------------------------------------------------------------------------------------------------------
# `ATTR_EMBEDDING_DIMS`: No definitive theory on `embedding_dim` for tabular (i.e., no semantic structure) data 
# .. (attributes "retail", "category", "color", and "fabric"). Thus, pick middle ground between two learning outcomes:
# .. 1. Fully unrelated tokens i.e., orthogonality, which necessitates `embedding_dim = num_embeddings` (in practice,  
# .. .. tokens with distinct purchase patterns are separated “far enough,” not orthogonally).
# .. 2. Full token overlap i.e., collinearity given `embedding_dim = 1` (in practice, similarly behaving tokens cluster 
# .. .. and are not colinear).
# To balance these outcomes, choose embedding dims around 0.4 * `num_embeddings`; adjust s.t. dims of downstream vectors
# .. that are first relevant to matrix multiply, `attr_IMG_DESCRIPTOR` and `fusion_cat_t`, are multiples of 8; intended
# .. for peak GPU throughput. Thus:
# .. .. retail:   110 + 1 UNK -> 40
# .. .. category: 27  + 1 UNK -> 12
# .. .. color:    10  + 1 UNK -> 4
# .. .. fabric:   58  + 1 UNK -> 24
# .. Therefore:
# .. .. `attr_EMBEDDING`:      81 = 40 + 12 + 4 + 24 + 1 (price)
# .. .. `attr_IMG_DESCRIPTOR`: 40 = 12 + 4 + 24
# .. .. `fusion_cat_t`:        216 = 81 + 7 (t-s) + 128 (img)
# NOTE: Consider using Welch bounds, however, the 2° slack from orthogonality is an unprincipled heuristic.
        
ATTR_EMBEDDING_DIMS = {"price" :    1,
                       "retail":   40,
                       "category": 12,
                       "color":     4,
                       "fabric":   24,
                       }


# `TS_DIM`: Time-series are mandatory input features; hence hardcoded. Sizes: 
# .. `ts_y` OR `y_hat_rollout`: 1
# .. `ts_y_star_mask`:          1
# .. `ts_in_market_week`:       1
# .. `ts_calendar_week`:        2 (cos, sin encoding)
# .. `ts_holiday`:              1 
# .. `ts_discount`:             1
     
TS_DIM = 7


# ATTRIBUTE ENC0DER 
# ======================================================================================================================
class AttributeEncoder(nn.Module):
    def __init__(self,
                 datamodule,
                 ):
        super().__init__()
    
        self.embedding_MODULES   = nn.ModuleDict() # Layers assigned to `nn.ModuleDict()` are registered with the model
        self.attr_embedding_dims = ATTR_EMBEDDING_DIMS # Declare hardcoded dims; used to size modules in `Jimmy.setup()` 
        
        for attr_name in ["retail", "category", "color", "fabric"]:
            attr_num_embeddings                 = datamodule.train_categ_attr_spec[attr_name]["num_embeddings"]
            attr_embedding_dim                  = self.attr_embedding_dims[attr_name]
            self.embedding_MODULES[attr_name]   = nn.Embedding(num_embeddings     = attr_num_embeddings,
                                                               embedding_dim      = attr_embedding_dim,
                                                               padding_idx        = 0, # See below
                                                               scale_grad_by_freq = True, # See below
                                                               )
            
            # `padding_idx = 0`: Val set tokens unknown in training are mapped to index 0 (see 
            # .. `DataProcessor.process_categ_attr()` in "dataset.py"). Index 0 never occurs in training, thus, 0's 
            # .. weight vector is random at val phase and injects noise; `padding_idx = 0` zeroes and freezes (no grads) 
            # .. this weight vector, so unknown tokens contribute a deterministic zero signal (i.e. are ignored) in val.

            # `scale_grad_by_freq = True`: The weight vector of a token that appears in the batch infrequently is not
            # .. updated by the sum of its gradients but by their mean, i.e., the sum is divided by that token’s 
            # .. frequency in the batch. As a result, frequent (head) and infrequent (tail) tokens drift across the loss
            # .. landscape more evenly; frequent tokens still get more updates, but not larger per-batch magnitudes.
        
        
    def forward(self, attr_price, attr_retail, attr_category, attr_color, attr_fabric): # (BS), (BS), (BS), (BS), (BS)

        def forward_helper(attr_name, attr_input):
            embedding_module = self.embedding_MODULES[attr_name] 
            attr_embedding   = embedding_module(attr_input) # NOTE: No dropout as adds noise; see "`dropout = False`" 
                                                            # .. comment for `self.fusion_mlp` in `Jimmy.__init__`
            return attr_embedding                           

        scalar_price       = attr_price.unsqueeze(1) # (BS) -> (BS, 1)
        
        embedding_retail   = forward_helper("retail",   attr_retail)   # (BS, 40)
        embedding_category = forward_helper("category", attr_category) # (BS, 12)
        embedding_color    = forward_helper("color",    attr_color)    # (BS, 4)
        embedding_fabric   = forward_helper("fabric",   attr_fabric)   # (BS, 24)
        
        attr_output        = (scalar_price, 
                              embedding_retail, 
                              embedding_category, 
                              embedding_color, 
                              embedding_fabric,
                              )
        
        return attr_output # ( (BS, 1), (BS, 40), (BS, 12), (BS, 4), (BS, 24) ); 81
    

# IMAGE ENC0DER
# ======================================================================================================================
class ImageEncoder(nn.Module):
    def __init__(self,
                 datamodule,
                 ):
        super().__init__()
        
        img_classifier = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2) # 81.9% top-1 accuracy
        
        # Backbone up to and inc. layer 4; (BS, 3, 224, 224) -> (BS, 2048, 7, 7) == (BS, channel_dims, spatial_dim_x,
        # .. spatial_dim_y); excludes global AvgPool2d across spatial dims, (C, H, W) -> (C, 1, 1), and linear FC to  
        # .. 1000 classes
        self.img_classifier_backbone = nn.Sequential(*list(img_classifier.children())[0:-2])

        self.img_classifier_backbone.eval() # `eval()`: `BatchNorm` uses stored mean/var batch stats; weights still 
                                            # ..  train; Dropout off (although no dropout in resnet101)
        
        # (BS, 2048, 7, 7) -> (BS, 2048, 49); flatten spatial grid for `AdditiveAttention`
        self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)
        
        # Used to size `AdditiveAttention.W_img` and `img_fc_reduction` in `Jimmy.setup()`
        self.img_embedding_dims = self.get_img_embedding_dims(datamodule) 
        
        
    def get_img_embedding_dims(self, datamodule):
        # Get `img_embedding` shape by passing an image sample, retrieved from `DatasetManager.train_processed`, to 
        # .. `ImageEncoder.forward`. Thus, downstream sizing is derived, not assumed in case of preprocessing/backbone 
        # .. changes. Assumes `datamodule` is built (i.e., `datamodule.setup()`); satisfied by module `setup` order of
        # .. `trainer.fit` in "trainer.py".
        img_probe         = datamodule.train_processed.__getitem__(0)[0] # Get image probe: (3, 224, 224)
        img_probe_batched = img_probe.unsqueeze(0) # Batchify image probe: (1, 3, 224, 224)
        with torch.no_grad():
            img_embedding_probe = self.forward(img_probe_batched) # (1, channel_dims, spatial_dims) == (1, 2048, 49)
            
        img_embedding_dims = (img_embedding_probe.shape[1], img_embedding_probe.shape[2]) # (2048, 49)
        return img_embedding_dims
        
    
    def forward(self, img):
        # `no_grad()`: MLP parameter tensors are constructed at `__init__`; the assoc. autograd graph is constructed at
        # .. `forward()`; `no_grad()` prevents graph construction; so ResNet101 cannot learn. 
        # NOTE: Don't use `inference_mode()` as backbone output tensor won't be eligibile for use in backward
        with torch.no_grad():
            img_features = self.img_classifier_backbone(img) # (BS, 3, 224, 224) -> (BS, 2048, 7, 7)
        
        img_embedding = self.flatten(img_features) # (BS, 2048, 7, 7) -> (BS, 2048, 49)

        return img_embedding  # (BS, 2048, 49)
    

# ADDITIVE ATTENTI0N 
# ======================================================================================================================
class AdditiveAttention(nn.Module): 
    def __init__(self, 
                 args, 
                 img_spatial_dims,
                 attr_img_desc_dim,
                 ):
        super(AdditiveAttention, self).__init__()
        # See `AdditiveAttention.__doc__` below

        # `attn_dim` < max would rank-limit input tensors (discard information) and likely degrade attention modulation 
        attn_dim = max(img_spatial_dims, attr_img_desc_dim, args.h_t_dim)

        self.W_img   = nn.Linear(img_spatial_dims,  attn_dim, bias = False)
        self.W_attr  = nn.Linear(attr_img_desc_dim, attn_dim, bias = False)
        self.W_h     = nn.Linear(args.h_t_dim,      attn_dim, bias = False)
        self.tanh    = nn.Tanh()
        self.V       = nn.Linear(attn_dim, 1, bias = False)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, img_embedding, attr_IMG_DESCRIPTOR, h_t_minus_1): # (BS, 2048, 49), (BS, 40), (1, BS, 128)
        h_t_minus_1 = h_t_minus_1.squeeze(0) # (num_layers, BS, hidden_size) -> (BS, hidden_size) == (BS, 128)

        img_embedding_W       = self.W_img(img_embedding) # (BS, 2048, 49) -> (BS, 2048, 128)
        attr_IMG_DESCRIPTOR_W = self.W_attr(attr_IMG_DESCRIPTOR) # (BS, attr_img_desc_dim) == (BS, 40) -> (BS, 128)
        h_t_minus_1_W         = self.W_h(h_t_minus_1) # (BS, h_t_dim) -> (BS, attn_dim) == (BS, 128) -> (BS, 128)
        
        # (BS, 2048, 128) + { (BS, 128) -> (BS, 1, 128) } + { (BS, 128) -> (BS, 1, 128) } -> (BS, 2048, 128)
        alignment_t = self.tanh(img_embedding_W + attr_IMG_DESCRIPTOR_W.unsqueeze(1) + h_t_minus_1_W.unsqueeze(1)) 
        energy_t    = self.V(alignment_t).squeeze(2) # (BS, 2048, 128) -> (BS, 2048, 1) -> (BS, 2048)
        alpha_t     = self.softmax(energy_t) # (BS, 2048)
        
        # { (BS, 2048) -> (BS, 2048, 1) } * (BS, 2048, 49) -> (BS, 2048, 49)
        img_attended_t = alpha_t.unsqueeze(2) * img_embedding # Hadamard
        
        return img_attended_t # (BS, 2048, 49)
    

AdditiveAttention.__doc__ = """
This `AdditiveAttention` implementation _mirrors_ standard ResNet’s channel emphasis and improves on plain
`AdaptiveAvgPool2d` by amplifying/suppressing channels (rather than pixels) before spatial information is collapsed 
later in `Jimmy`. The exposition below the explains the motivation for using channel (rather than the canonical spatial)
`AdditiveAttention` to encode the SKU image, and shows that this gate can be interpreted as a spatially dependent self-weighting of channel relevance:

  1.  `nn.Linear(49, 128)` returns `W_img` matrix of size (128, 49) == (attn_dim, img_spatial_dims); the input, 
      `img_embedding`, is of size (BS, 2048, 49); PyTorch computes matrix multiply as:
        `img_embedding` @ `W_img`^{transpose}        -> `img_embedding_W` == 
        (BS, 2048, 49)  @ { (128, 49) -> (49, 128) } -> (BS, 2048, 128);
  2.  For simplicity, assume `attn_dim = 1` (i.e., `W_img` is of size (1, 49)) and use an unbatched view (i.e., 
      `img_embedding` is a 2D matrix of size (2048, 49));
  3.  Move to textbook linear algebraic transform perspective by taking a transpose of the expression in step 1: 
        ( `img_embedding` @ `W_img`^{transpose} -> `img_embedding_W` )^{transpose} == 
        `W_img` @ `img_embedding`^{transpose}   -> `img_embedding_W`^{transpose}   == (1, 49) @ (49, 2048) -> (1, 2048)
  4.  The size (49, 2048) `img_embedding` matrix is comprised of 2048 distinct, size (1, 49) 'img_embedding_C' vectors. 
      That is, for each channel C, the 'img_embedding_C' vector is that channel’s 49-D spatial strip;
  5.  In `W_img` @ `img_embedding`^{transpose}, interpret each 'img_embedding_C' vector as a single (of 2048) "sample"
      that is transformed by one and the same size (1, 49) `W_img` matrix; `W_img` weighs the 49 `img_spatial_dims` and, 
      as such, is a spatial preference pattern that encodes where the signal matters (e.g., center-heavy, top-band, 
      left–right contrast);
  5.1 `W_img` is applied to every channel C, indicating that one and the same spatial preference pattern is shared by
      all channels, and thus applied to the entire image, and to every image sample. A corollary is that image samples 
      (i.e., SKUs) whose signal is off-center, a single center-heavy `W_img` spatial preference pattern will under-score 
      said SKU's informative region. This is mediated by setting `attn_dim = 128`, where each dim (i.e. each `W_img` 
      row) _can_ constitute a distinct spatial preference pattern. Note that `attn_dim = 128` also increases 
      `img_embedding_W` <> `h_t_minus_1_W` alignment granularity and capacity;
  6.  With `attn_dim = 1`, each scalar element in the size (1, 2048) `img_embedding_W` product is the summed (see the 
      arithmetic in step 3) energy (!= `energy_t`) of a channel’s 49-D spatial strip, 'img_embedding_C'. Then:
        a) Presuming no conditioning (i.e., `h_t_minus_1 = 0` and `attr_IMG_DESCRIPTOR = 0`);
        b) Since `nn.Linear(attn_dim, 1, bias = False)`, `V` is a scalar multiplier which, for simplicity, we ignore:
             `V` @ `img_embedding_W` -> `energy_t` == (1,1) @ (1, 2048) -> (1, 2048)  [linear transform perspective]
        c) Barring the nonlinearity of Tanh (in `alignment_t`) and Softmax (in `alpha_t`), we take 
           `alpha_t` ≈ `img_embedding_W` in ~`img_attended_t = alpha_t * img_embedding`
      We observe that `img_embedding_W` constitutes a _spatially dependent self-weighting_ of channel relevance. 
      Specifically:
        - Spatial dependence stems from `W_img` being a spatial preference pattern (see step 5);
        - "Self" stems from each channel's energy scalar in `img_embedding_W` being applied to that channel's own 
          'img_embedding_C':
            `img_attended_t` = `img_embedding` * 
                               `img_embedding_W` [ ~ `alpha_t`(`energy_t`(`alignment_t`(`img_embedding_W`))) ]
         
DISCUSSION: Standard ResNet’s classification head is not a spatial encoder: it explicitly collapses spatial layout right
before classification by collapsing (C, H, W) -> (C, 1, 1) via AdaptiveAvgPool2d. This channel-emphasis is the 
motivation for using channel (rather than spatial) `AdditiveAttention`. `AdditiveAttention` computes per-channel gates 
`alpha_t` and reweights `img_embedding` channel-wise; amplifying/suppressing channels before a spatial squeeze is 
applied. The uniform spatial average pooling per channel (the operation analogous to AdaptiveAvgPool2d) happens later in
`Jimmy`, when `self.img_avg_pool(img_attended_t)` collapses (BS, 2048, 49) -> (BS, 2048). Note that _avg-pool after 
gating_ is equivalent to _avg-pool then gate_ once `alpha_t` is known.
""".strip()


# L0CKED DR0P0UT
# ======================================================================================================================
class LockedDropout(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout_P = args.dropout_P
        self.mask      = None # Mask tensor; created in `Jimmy.forward`

    def reset(self): # Called once per batch (per `Jimmy.forward`) to force a new mask for the next sequence/batch
        self.mask = None

    def forward(self, x):
        if (not self.training) or (self.dropout_P == 0.0): # Dropout disbaled in eval model; dropout no-op if 
            return x                                       # .. `dropout_P == 0`: return `x` unchanged
        elif self.dropout_P == 1.0: # Drops all activations; avoid division by zero in `/ keep_P` below
            return torch.zeros_like(x)

        if self.mask is None: # Create mask
            keep_P = 1.0 - self.dropout_P
            
            # `self.mask`: `rand_like` draws Uniform(0, 1) on the same device and shape (inc. final partial batch) as 
            # .. `x`; `< keep_P` returns a boolean keep/drop decision per element (BS * D independent draws); `type_as`
            # .. converts bool to float in `x`’s dtype
            self.mask = (torch.rand_like(x) < keep_P).type_as(x) / keep_P

        return x * self.mask # Apply mask elementwise (Hadamard); (BS, D) == (BS, 216)
    
    
LockedDropout.__doc__ = """
On each for-loop iteration `j`, `Jimmy.forward` passes `fusion_cat_t` (of shape (BS, D) == (BS, 216)) through 
`fusion_mlp`, which invokes dropout once. Denote `fusion_cat_t` entering dropout as 'x_j'.

Standard `nn.Dropout()` samples a mask tensor 'M_j' of shape 'x_j', containing (BS * D) independent keep/drop decisions.
The mask is applied elementwise: 'y_j = x_j * M_j' (Hadamard).

Thus, across the rollout, `nn.Dropout()` would sample a fresh mask on each iteration `j`, producing { M_0, ... , 
M_{`timestep_horizon` - 1} } ~distinct mask tensors and creating (BS * D * `timestep_horizon`) independent dropout
keep/drop decisions per batch rollout.

This per-step mask re-sampling stochastically censors sample features (activations) on every step `j`: the decoder then 
must ~model not only the sequence dynamics, but also a time-varying pattern of missing coordinates, i.e., noise.  By 
contrast, `LockedDropout` creates a single mask, 'M' at step `j = 0` and reuses it on each step (until 'M' is `reset()` 
in `Jimmy.forward` for the next batch), so the decoder sees a consistent _thinned_ feature subspace (of a sample) over 
time rather than a different subspace at each step. Note that keep/drop decisions are sampled independently for each 
feature of each sample.

Below is an exposition of standard dropout mechanics for a scalar sample activation 'x':

Let `dropout_P` = 'p' denote the probability of dropping (zeroing) 'x', and draw 'u' ~ Uniform(0, 1). Define `keep_P` =
'1 - p' as the probability of keeping 'x'. If 'u' < `keep_P`, we keep 'x'; otherwise (i.e., 'u' >= `keep_P`) we drop it.
For example, if 'p' = 0.2, then `keep_P` = 0.8: draws in [0.0, 0.8) keep 'x', while draws in [0.8, 1.0) drop it.

Dropout implements this keep/drop decision via a multiplicative mask 'm': 'y = x * m';
  - If 'u' >= `keep_P`, set 'm' = 0            i.e., 'y' = 0
  - If 'u' < `keep_P`,  set 'm' = 1 / `keep_P` i.e., 'y' = 'x' / `keep_P` e.g., if 'x' = 1, then 'y' = 1.25 (see below)
  
Dropout zeros activations (with probability 'p') during training, but is disabled at eval. For a bias-free linear 
neuron, applying dropout without scaling gives the _mean_ training prediction: `keep_P * w_1 * x_1 + ... + keep_P * 
w_n * x_n`. At evaluation, all inputs are retained, so the prediction is: `w_1 * x_1 + ... + w_n * x_n`. We therefore 
divide each activation `x` by `keep_P` during training. Its mean weighted contribution becomes `keep_P * w_i * x_i / 
keep_P = w_i * x_i`, so the mean training prediction matches the evaluation prediction.

In all, with locked dropout, masking is applied at the sample level. Each sample receives its own feature mask, reused
throughout its rollout and resampled when the sample is processed again. The intent is to discourage reliance on 
particular learned features while keeping their availability consistent throughout the sample’s rollout.
""".strip()


# M0DEL
# ======================================================================================================================
class Jimmy(pl.LightningModule):
    def __init__(self, 
                 args,
                 ):
        super().__init__() # Must run parent `init` to setup internal registries
        
        self.args = args # `args` is used in `Jimmy.forward`; to not override the default `return forward(batch)` 
                         # .. behaviour of `predict_step` (part of PyTorch training logic), `args` is passed to 
                         # .. `Jimmy.forward` via `self`
                         
        # Saves `ArgConfig` args into .ckpt to retain args context for model weights
        self.save_hyperparameters({"args": vars(self.args)}) # Extracts vars in `ArgConfig.__init__` as dict
    
    
    # `Jimmy` is passed to `Trainer` as `model` (see "train.py"). `Trainer` expects the `setup` method to setup
    # .. model at the _correct_ point in its lifecycle
    def setup(self, stage = None):
        # `Trainer` exposes `fit`, `validate`, `test`, and `predict` methods; each calls `setup(stage = "<method>")`; 
        # .. we call `trainer.fit()` only (see "train.py"), which contains both train and val loops, both belonging to 
        # .. `stage = "fit"`. We retain `stage` arg to uphold `Trainer` contract but do not branch on `stage` since 
        # .. we've got only one train/val model build; `stage = None` applies only when `setup()` is called manually.
        
        datamodule = self.trainer.datamodule # Retrieve `DataModule` from `Trainer`
        
        # Saved in .ckpt s.t. at "predict" stage we could (i) standardize numerical inputs to train-set statistics, (ii) 
        # .. map categorical values unseen during training to UNK = 0, and (iii) invert standardized predictions back to
        # .. the original scale
        self.hparams.update({"train_categ_attr_spec":  datamodule.train_categ_attr_spec,
                             "train_numer_data_stats": datamodule.train_numer_data_stats,
                             })
                
        # Construct `DataScaler` object to be called in `on_validation_epoch_end()`; see "utils.py"
        self.sales_scaler = u.DataScaler(datamodule.train_numer_data_stats["sales"]["mean"],    
                                         datamodule.train_numer_data_stats["sales"]["std"],
                                         )
        
        
        # M0DEL DECLARATI0N
        # --------------------------------------------------------------------------------------------------------------
        self.attr_encoder = AttributeEncoder(datamodule)
        attr_embed_dim    = sum(self.attr_encoder.attr_embedding_dims.values()) # 81; `attr_img_desc_dim` below is 40
        attr_img_desc_dim = sum(self.attr_encoder.attr_embedding_dims[val] for val in ("category", "color", "fabric"))
        
        # `img_avg_pool` mirrors ResNet's `AdaptiveAvgPool2d` + `Flatten` (i.e., (N, 2048, 7, 7) -> (N, 2048)) 
        # `img_fc_reduction` mirrors ResNet's 1000-class FC head
        self.img_encoder                   = ImageEncoder(datamodule)
        img_channel_dims, img_spatial_dims = self.img_encoder.img_embedding_dims # 2048, 49
        self.img_attn                      = AdditiveAttention(self.args, img_spatial_dims, attr_img_desc_dim)
        self.img_avg_pool                  = lambda x: x.mean(dim = 2, keepdim = False) # See above
        self.img_fc_reduction              = nn.Linear(img_channel_dims, self.args.fusion_img_dim, 
                                                       bias = False,
                                                       ) # See above
    
                
        # `LayerNorm`: `fusion_cat_t` cats modalities with different native scales, creating within-sample scale 
        # .. heterogeneity. LN:
        # .. .. a) Couples `fusion_cat_t` elements using per-sample statistics (mean and std); 
        # .. .. b) Before the γ scale and β shift and assuming ε = 0, centers `fusion_cat_t` element distribution at zero;
        # .. .. c) Sets the magnitude of the `fusion_cat_t` vector to { sqrt(`fusion_mlp_dim`) }; 
        # .. This creates preactivation stability at `fusion_mlp` entry. Flexible scaling is restored with learned γ and
        # .. β and is applied to `fusion_cat_t` samples across the epoch and for-loop iterations (i.e., sample-invariant). 
        # .. NOTE: # LN also sets the per-sample RMS of `fusion_cat_t` (i.e., the average amplitude across features) to 1,
        # .. .. thereby equalizing per-sample energy; γ and β do not reintroduce sample-specific amplitude.
        # On dropout: 
        # .. 1. Do not place dropout right after `LayerNorm`: T-S signals are singleton elements in `ts_cat_t` -- 
        # .. .. applying dropout right after LN can zero these elements, causing hard signal loss. Dropout is less 
        # .. .. likely to erase T-S signal after a mixing layer in `fusion_mlp`, as the signal is then distributed 
        # .. .. across multiple activations;
        # .. 2. Dropout removes features and damages existing representations; a following nonlinear layer can recombine
        # .. .. surviving features into alternative representations. A linear layer only affinely reprojects the damaged
        # .. .. representation. To realize dropout's intent --encouraging representational robustness-— place dropout 
        # .. .. before a nonlinear layer and not immediately before the final linear layer, `fusion_fc_reduction`;
        # .. 3. Do not place dropout after `fusion_mlp`: the output of a transformation block constitutes a finalized 
        # .. .. embedding and must be dropout-free to avoid injecting fresh synthetic mask noise (stochastic jitter) 
        # .. .. into the ensuing block (i.e., the GRU gates);
        # .. 4. Lock dropout if applied to a middle layer of a transformation block, i.e., sample one mask per sequence
        # .. .. (sales curve) and reuse it across all time-steps -- this acts as feature thinning (model averaging) and
        # .. .. avoids _temporal flicker_; see `LockedDropout.__doc__`.
        fusion_mlp_dim           = attr_embed_dim + self.args.fusion_img_dim + TS_DIM # 81 + 128 + 7 = 216
        self.fusion_mlp          = nn.Sequential(nn.LayerNorm(fusion_mlp_dim), # See above
                                                 self.fc_template(fusion_mlp_dim, fusion_mlp_dim),
                                                 LockedDropout(self.args),
                                                 self.fc_template(fusion_mlp_dim, fusion_mlp_dim),
                                                 ) # See `Jimmy._on_fusion_out_reduction_` for architecture logic
        self.fusion_fc_reduction = nn.Linear(fusion_mlp_dim, self.args.h_t_dim)

        # `nn.GRU` Config:
        # .. 1. `num_layers = 1` (unstacked);
        # .. 2. `sequence_length = 1` because, with teacher forcing, we iterate GRU timesteps manually via a for-loop;
        # .. 3. `batch_first = True` to match the canonical shape order of (BS, :, :) for inputs only (see below).
        # .. Thus, `nn.GRU` in/out (`encoder_out_t`/decoder_out) have shape (BS, sequence_length, input_size/hidden_size)
        # .. .. == (BS, 1, h_t_dim); `h_t` is invariant to `batch_first = True`: in/out shape
        # .. .. (num_layers, BS, hidden_size) == (1, BS, h_t_dim). While decoder_out and `h_t` shapes differ, both carry
        # .. .. the same GRU hidden state tensor; thus, we discard the GRU output decoder_out (see `Jimmy.forward`).
        # NOTE: One and the same `nn.GRU` instance is trained despite manually invoking the GRU on each time-step.
        self.decoder_rnn = nn.GRU(input_size  = self.args.h_t_dim,
                                  hidden_size = self.args.h_t_dim,
                                  batch_first = True,
                                  ) # See above
 
        # `readout_mlp`: without non-linear `readout_mlp`, `h_t` must capture temporal structure and support direct 
        # .. (affine) readout; `readout_mlp` separates concerns. NOTE: `h_t` is in ~(-1, 1) by GRU construction (Tanh);
        # .. outgoing `nn.Linear` maps to ~[-2.6, 31.7] (standardized) `y_t` range; 
        self.readout_mlp = nn.Sequential(self.fc_template(self.args.h_t_dim, self.args.h_t_dim),
                                         nn.Linear(self.args.h_t_dim, 1),
                                         ) # See above
        
        
        self.to(self.device) # Ensure all model parameters created in setup move to device as can vary by hardware.
        
        # L0SS & 0PTIMIZATI0N
        # --------------------------------------------------------------------------------------------------------------
        self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.nn.MSELoss() # `MSELoss.forward(input: model’s predictions, target: ground truth)`
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.args.lr)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr = self.args.lr)


    # HELPER METHODS
    # ------------------------------------------------------------------------------------------------------------------
    def fc_template(self, 
                    in_features, 
                    out_features, 
                    activation = nn.ReLU, 
                    bias       = True, 
                    dropout_P  = 0.0, # i.e., standard dropout off
                    ):
        
        layers = [nn.Linear(in_features, out_features,
                            bias = bias,
                            ), 
                  activation(), # Instantiate class (fresh module per call)
                  ]
        
        if dropout_P: layers.append(nn.Dropout(p = dropout_P)) 
        
        return nn.Sequential(*layers) # Unpack layers
        
            
    # M0DEL F0RWARD 
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, batch):
        (img, # (BS, 3, 224, 224)
         ts_y, y_star_zero, ts_y_star_mask, # (BS, 12), (BS, 1), (BS, 12)
         ts_in_market_week, ts_calendar_week, ts_holiday, ts_discount, # (BS, 12), (BS, 12, 2), (BS, 12), (BS, 12)
         attr_price, # (BS)
         attr_retail, attr_category, attr_color, attr_fabric, # (BS), (BS), (BS), (BS)
         ) = batch
        
        # Force a new mask for the next sequence/batch on _every_ `LockedDropout` instance
        for module in self.modules(): getattr(module, "reset", lambda: None)()
                
        # Attribute Block: `attr_output` is 5-tuple of tensor encodings; `attr_EMBEDDING` is passed to `fusion_mlp` 
        # .. for cross-modal feature mixing; `attr_IMG_DESCRIPTOR` comprises category/fabric/color and (alongside the  
        # .. temporal context `h_t_minus_1`) is passed to `AdditiveAttention` to condition `img_embedding` on a SKU's 
        # .. visual semantics -- price/retail are omitted as these encode merchandising/economics and don't inform  
        # .. visual salience. As a result, category, color, and fabric feed both attention and fusion; their embeddings'
        # .. autograd paths are bifurcated and gradient contributions are summed <?>.
        attr_output                 = self.attr_encoder(attr_price, attr_retail, attr_category, attr_color, attr_fabric)
        _, _, *attr_img_descriptors = attr_output # OUT: _, _, [embedding_category, embedding_color, embedding_fabric]
        
        attr_EMBEDDING              = torch.cat(attr_output, dim = 1) # (BS, 81) == (BS, `attr_embed_dim`); concat all
        attr_IMG_DESCRIPTOR         = torch.cat(attr_img_descriptors, dim = 1) # (BS, 40)
        
        img_embedding = self.img_encoder(img) # (BS, 2048, 49)

        # `h_0`: cold-start `h_t` w/zeroes; `h_0` explicitly declared because custom `AdditiveAttention`, unlike
        # .. `nn.GRU`, does not default its initial hidden state to zeros (when arg is omitted). `h_0` tensor must 
        # .. manually be moved `.to(self.device)`. 
        batch_size    = ts_y.size(0) # Final batch can be smaller, hence, `args.batch_size` can lie
        h_0           = torch.zeros(1, batch_size, self.args.h_t_dim).to(self.device) # (1, BS, 128); see above
        y_hat_rollout = [] # y_hat aggregator; appended tensors remain on device and carry gradient graphs (see 
                           # .. `Jimmy._on_detach_`)
                                                                                
        pass_y = True if self.args.forecast_mode == "next-week" else None # This is a "TEACHER F0RCING" excerpt
        
        for j in range(self.args.timestep_horizon): # For for-loop logic see `Jimmy._on_loop_mechanics_` 
            
            if j == 0:
                h_t_minus_1 = h_0
                y_star_t    = y_star_zero 
            else:
                h_t_minus_1 = h_t # `h_t` declared below; j >= 1: uses `h_t` from previous step
            
            
                # TEACHER F0RCING
                # ------------------------------------------------------------------------------------------------------
                if self.args.forecast_mode == "horizon" and self.training: # `True` in `training_step`
                    tf_decision = (bool(self.args.teacher_forcing_P) and # Short-circuit ON/OFF check; `None` -> `False`
                                   (random.random() <= self.args.teacher_forcing_P) # Uniform [0, 1) draw
                                   )
                    pass_y      = tf_decision

                if pass_y:
                    y_t      = ts_y[:, j - 1].unsqueeze(1) # (BS) -> (BS, 1)
                    y_star_t = y_t
                else: # Pass y_hat (TF OFF)
                    y_hat_t  = y_hat_rollout[j - 1].detach() # (BS, 1); see `Jimmy._on_detach_`!
                    y_star_t = y_hat_t
            
            
            # TIME-SERIES
            # ----------------------------------------------------------------------------------------------------------
            y_star_mask_t           = ts_y_star_mask[:, j].unsqueeze(1) # (BS, 12) -> (BS) -> (BS, 1)
            in_market_week_t_plus_1 = ts_in_market_week[:, j].unsqueeze(1) # (BS, 12) -> (BS) -> (BS, 1)
            calendar_week_t_plus_1  = ts_calendar_week[:, j, :] # (BS, 12, 2) -> (BS, 2)
            holiday_t_plus_1        = ts_holiday[:, j].unsqueeze(1) # (BS, 12) -> (BS) -> (BS, 1)
            discount_t_plus_1       = ts_discount[:, j].unsqueeze(1) # (BS, 12) -> (BS) -> (BS, 1)
            
            ts_cat_t = torch.cat([y_star_t, # (BS, 1)
                                  y_star_mask_t, # (BS, 1)
                                  in_market_week_t_plus_1, # (BS, 1)
                                  calendar_week_t_plus_1, # (BS, 2)
                                  holiday_t_plus_1, # (BS, 1)
                                  discount_t_plus_1, # (BS, 1)
                                  ], dim = 1) # (BS, 7)
                                 
                 
            # IMAGE
            # ----------------------------------------------------------------------------------------------------------
            img_attended_t    = self.img_attn(img_embedding, # (BS, channel_dims, spatial_dims) == (128, 2048, 49)
                                              attr_IMG_DESCRIPTOR, # (BS, attr_img_desc_dim) == (128, 40)
                                              h_t_minus_1, # (num_layers, BS, hidden_size) == (1, 128, 128)
                                              ) # (BS, 2048, 49); reweight channels
            img_pooled_t      = self.img_avg_pool(img_attended_t) # (BS, 2048, 49) -> (BS, 2048); spatial pooling
            img_reduced_dim_t = self.img_fc_reduction(img_pooled_t) # (BS, 2048) -> (BS, 128) == (BS, fusion_img_dim)


            # FUSI0N
            # ----------------------------------------------------------------------------------------------------------
            fusion_cat_t  = torch.cat((attr_EMBEDDING, 
                                       img_reduced_dim_t,
                                       ts_cat_t,
                                       ), dim = 1) # (BS, 81) + (BS, 128) + (BS, 7) = (BS, 216)
            
            fusion_out_t  = self.fusion_mlp(fusion_cat_t) # (BS, 216) == (BS, fusion_mlp_dim)
            encoder_out_t = self.fusion_fc_reduction(fusion_out_t) # (BS, 128) == (BS, h_t_dim)


            # DEC0DER AND READ0UT
            # ----------------------------------------------------------------------------------------------------------
            _, h_t         = self.decoder_rnn(encoder_out_t.unsqueeze(1), # (BS, 128) == (BS, h_t_dim) -> (BS, 1, 128)
                                              h_t_minus_1, # (1, BS, 128) == (1, BS, h_t_dim)
                                              ) # (BS, 1, h_t_dim), (1, BS, h_t_dim) == (1, BS, 128)

            y_hat_t_plus_1 = self.readout_mlp(h_t.squeeze(0)) # { (1, BS, 128) -> (BS, 128) } -> (BS, 1)


            # ----------------------------------------------------------------------------------------------------------
            y_hat_rollout.append(y_hat_t_plus_1) # [ (BS, 1), ... , (BS, 1) ]
                
        y_hat_rollout = torch.cat(y_hat_rollout, dim = 1) # (BS, timestep_horizon) tensor: [ (BS, 1), ... , (BS, 1) ] ->
                                                          # .. [ (BS), ... , (BS) ]
        return y_hat_rollout, ts_y, batch_size # (BS, timestep_horizon) {OR (1, timestep_horizon] in single sample 
                                               # .. `model_ckpt.eval()` mode}, (BS, timestep_horizon), int
                                               # Returning `ts_y` guarantees y_hat-aligned, device safe ground truth for
                                               # .. loss calculation as DataLoader shuffles data in training
    

    # TRAINING L0GIC 
    # ------------------------------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        return self.optimizer # See `setup()`
    
    
    def training_step(self, train_batch, batch_idx):        
        y_hat_rollout, ts_y, batch_size = self.forward(train_batch) # (BS, 12), (BS, 12); == (BS, timestep_horizon), int
        train_batch_loss                = self.criterion(y_hat_rollout, ts_y) # Per-batch loss i.e., on step 

        # Log to `callback_metrics` (and configured logger, see "train.py") for monitoring
        self.log(name       = "train_loss", 
                 value      = train_batch_loss,
                 on_step    = True, # Logs `train_batch_loss` on hand in `training_step()`: "train_loss_step"
                 on_epoch   = True,       # To accumulate batch losses across steps, then compute/log their epoch-level
                 batch_size = batch_size, # .. average; epoch loss is the weighted (by sample num per batch) mean of 
                 )                        # .. per-batch losses (final batch is smaller); logged as "train_loss_epoch"
        
        return train_batch_loss


    def on_validation_epoch_start(self):        
        # Init val `y_hat_rollout` and `ts_y` tensor aggregators for custom epoch metrics in `validation_step`;  
        # .. aggregated (appended) tensors remain on device
        self.val_Y_hat_epoch = []
        self.val_ts_Y_epoch  = []
        

    def validation_step(self, val_batch, batch_idx):
        y_hat_rollout, ts_y, _ = self.forward(val_batch) # (BS, 12), (BS, 12); == (BS, timestep_horizon), int
        
        self.val_Y_hat_epoch.append(y_hat_rollout.detach()) # Aggregate tensors for custom epoch metrics in 
        self.val_ts_Y_epoch.append(ts_y.detach())           # .. `on_validation_epoch_end`; gradient graphs superfluous
        

    def on_validation_epoch_end(self):
        val_Y_hat_epoch = torch.vstack(self.val_Y_hat_epoch) # [ (BS, 12), ... , (BS, 12) ] -> (N_batches * BS, 12); at 
        val_ts_Y_epoch  = torch.vstack(self.val_ts_Y_epoch)  # .. 10% val: (84 * BS, 12) == (10,685 {not 10,752 as final
                                                             # ..  batch is smaller}, 12)                               
        val_Y_hat_epoch_inv = self.sales_scaler.inv(val_Y_hat_epoch) # Destandardize 
        val_ts_Y_epoch_inv  = self.sales_scaler.inv(val_ts_Y_epoch)

        val_loss_epoch = self.criterion(val_Y_hat_epoch, val_ts_Y_epoch) # Compute metrics
        val_wape_epoch = u.WAPE(val_Y_hat_epoch_inv, val_ts_Y_epoch_inv)
        val_mae_epoch  = u.MAE(val_Y_hat_epoch_inv,  val_ts_Y_epoch_inv)
        
        self.log_dict({"val_loss_epoch": val_loss_epoch, 
                       "val_WAPE_epoch": val_wape_epoch, 
                       "val_MAE_epoch":  val_mae_epoch,
                       }) # Log to `callback_metrics` (and configured logger, see "train.py") for monitoring


    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics # Retrieve metrics storage
        lr      = self.optimizers().param_groups[0]["lr"] # Pull learning rate
        
        self.print(f'Epoch {self.current_epoch}  '
                   '|   Train:   '
                   f'LR {lr:.6f}   '
                   f'Loss {float(metrics["train_loss_epoch"]):.6f}   '
                   '|   Validation:   '
                   f'Loss {float(metrics["val_loss_epoch"]):.6f}   '
                   f'WAPE {float(metrics["val_WAPE_epoch"]):.6f}   '
                   f'MAE {float(metrics["val_MAE_epoch"]):.6f}   '
                   )
        
        
Jimmy.__doc__ = """
Navigate to sections below:

- `print(Jimmy._on_loop_mechanics_)`
- `print(Jimmy._on_detach_)`
- `print(Jimmy._on_fusion_out_reduction_)`.
""".strip()


Jimmy._on_loop_mechanics_ = """
NOTATION: For clarity, symbol "t" describes the prediction-step semantically (i.e., weeks); symbol "i" describes how 
.. time-steps "t" are laid out across the time-series tensors; symbol "j" is a for-loop counter that drives computation. 
.. Indices "t", "i", "j" are distinct by mechanism design; equality must not be assumed. 

This model constitutes a multimodal nonlinear autoregressive forecaster w/eXogenous covariates, implemented w/GRU:
.. y_hat_{t+1} = f(y_star_{t}, MASK_{t}, X_{t+1}, IMG_{t}, ATTR) + error_{t+1}; 
.. .. 1. y_star_{t}: One-lag autoregressive input where, in:
.. .. .. `forecast_mode = "horizon"`: y_star_{t} = y_{t} if Teacher Forcing is ON AND if `self.training` else y_hat_{t};
.. .. .. `forecast_mode = "next-week"`: y_star_{t} = y_{t};
.. .. 2. MASK_{t}: Binary delimiter `y_star_mask_t`; MASK_{i = 0} = 1, MASK_{i > 0} = 0 (see below);
.. .. 3. X_{t+1}: Known future covariates for the predicted week (t + 1): discount_{t+1}, holiday_{t+1}, 
.. .. .. calendar_week_{t+1}, in_market_week_{t+1};
.. .. 4. IMG_{t}: Channel-attended SKU image embedding conditioned on GRU hidden state h_{t-1} and static attributes
.. .. .. subset (category, color, fabric);
.. .. 5. ATTR: Static attributes (retail, category, color, fabric) and price scalar.
Each sales curve comprises 12 weeks. The first week of the sales curve is denoted t = 1. Time-series tensors (`ts_y`, 
.. `ts_in_market_week`, `ts_calendar_week`, `ts_holiday`, `ts_discount`, and the y_hat_{t+1} accumulator 
.. `y_hat_rollout`) are aligned as (NOTE: "i" below denotes tensor index):
.. .. ( i_{0} : week_{t = 1}, i_{1} : week_{t = 2}, ..., i_{10} : week_{t = 11}, i_{11} : week_{t = 12} )
In the context of time-series inputs, we predict y_hat_{t+1} from y_star_{t}, X_{t+1}, (and MASK_{t}; see below). We 
.. iterate the for-loop counter j: 0, 1, ..., 11 (t = 12-week sales curve): 
.. .. j = 0: predict y_hat_{t = 1} from X_{t = 1} at (i = 0); for y_star_{t = 0}, whereby neither y_hat_{t = 0} nor 
.. .. .. y_{t = 0} exist, we pass a zero vector `y_star_zero`; we disambiguate this synthetic cold-start input from
.. .. .. true zero sales using MASK_{t = 0} at (i = 0). NOTE: We tie the time-step "t" of MASK_{t} to that of 
.. .. .. y_star_{t}.
.. .. j = 1: predict y_hat_{t = 2} from X_{t = 2} at (i = 1); y_star_{t = 1} at (i = 0), whereby both 
.. .. .. y_hat_{t = 1} and y_{t = 1} exist; MASK_{t = 1} at (i = 1).
.. .. [ ... ]
.. .. j = 11: predict y_hat_{t = 12} from X_{t = 12} at (i = 11); y_star_{t = 11} at (i = 10) -- y_hat_{t = 11} or
.. .. .. y_{t = 11}; MASK_{t = 11} at (i = 11). NOTE: The final week, y_{t = 12} at (i = 11), is never passed as input
.. .. .. because y_hat_{t = 13} does not exist; we retain y_{t = 12} in the `ts_y` tensor for loss calculation.   
""".strip()


Jimmy._on_detach_ = """
`.detach()` severs the unintended gradient paths created by the model auto-regressively consuming its own predictions. 
Notation: 
.. 'l_t': Per-step (for-loop iteration) loss term; total loss: 'L = Σ_t l_t'
.. 'F': Notional transformation block; a map from time-series input data -> `h_t` (implemented as `fusion_mlp`, 
.. .. `decoder_rnn`, etc.).
In multi-step horizon prediction, (barring TF is ON) the model consumes its outputs (i.e., `y_star_t` = `y_hat_t`). The 
.. gradient chain for the first two for-loop iterations is:
.. .. `y_star_zero` -> 'F' -> 'h_1' -> `readout_mlp` -> 'y_hat_1' -> 'l_1'
.. ..                         "   "  +                  "       " -> 'F' -> 'h_2' -> `readout_mlp` -> 'y_hat_2' -> 'l_2'
Without `.detach()`, an extra gradient path appears from 'l_2' to 'h_1' via 'y_hat_1', where 'y_hat_1' is both an 
.. outgoing edge (i.e., loss site 'l_1', where error is computed against 'y_1') and an internal node of the 
.. gradient-graph. Importantly, with probabilistic teacher forcing, the presence of this extra path varies from 
.. (decoder) step to step, yielding an irregular BPTT graph. With `.detach()`, this extra gradient path is severed.
.. Note that observed ground-truth values (i.e., `y_star_t` = `y_t`) carry no preceding gradient into 'F': 
.. .. `y_star_zero`      -> 'F' -> 'h_1' -> `readout_mlp` -> 'y_hat_1' -> 'l_1'
.. .. 'y_hat_1' OR 'y_1'  +        "   " -> 'F' -> 'h_2' -> `readout_mlp` -> 'y_hat_2' -> 'l_2'
In all, probabilistic teacher forcing varies gradient-graph connectivity (not inherently better or worse for learning); 
.. `.detach()` removes this variation, keeping connectivity consistent.
""".strip()


Jimmy._on_fusion_out_reduction_ = """
Two ways to reduce `fusion_out_t` from size {216 -> 128} ( == `fusion_mlp_dim` -> `h_t_dim`); pass `fusion_out_t` to:
.. Case A. The three distinct GRU affines (update gate, reset gate, candidate h_t), each mapping {216 -> 128};
.. Case B. An affine `fusion_fc_reduction` mapping {216 -> 128}, then each GRU affine maps {128 -> 128}.
.. Linear algebraic view:
.. .. Case B. Let x_v be a non-zero vector in R^3 (analogy for `fusion_out_t` of size 216). Let A_M be a {2 x 3} linear 
.. .. .. R^3 -> R^2 map (analogy for `fusion_fc_reduction` {216 -> 128} map). For simplicity, assume A_M is full rank 
.. .. .. i.e., 2. Let P_M and Q_M be {2 x 2} linear maps on R^2 (analogues for gate-specific {128 -> 128} affines; 
.. .. .. same logic applies to candidate h_t). Assume P_M and Q_M are full rank i.e., 2 (for simplicity; see "*" below)
.. .. .. and { P_M != Q_M } (as equivalence upon training is improbable). This setup yields two two-stage transforms:
.. .. .. .. (1) P_M A_M x_v = p_v
.. .. .. .. (2) Q_M A_M x_v = q_v
.. .. Case A. Let B_M and C_M be {2 x 3} linear R^3 -> R^2 maps (analogues for gate-specific {216 -> 128} affines in 
.. .. .. Case A; again, same logic applies to candidate h_t) w/the same properties as A_M, and w/{ A_M != B_M != C_M }
.. .. .. (again, as equivalence upon training is improbable). This setup yields two single-stage transforms:
.. .. .. .. (3) B_M x_v = b_v
.. .. .. .. (4) C_M x_v = c_v
.. .. Since A_M: R^3 to R^2 has rank 2, Null(A_M) is a 1-D subspace (line) in R^3. This forces a many-to-one behaviour: 
.. .. .. .. (5) Let A_M x_v = y_v ; A_M (x_v + k n_v) => (A_M x_v) = y_v : k ∈ R, n_v ∈ Null(A) excluding 0 } 
.. .. .. That is, through every x_v in R^3, the line that is parallel to Null(A_M) collapses to a single point in R^2, 
.. .. .. the product { (A_M x_v) = y_v }. This implies that A_M is blind to the null direction, which determines the 
.. .. .. information lost from R^3 to R^2 and, in turn, defines the semantics of R^2.
.. .. Cogent GRU gating is contingent on a shared notion of “what the features mean” across gates. In Case B, P_M and  
.. .. .. Q_M merely re-encode y_v coordinates in R^2 i.e., the image of A_M (see Eq. (1), (2)), so distinct GRU affines 
.. .. .. operate on the same A_M-induced semantics. In Case A, exact collinearity between B_M and C_M is improbable,  
.. .. .. implying Null(B_M) != Null(C_M); this induces different information loss in each gate's reduction. We prefer 
.. .. .. Case B: A_M determines what information survives compression, then P_M and Q_M use that shared information
.. .. .. differently.
.. .. .. * -- Semantic drift induced by rank-deficient P_M, Q_M w/non-trivial null spaces is constrained by the shared
.. .. .. .. Im(A), whereas distinct and direct per-gate B_M, C_M reductions risk greater drift.
""".strip()
