# -*- coding: utf-8 -*-
############################################################
# GCM_APPLY.py -- Diffusion Edition, with 3x sampling
#
#   - Reads normalization stats from `normalisation_stats_ALL_AUS.json`
#   - Loads historical/future GCM data
#   - Coarsens to 1.5, normalizes using precomputed stats
#   - Applies your EDM diffusion model day-by-day (3 times!)
#   - Denormalizes & coarsens final outputs to 0.5
#   - Saves each run's NetCDF & sample plots with _1, _2, _3 suffixes
#
############################################################

import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import json

# ============================
# 1) Load your EDM model code�
# ============================
from src.Network import EDMPrecond

###################################
# 2) Configuration / Paths
###################################

# Paths/config
DATA_PATH            = '/g/data/al33/replicas/CMIP5/combined'
save_dir             = "/g/data/hn98/Declan/Downscale-SR/Checkpoints/Diffusion/" #"/g/data/hn98/Declan/Downscale-SR/Checkpoints/Earth-ViT/0.5deg_0.25deg/"
images_save_dir      = "/g/data/hn98/Declan/Downscale-SR/Output_Files/0.5deg_0.25deg/Visuals/"
normal_stats_json    = "normalisation_stats_ALL_AUS_30yr_sv2.json" #"normalisation_stats_ALL_AUS_30yr.json" #"normalisation_stats_ALL_AUS.json"  #NEED TO CHANGE THE NORMALISATION FILE AS WELL HERE FOR 20YR# precomputed stats
#SAVE THE NORMALISATION FILE MORE CAREFULLY WHEN RUNNING AGAIN ^^^
diffusion_ckpt       = "Equal_Aus_FPN416-30yrs.pt" #"Equal_Aus_FPN361-30yrs.pt" #"Equal_Aus_FPN351-30yrs.pt" #"Equal_Aus_FPN140-10yr.pt" #Switching to the FPN models #"Equal_Aus_MASS360-30yr.pt" #Equal_Aus_MASS119-240-30yr.pt #"Equal_Aus_MASS360-30yr.pt" #Equal_Aus_MASS199-20yr.pt" #"Equal_Aus_150-10yr-no-mass-loss.pt" #"Equal_Aus_30.pt"  #Australia over 30 epochs, 2 years data?      # your trained EDM checkpoint


#Trying again here with new
#diffusion_ckpt = "Equal_Aus_FPN655-60yrs.pt"
#normal_stats_json = "normalisation_stats_ALL_AUS_60yr_sv.json"

# GCM specifics
'''
GCM_PATH_MIDDLE      = 'CNRM-CERFACS/CNRM-CM5'
HISTORICAL_VERSION   = 'v20120530'
FUTURE_VERSION       = 'v20130101'
'''

#


GCM_PATH_MIDDLE = 'MIROC/MIROC5' # 'CNRM-CERFACS/CNRM-CM5'  # Replace with desired GCM
HISTORICAL_VERSION = 'v20120710'  # Version subfolder for historical data
FUTURE_VERSION = 'v20120710'     # Version subfolder for future data


YEARS_TO_APPLY       = [1976,2005] #[1976, 2005]   # Historical
FUTURE_YEARS_TO_APPLY= [2070,2070] #[2070, 2099]   # Future

# Spatial shapes
Model_shape          = [144, 272]  # 0.25 deg shape your model uses
model_coarse_shape   = [24, 46]    # ~1.5 deg shape the model was trained for

# Geographic region
lat_min, lat_max     = -43.5, -7.75
lon_min, lon_max     = 109.0, 176.75

# For final "coarsen to 0.5 deg"
final_coarse_shape   = (72, 136)

# Mapping from ERA5-like variable names to GCM variable names
VARIABLE_MAPPING = {
    'surface': {
        '10u': 'uas',
        '10v': 'vas',
        '2t': 'tas',
        'tp': 'pr',
    },
    'pressure': {
        't': 'ta',
        #'u': 'ua',
        #'v': 'va',
        #'q': 'hus',
        #'z': 'zg',
    }
}
PRESSURE_LEVELS = ['50'] #, '100', '250']  # e.g.






#############################################################
#ADD PORTION TO SAMPLE THIS IN A HIERARCHICAL MANNER FROM COARSE TO FINE (FPN)

################################################################################
# HIERARCHICAL SAMPLER UTILITIES
################################################################################
import torch
import torch.nn.functional as F
import numpy as np

class HierarchicalScheduler:
    """
    A simplified scheduler that linearly interpolates from (1,1) up to full_size
    across total_steps. For example:
      - total_steps=2 => [(1,1), (144,272)]
      - total_steps=3 => [(1,1), (72,136), (144,272)]
      - total_steps=10 => 10 equally spaced shapes from (1,1) to (144,272)
    """
    def __init__(self, full_size=(144, 272), total_steps=10, device='cuda'):
        """
        :param full_size: The final shape, e.g. (144, 272)
        :param total_steps: Number of steps in the schedule
        :param device: Torch device
        """
        self.full_size = full_size
        self.total_steps = total_steps
        self.device = device
        # Build shape_list from (1,1) up to full_size
        self.shape_list = self._compute_shape_schedule()

    def _compute_shape_schedule(self):
        """
        Build a list of length `total_steps`. Each entry is (h,w).
        shape_list[0]   = (1,1)
        shape_list[-1]  = self.full_size
        in between is linearly spaced in each dimension, then rounded.
        """
        shape_list = []
        (final_h, final_w) = self.full_size
        for step_idx in range(self.total_steps):
            if self.total_steps == 1:
                frac = 1.0
            else:
                frac = step_idx / (self.total_steps - 1)
            cur_h = 1 + frac * (final_h - 1)
            cur_w = 1 + frac * (final_w - 1)
            shape_list.append((int(round(cur_h)), int(round(cur_w))))
        return shape_list

    def upsample_to_full(self, x_small: torch.Tensor):
        """Upsample x_small from shape(...) to self.full_size using bilinear."""
        return F.interpolate(
            x_small,
            size=self.full_size,
            mode='bilinear',
            align_corners=False
        )



class OLDHierarchicalScheduler:
    def __init__(self,
                 full_size=(144, 272),
                 total_steps=500,
                 x_over_y_ratio=None,  # e.g. 144 / 272 or 272 / 144
                 noise_steps_per_split=1,
                 shape_splits_per_shrink=1,
                 use_exponential=False,
                 device='cuda'):
        """
        full_size: (H, W) = (144, 272)
        total_steps: e.g. 500
        x_over_y_ratio: e.g. 272/144 = 1.888... if you want to reduce W more often
        noise_steps_per_split: how many noisy steps to do before actually
                               decrementing shape by 1.
        use_exponential: if True, shape stays the same for e.g. i^2 steps, etc.
        """
        self.full_size = full_size
        self.total_steps = total_steps
        self.device = device

        if x_over_y_ratio is None:
            # ratio for how many times to reduce row vs col
            # default is W/H if you want col to reduce more often
            self.x_over_y_ratio = full_size[1] / full_size[0]
        else:
            self.x_over_y_ratio = x_over_y_ratio

        self.noise_steps_per_split = noise_steps_per_split
        self.shape_splits_per_shrink = shape_splits_per_shrink
        self.use_exponential = use_exponential

        # Precompute shape schedule for t = 0..T
        # shape_list[t] = (h_t, w_t)
        self.shape_list = self._compute_shape_schedule()

    def _compute_shape_schedule(self):
        """
        Return a list of length total_steps+1 of shapes from (H,W) down to (1,1).
        If shape cannot land exactly at (1,1) in exactly total_steps, we
        clamp at (1,1) once both dims have reached 1.
        """
        # Start at (144,272), end at (1,1)
        shape_list = []
        cur_h, cur_w = self.full_size
        # counters that say how many times we have shrunk each dimension
        done_h = (cur_h <= 1)
        done_w = (cur_w <= 1)

        t = 0
        shape_list.append((cur_h, cur_w))

        # We'll define when to reduce each dimension
        # so that ratio ~ self.x_over_y_ratio
        # and we also skip the dimension if its already at 1
        while t < self.total_steps:
            t += 1

            # Optionally do not shrink shape yet if we haven't
            # reached the 'noise_steps_per_split' or an exponential schedule, etc.
            shrink_now = False

            if self.use_exponential:
                # Example of an exponential schedule:
                # stay for (cur_h * cur_w) steps or cur_h^2 steps, etc.
                # Modify as desired.
                # We do something simpler here: shrink if t mod (cur_h+cur_w) == 0, e.g.
                shrink_now = (t % (cur_h + cur_w)) == 0
            else:
                # linear approach:
                # every 'noise_steps_per_split' steps, we shrink by 1 in whichever dimension
                # is bigger or we check the ratio
                shrink_now = ((t % self.noise_steps_per_split) == 0)
                
                

            
            #PART BELOW HERE, WORKS LIKE A CHARM - JUST ADDING FUNCTIONALITY TO DO MULTIPLE SHRINKS PER NOISE STEP
            

            if shrink_now and (not done_h or not done_w):
                # Decide which dimension to shrink
                # e.g. if w/h = 1.88, shrink w more often
                if not done_h and not done_w:
                    ratio_now = cur_w / float(cur_h)
                    if ratio_now > self.x_over_y_ratio:
                        # means w is relatively bigger
                        cur_w -= 1
                    else:
                        cur_h -= 1
                elif not done_h:
                    # w is done => must shrink h if possible
                    cur_h -= 1
                elif not done_w:
                    # h is done => must shrink w if possible
                    cur_w -= 1

            # clamp to minimum 1
            if cur_h < 1: cur_h = 1
            if cur_w < 1: cur_w = 1
            if (cur_h <= 1) and (cur_w <= 1):
                done_h, done_w = True, True

            shape_list.append((cur_h, cur_w))
            
            

        return shape_list

    def forward_noising_chain(self, x0, add_noise=True):
        """
        Build the forward chain x_0-> x_1-> ...-> x_T, storing them in a list.
        Each step:
          1) add random noise if desired
          2) downsample from shape[t] to shape[t+1]
          3) store the result
        Then upsample back to (144,272) so its always the same shape for the next step?
        Actually in forward pass we keep it native, but for the UNet wed upsample.
        But for this function we store the true smaller shape version as well.

        x0: the original image in R{B,C,H,W} = (144,272)
        """
        import torch.nn.functional as F

        chain = []
        current_x = x0

        for t in range(self.total_steps): #removed ####aDDED THE -1 HERE AS WELL
            # shape[t], shape[t+1]
            (h_t, w_t)   = self.shape_list[t]
            (h_next, w_next) = self.shape_list[t+1]

            # 1) Add random noise
            if add_noise:
                noise = torch.randn_like(current_x) * 0.01  # scale can vary
                current_x = current_x + noise

            # 2) Downsample from (h_t,w_t) to (h_next,w_next)
            if (h_next != h_t) or (w_next != w_t):
                # first reshape/upsample current_x to (h_t, w_t) if needed
                # in practice you might store or ensure it *already* is (h_t, w_t).
                # for demonstration:
                current_size = (current_x.shape[-2], current_x.shape[-1])
                if current_size != (h_t, w_t):
                    # upscale/downscale to (h_t, w_t) first
                    current_x = F.interpolate(
                        current_x, size=(h_t, w_t), mode='bilinear', align_corners=False
                    )
                # now downsample to (h_next, w_next)
                current_x = F.interpolate(
                    current_x, size=(h_next, w_next), mode='bilinear', align_corners=False
                )

            chain.append(current_x)

        return chain

    def upsample_to_full(self, x_small):
        """Upsample from x_small's shape to (144,272)."""
        import torch.nn.functional as F
        return F.interpolate(
            x_small, size=self.full_size, mode='bilinear', align_corners=False
        )
    


import imageio


@torch.no_grad()
def sample_model_hierarchical(
    model,
    images_input: torch.Tensor,
    coarse: torch.Tensor,
    scheduler: OLDHierarchicalScheduler, #HierarchicalScheduler
    device: str = 'cuda',
    num_inference_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 30.0, #80.0
    rho: float = 4, #7
    S_churn: float = 0, #20.0 
    S_min: float = 0,
    S_max: float = float('inf'),
    S_noise: float = 1.0,
    surface_mask: torch.tensor = None,
    dataset = None,
    make_gif: bool = False,
):
    """
    Multi-scale hierarchical diffusion sampling:
      - Start from noise in shape (1,1)
      - Repeatedly upsample partial latent to (144,272) => pass it through the model => 
        downsample to the next bigger shape => loop
      - Returns the final predicted (residual) which you can combine with 'coarse'
        if your training was "coarse + residual => fine".
    """
    B, C, H, W = images_input.shape  # e.g. (1,4,144,272)
    # 1) Reverse shape_list => small->large
    small_to_large = list(scheduler.shape_list)
    small_to_large.reverse()  # e.g. from (1,1)->...->(144,272)
    print(f"Hierarchical shapes: {small_to_large}")
    #small_to_large.reverse() #Reversing again to test inverse  #25/04/25 - DOUBLE REVERSE TO SEE IMPACT HERE ON ACTUAL RESULTS
    print(f"FINAL Hierarchical shapes: {small_to_large}")
    frames = [] #For the gif

    # 2) We'll create EDM time steps:
    step_indices = torch.arange(num_inference_steps, dtype=torch.float64, device=device)
    t_steps = (
        (sigma_max ** (1 / rho)
         + step_indices / (num_inference_steps - 1) * (sigma_min**(1 / rho) - sigma_max**(1 / rho))) ** rho
    )
    # Append final t=0
    t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # 3) Start from random noise at shape(1,1)
    (tiny_h, tiny_w) = small_to_large[0]  # should be (1,1)
    x_next = torch.randn(B, C, tiny_h, tiny_w, dtype=torch.float64, device=device) * t_steps[0]

    # 4) Loop over t-steps and shapes
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # (a) upscale latent to next shape in "small_to_large"
        
        curr_shape = small_to_large[i]
        x_cur_sm = F.interpolate(x_next, size=curr_shape, mode='bilinear', align_corners=False)

        # (b) apply S_churn if t_cur in [S_min,S_max]
        if S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_inference_steps, np.sqrt(2) - 1)
        else:
            gamma = 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        if gamma > 0:
            x_hat_sm = x_cur_sm + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur_sm)
        else:
            x_hat_sm = x_cur_sm

        # (c) Interpolate x_hat_sm to (H,W) because model expects (144,272)
        x_hat_up = F.interpolate(x_hat_sm, size=(H, W), mode='bilinear', align_corners=False)

        # (d) We combine with 'images_input' if your model uses "coarse + mask" as conditions. 
        print("size of surface mask is {surface_mask.shape}")

        mask_3ch     = surface_mask.unsqueeze(0).to(torch.float64)
        
        cond_img = torch.cat([images_input, mask_3ch], dim=1)
        #cond_img = images_input  # adapt if needed

        # (e) Model forward
        denoised_up = model(x_hat_up, t_hat, cond_img, class_labels=None).to(torch.float64)
        # Euler step
        d_cur = (x_hat_up - denoised_up) / t_hat
        x_euler_up = x_hat_up + (t_next - t_hat) * d_cur

        if i < num_inference_steps - 1:
            # 2nd order correction
            denoised_euler_up = model(x_euler_up, t_next, cond_img, class_labels=None).to(torch.float64)
            d_prime = (x_euler_up - denoised_euler_up) / t_next
            x_next_up = x_hat_up + (t_next - t_hat) * 0.5 * (d_cur + d_prime)
        else:
            x_next_up = x_euler_up
        
        
        #
        # (f) Now we downsample x_next_up back to the *next shape* for the next iteration
        if i < num_inference_steps - 1:
            next_shape = small_to_large[i + 1]
            x_next = F.interpolate(x_next_up, size=next_shape, mode='bilinear', align_corners=False)
        else:
            x_next = x_next_up
        

        #This was testing the other iteration you had to remove upsample - note that checkpoint model here was not trained w/ this in mind 
        '''
        #Added this in here after commenting out the above - final returned value below is x_next so...
        if i == num_inference_steps - 1:        # last iteration only
            x_next = x_next_up
        '''

        #####################
        #DEC - TRY ADDING VIDEO???
        create_video = make_gif
        video_path = "GCM_APPLY_HIERARCHICAL.gif"
        print(f"Coarse shape is {coarse.shape}") 
        print(f"Pred shape is {x_next_up.shape}")

        NUM_DAYS_GIF = 5 #Generate it for the first 5 days

        if create_video:
            fig, axs = plt.subplots(4, 3, figsize=(16, 16))
            fig.suptitle(f"Step {i+1} / {num_inference_steps}", fontsize=16)

            # For each of the 4 variables:
            for var_idx in range(1):
                # -- 1) Coarse --
                coarse_var = coarse[0, var_idx].detach().cpu().numpy()
                vmin_c, vmax_c = np.percentile(coarse_var, [1, 99])
                im0 = axs[var_idx, 0].imshow(
                    coarse_var, cmap="jet", vmin=vmin_c, vmax=vmax_c, origin="upper",
                )
                axs[var_idx, 0].set_title(f"Coarse var {var_idx+1}")
                axs[var_idx, 0].axis("off")
                fig.colorbar(im0, ax=axs[var_idx, 0], fraction=0.046, pad=0.04)

                '''
                # -- 2) Fine (GT) --
                fine_var = fine[0, var_idx].detach().cpu().numpy()
                vmin_f, vmax_f = np.percentile(fine_var, [1, 99])
                im1 = axs[var_idx, 1].imshow(
                    fine_var, cmap="jet", vmin=vmin_f, vmax=vmax_f, origin="upper",
                )
                axs[var_idx, 1].set_title(f"Fine (GT) var {var_idx+1}")
                axs[var_idx, 1].axis("off")
                fig.colorbar(im1, ax=axs[var_idx, 1], fraction=0.046, pad=0.04)
                '''

                # -- 3) Prediction (coarse + residual) --
                pred_var_unnorm = dataset.residual_to_fine_image(
                    x_next_up[0, var_idx].detach().cpu(), coarse.cpu()
                )
                pred_var = pred_var_unnorm[0, var_idx].numpy()
                vmin_p, vmax_p = np.percentile(pred_var, [1, 99])
                im2 = axs[var_idx, 1].imshow(
                    pred_var, cmap="jet", vmin=vmin_p, vmax=vmax_p, origin="upper",
                )
                axs[var_idx, 1].set_title(f"Prediction var {var_idx+1}")
                axs[var_idx, 1].axis("off")
                fig.colorbar(im2, ax=axs[var_idx, 1], fraction=0.046, pad=0.04)

                # -- 4) *Raw* Residual (directly from x_next_up) --
                residual_var = x_next_up[0, var_idx].detach().cpu().numpy()
                # For display only, we still do some percentile‐based color scaling:
                vmin_r, vmax_r = np.percentile(residual_var, [1, 99])
                im3 = axs[var_idx, 2].imshow(
                    residual_var, cmap="jet", vmin=vmin_r, vmax=vmax_r, origin="upper",
                )
                axs[var_idx, 2].set_title(f"Residual var {var_idx+1}")
                axs[var_idx, 2].axis("off")
                fig.colorbar(im3, ax=axs[var_idx, 2], fraction=0.046, pad=0.04)

            plt.tight_layout()

            # Convert the figure to a NumPy array and add to frames
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer._renderer)[..., :3]  # keep RGB
            frames.append(frame)
            plt.close(fig)


    if create_video and len(frames) > 1:
        # Convert frames to uint8 if needed
        frames_uint8 = [frame.astype(np.uint8) for frame in frames]
    
        # Save as GIF with 5 FPS
        imageio.mimsave(video_path, frames_uint8, fps=3, format='GIF')



    # x_next is final residual (H,W) if your shapes_list[-1] = (144,272)
    return x_next.float()







##############################################################

# ================
# 3) Utility Fns
# ================

def parse_date_range(filename):
    import re
    match = re.search(r'_(\d{8})-(\d{8})\.nc$', filename)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None

def load_gcm_variable(variable_folder, variable_name, years_to_apply):
    """Loads the GCM data for a single variable & year range."""
    if not os.path.exists(variable_folder):
        print(f"Variable folder {variable_folder} does not exist")
        return None

    data_list = []
    files = sorted([f for f in os.listdir(variable_folder) if f.endswith('.nc')])
    for file in files:
        start_end = parse_date_range(file)
        if not start_end:
            continue
        start_date, end_date = start_end
        start_year = int(start_date[:4])
        end_year   = int(end_date[:4])
        if end_year >= years_to_apply[0] and start_year <= years_to_apply[1]:
            ds = xr.open_dataset(os.path.join(variable_folder, file))

            if variable_name not in ds.variables:
                print(f"Variable {variable_name} not found in {file}")
                continue
            data_var = ds[variable_name].sel(
                time=slice(f'{years_to_apply[0]}-01-01', f'{years_to_apply[1]}-12-31')
            )
            data_list.append(data_var)
        else:
            print(f"File {file} does not cover the required years")

    if data_list:
        return xr.concat(data_list, dim='time')
    else:
        print(f"No data found for variable {variable_name} in {variable_folder}")
        return None

def load_gcm_data(variable_mappings, years_to_apply, gcm_path_middle,
                  scenario, version_folder):
    """Load surface & pressure data dicts for the GCM scenario & version."""
    surface_data  = {}
    pressure_data = {}

    # Surface variables
    for era_var, gcm_var in variable_mappings['surface'].items():
        var_dir = os.path.join(DATA_PATH, gcm_path_middle, scenario,
                               'day', 'atmos', 'day', 'r1i1p1',
                               version_folder, gcm_var)
        data = load_gcm_variable(var_dir, gcm_var, years_to_apply)
        if data is not None:
            surface_data[era_var] = data
        else:
            print(f"Data for surface var {gcm_var} not loaded.")

    # Pressure variables
    for era_var, gcm_var in variable_mappings['pressure'].items():
        var_dir = os.path.join(DATA_PATH, gcm_path_middle, scenario,
                               'day', 'atmos', 'day', 'r1i1p1',
                               version_folder, gcm_var)
        data = load_gcm_variable(var_dir, gcm_var, years_to_apply)
        if data is not None:
            pressure_data[era_var] = data
        else:
            print(f"Data for pressure var {gcm_var} not loaded.")

    return surface_data, pressure_data

def process_data_variables(data_dict, lat_min, lat_max, lon_min, lon_max,
                           target_shape):
    processed_data = {}
    for var_name, data in data_dict.items():
        print(f"Processing surface variable {var_name}")
        # rename if needed
        if 'latitude' in data.coords:  data = data.rename({'latitude': 'lat'})
        if 'longitude' in data.coords: data = data.rename({'longitude': 'lon'})
        # sort lat/lon
        if data.lat[0] > data.lat[-1]:
            data = data.sortby('lat')
        if data.lon[0] > data.lon[-1]:
            data = data.sortby('lon')
        # region subset
        data = data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        if data.size == 0:
            print(f"WARNING: {var_name} is empty after selection.")
            continue
        # interpolation
        new_lats = np.linspace(data.lat.values.min(), data.lat.values.max(), target_shape[0])
        new_lons = np.linspace(data.lon.values.min(), data.lon.values.max(), target_shape[1])
        data = data.interp(lat=new_lats, lon=new_lons, method='linear')
        processed_data[var_name] = data
    return processed_data

def process_pressure_data_variables(data_dict, lat_min, lat_max, lon_min, lon_max,
                                    target_shape, pressure_levels):
    processed_data = {}
    for var_name, data in data_dict.items():
        print(f"Processing pressure variable {var_name}")
        if 'latitude' in data.coords:   data = data.rename({'latitude': 'lat'})
        if 'longitude' in data.coords:  data = data.rename({'longitude': 'lon'})
        if 'plev' in data.coords:       data = data.rename({'plev': 'level'})
        # sort lat/lon
        if data.lat[0] > data.lat[-1]:
            data = data.sortby('lat')
        if data.lon[0] > data.lon[-1]:
            data = data.sortby('lon')
        data = data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # convert pressure levels to Pa
        pl_pa = [int(p)*100 for p in pressure_levels]
        avail_levels = data.level.values
        sel_levels   = [p for p in pl_pa if p in avail_levels]
        if not sel_levels:
            print(f"No matching p-levels found for {var_name}")
            continue
        data = data.sel(level=sel_levels)
        # interpolation
        new_lats = np.linspace(data.lat.values.min(), data.lat.values.max(), target_shape[0])
        new_lons = np.linspace(data.lon.values.min(), data.lon.values.max(), target_shape[1])
        data = data.interp(lat=new_lats, lon=new_lons, method='linear')
        processed_data[var_name] = data
    return processed_data

def coarsen_data(data_array, target_shape):
    """
    Coarsen data to target shape with 'area' pooling.
    data_array shape: (time, variables, lat, lon)
    """
    time_steps, variables, h, w = data_array.shape
    data_tensor = torch.from_numpy(data_array).float()  # [T, C, H, W]
    data_tensor = data_tensor.view(time_steps*variables, 1, h, w)
    coarsened   = nn.functional.interpolate(data_tensor, size=target_shape, mode='area')
    coarsened   = coarsened.view(time_steps, variables, target_shape[0], target_shape[1])
    return coarsened.numpy()

def save_to_netcdf(data_array, filename, latitudes, longitudes, times):
    """
    Save data_array (time, var, lat, lon) to netCDF as 'pr'.
    """
    ds = xr.Dataset(
        {'pr': (('time','lat','lon'), data_array[:,0,:,:])},
        coords={'time': times, 'lat': latitudes, 'lon': longitudes}
    )
    ds.to_netcdf(filename)
    

    
###################

# Interpolate to model input resolution (144, 272)
def interpolate_to_model_resolution(data_tensor, target_shape):
    """
    Interpolate data tensor to the target shape.

    :param data_tensor: Torch tensor with shape (time, variables, lat, lon) or (time, variables, level, lat, lon)
    :param target_shape: Tuple (height, width)
    :return: Interpolated data tensor
    """
    if len(data_tensor.shape) == 4:
        # Surface data
        n, c, h, w = data_tensor.shape
        data_tensor = data_tensor.view(n * c, 1, h, w)
        interpolated_data = nn.functional.interpolate(data_tensor, size=target_shape, mode='bilinear', align_corners=False)
        interpolated_data = interpolated_data.view(n, c, target_shape[0], target_shape[1])
    elif len(data_tensor.shape) == 5:
        # Pressure data
        n, c, l, h, w = data_tensor.shape
        data_tensor = data_tensor.view(n * c * l, 1, h, w)
        interpolated_data = nn.functional.interpolate(data_tensor, size=target_shape, mode='bilinear', align_corners=False)
        interpolated_data = interpolated_data.view(n, c, l, target_shape[0], target_shape[1])
    else:
        raise ValueError("Invalid data tensor shape")
    return interpolated_data



######################################
# Plotting utility
######################################
shapefile_path = '/g/data/hn98/Declan/Downscale-SR/Output_Files/0.5deg_0.25deg/Visuals/ne_110m_coastline/ne_110m_coastline.shp'
coastline_shp  = shapereader.Reader(shapefile_path)

def plot_data_sample(data_array, variable_index, latitudes, longitudes, title, filename):
    """
    data_array shape (variables, lat, lon) or (lat, lon).
    """
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_geometries(coastline_shp.geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='black')

    if data_array.ndim == 3:
        data = data_array[variable_index]  # (lat, lon)
    else:
        data = data_array

    mesh = ax.pcolormesh(longitudes, latitudes, data,
                         transform=ccrs.PlateCarree(),
                         cmap='viridis', shading='auto')
    plt.title(title)
    plt.colorbar(mesh, label='Precipitation')
    plt.savefig(filename)
    plt.close()

#######################################
# 4) Main Script
#######################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------------------------------
    # 4.1) Load the precomputed normalization stats from JSON
    # -----------------------------------------------------
    with open(normal_stats_json,'r') as f:
        all_stats = json.load(f)

    # For the 4 surface vars (assuming order [10u, 10v, 2t, tp]):
    var_order = ['10u','10v','2t','tp']
    fine_means = all_stats['fine_means']  # e.g. dict { '10u':val, '10v':val, ... }
    fine_stds  = all_stats['fine_stds']

    # Build arrays of shape (1,4,1,1) so we can broadcast over (time,4,lat,lon)
    mean_arr = []
    std_arr  = []
    for v in var_order:
        mean_arr.append(fine_means[v])
        std_arr.append(fine_stds[v])

    mean_arr = np.array(mean_arr, dtype=np.float32).reshape(1,len(var_order),1,1)
    std_arr  = np.array(std_arr,  dtype=np.float32).reshape(1,len(var_order),1,1)

    # -----------------------------------------------------
    # 4.2) Load historical data, process, convert to arrays
    # -----------------------------------------------------
    surface_data_hist, pressure_data_hist = load_gcm_data(
        VARIABLE_MAPPING, YEARS_TO_APPLY, GCM_PATH_MIDDLE,
        scenario='historical', version_folder=HISTORICAL_VERSION
    )
    surface_data_processed_hist = process_data_variables(
        surface_data_hist, lat_min, lat_max, lon_min, lon_max, model_coarse_shape
    )
    pressure_data_processed_hist = process_pressure_data_variables(
        pressure_data_hist, lat_min, lat_max, lon_min, lon_max,
        model_coarse_shape, PRESSURE_LEVELS
    )

    # Stack historical surface data => shape (time, #vars, lat, lon)
    hist_surf_tensors = []
    for v in var_order:
        if v in surface_data_processed_hist:
            hist_surf_tensors.append(surface_data_processed_hist[v].to_numpy())
        else:
            print(f"WARNING: {v} not found in historical processed data.")
    hist_surface_array = np.stack(hist_surf_tensors, axis=1)  # (time,4,lat,lon)

    # Stack historical pressure data => shape (time, #vars, level, lat, lon)
    # ... only needed if you feed pressure into your model
    # For demonstration, we'll do it but might not use it:
    hist_press_vars = list(pressure_data_processed_hist.keys())
    hist_press_tensors = []
    for v in hist_press_vars:
        arr = pressure_data_processed_hist[v].to_numpy()
        hist_press_tensors.append(arr)  # shape (time, level, lat, lon)
    if len(hist_press_tensors)>0:
        hist_pressure_array = np.stack(hist_press_tensors, axis=1)
    else:
        hist_pressure_array = None

    # Interpolate to 0.25 deg (Model_shape)
    surf_tensor_hist  = torch.from_numpy(hist_surface_array).float()
    surf_tensor_hist  = interpolate_to_model_resolution(surf_tensor_hist, Model_shape)

    # If you want pressure too, do similarly:
    if hist_pressure_array is not None:
        press_tensor_hist = torch.from_numpy(hist_pressure_array).float()
        press_tensor_hist = interpolate_to_model_resolution(press_tensor_hist, Model_shape)
    else:
        press_tensor_hist = None

    # -----------------------------------------------------
    # 4.3) Normalization with loaded stats
    # -----------------------------------------------------
    # shape => (time, 4, lat, lon)
    # broadcast mean/std over time, lat, lon
    surf_data_hist = surf_tensor_hist.numpy()  # (time,4,lat,lon)
    norm_surf_hist = (surf_data_hist - mean_arr) / std_arr

    # We won't specifically handle pressure normalization here,
    # unless you also have "fine_means/stds" for the pressure variables.
    # If the model was only trained on the 4 surface channels (+ mask),
    # you can skip normalizing pressure or do something consistent with training.

    # Convert to torch
    norm_surf_hist_torch = torch.tensor(norm_surf_hist, dtype=torch.float32)

    # -----------------------------------------------------
    # 4.4) Load future data (rcp85), process, convert
    # -----------------------------------------------------
    surf_data_fut, press_data_fut = load_gcm_data(
        VARIABLE_MAPPING, FUTURE_YEARS_TO_APPLY, GCM_PATH_MIDDLE,
        scenario='rcp85', version_folder=FUTURE_VERSION
    )
    surf_data_processed_fut = process_data_variables(
        surf_data_fut, lat_min, lat_max, lon_min, lon_max, model_coarse_shape
    )
    press_data_processed_fut = process_pressure_data_variables(
        press_data_fut, lat_min, lat_max, lon_min, lon_max,
        model_coarse_shape, PRESSURE_LEVELS
    )
    # Future surface
    fut_surf_tensors = []
    for v in var_order:
        if v in surf_data_processed_fut:
            fut_surf_tensors.append(surf_data_processed_fut[v].to_numpy())
        else:
            print(f"WARNING: {v} not found in future processed data.")
    fut_surface_array = np.stack(fut_surf_tensors, axis=1)
    # Future pressure
    fut_press_vars = list(press_data_processed_fut.keys())
    fut_press_tensors = []
    for v in fut_press_vars:
        arr = press_data_processed_fut[v].to_numpy()
        fut_press_tensors.append(arr)
    if len(fut_press_tensors)>0:
        fut_pressure_array = np.stack(fut_press_tensors, axis=1)
    else:
        fut_pressure_array = None

    # Interpolate future to 0.25 deg
    fut_surf_tensor  = torch.from_numpy(fut_surface_array).float()
    fut_surf_tensor  = interpolate_to_model_resolution(fut_surf_tensor, Model_shape)
    if fut_pressure_array is not None:
        fut_press_tensor = torch.from_numpy(fut_pressure_array).float()
        fut_press_tensor = interpolate_to_model_resolution(fut_press_tensor, Model_shape)
    else:
        fut_press_tensor = None

    # Normalize future
    fut_surf_data       = fut_surf_tensor.numpy()
    norm_fut_surf_data  = (fut_surf_data - mean_arr) / std_arr
    norm_fut_surf_torch = torch.tensor(norm_fut_surf_data, dtype=torch.float32)

    # -----------------------------------------------------
    # 4.5) Load your EDM diffusion model & checkpoint
    # -----------------------------------------------------
    diffusion_model = EDMPrecond(
        img_resolution=(144,272),
        in_channels= 11, #11 for 4 variables + 4 conditions + 3 surface masks#len(var_order),   # e.g. 4 if [10u,10v,2t,tp] #Not just our four 
        out_channels=len(var_order),   # same # of channels
        label_dim=4,                   #2 for day/hour
        sigma_min=0.002,
        sigma_max=30.0, #1.0
        sigma_data=1.0,
        use_fp16=False,                # adjust if you use fp16
        model_type='UNet'              # or your custom model if needed
    ).to(device)

    diffusion_model.load_state_dict(torch.load(os.path.join(save_dir, diffusion_ckpt)))
    diffusion_model.eval()

    # -----------------------------------------------------
    # 4.6) Prepare surface mask (soil_type, topography, etc.)
    # -----------------------------------------------------
    soil_type   = np.load('/g/data/hn98/Declan/soil_type.npy')
    topography  = np.load('/g/data/hn98/Declan/topography.npy')
    land_sea    = np.load('/g/data/hn98/Declan/land_mask.npy')
    # Subset
    latitudes_full  = np.linspace(90, -90, 721)
    longitudes_full = np.linspace(0, 360, 1440, endpoint=False)
    lat_inds = np.where((latitudes_full >= lat_min) & (latitudes_full <= lat_max))[0]
    lon_inds = np.where((longitudes_full >= lon_min) & (longitudes_full <= lon_max))[0]

    soil_sub   = soil_type[np.ix_(lat_inds, lon_inds)]
    topo_sub   = topography[np.ix_(lat_inds, lon_inds)]
    lsm_sub    = land_sea[np.ix_(lat_inds, lon_inds)]

    mask_stack     = np.stack([soil_sub, topo_sub, lsm_sub], axis=0)  # shape (3, lat, lon)
    surface_mask   = torch.tensor(mask_stack, dtype=torch.float32).to(device)
    
    #NORMALISE SURFACE DATA
    surface_mask = (surface_mask - surface_mask.mean()) / surface_mask.std()

    # We might need to broadcast the mask to batch dimension each iteration
    
    
    
    ######################
    #NOTE HERE - WE ARE GOING TO USE IMPORT A DUMMY ERA5 DATASET TO USE THE 'RESIDUAL_TO_FINE_IMAGE' FUNCTION CREATED AS A PART OF THAT CLASS - TOO LAZY TO REPLCIATE HERE
    from src.DatasetAUS import UpscaleDataset
    
    # Load normalization stats
    with open(normal_stats_json, 'r') as f: #THE STATS HERE RIGHT NOW ARE FOR YOUR 30-YEAR DATA MODEL
        precomputed_stats = json.load(f)
    

    # Define your validation period (adjust as needed)
    val_years = [2001]  # example, change as needed
    val_months = [2] # example, change as needed
    base_shape = (144, 272)   # HR shape
    in_shape   = (24, 46)     # Coarse shape
    out_shape = base_shape
    
    print("\n=== Running Validation Evaluation ===")

    # Load validation dataset
    dataset_val = UpscaleDataset(
        data_dir="/g/data/rt52/era5/single-levels/reanalysis/",
        year_start=min(val_years),
        year_end=max(val_years),
        in_shape=in_shape,
        out_shape=out_shape,
        variables=['10u', '10v', '2t', 'tp'],  # same variables as train/test
        months=val_months,
        aggregate_daily=True,
        precomputed_stats=precomputed_stats  # use the same normalization stats
    )

    dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    ######################

    # -----------------------------------------------------
    # 4.7) Historical & Future inferences (3 samples each)
    # -----------------------------------------------------
    # Helper: get lat/lon/time for final saving
    hist_times = surface_data_processed_hist[var_order[0]].time.values
    fut_times  = surf_data_processed_fut[var_order[0]].time.values

    # For 0.5 deg saving
    lat_05deg = np.linspace(lat_min, lat_max, final_coarse_shape[0])
    lon_05deg = np.linspace(lon_min, lon_max, final_coarse_shape[1])
    
    
    # (A) Define sampling params exactly as training (shouldn't matter though as training is done over all noise levels)
    num_steps  = 50
    sigma_min  = 0.002
    sigma_max  = 30.0 #1.0 #30 #80.0 #CHANGED IN EDMPRECOND TOO ABOVE
    rho        = 4 #7
    S_churn    = 0 #1 #5 #20 #Your older diffusion output 
    S_min      = 0
    S_max      = float('inf')
    S_noise    = 1 #5
    
    ######
    ################################################################################
    # 1) Create a scheduler for e.g. 50 steps
    ################################################################################



    hier_scheduler = HierarchicalScheduler(
        full_size=(Model_shape[0], Model_shape[1]),  # (144,272)
        total_steps=num_steps,                              # or however many steps
        device=device
    )

    #Reverse scheduler to match hierarchical sampling
    hier_scheduler.shape_list.reverse()  # e.g. from (144,272) ->...->(1,1)



    #REDEFINE SCHEDULER HERE BELOW:
    
    SAMPLING_STEPS_INFERENCE = num_steps #Defined just above
    #OLD SCHEDULER
    NSE_STEPS_PER_SPLIT = 3
    
    hier_scheduler = OLDHierarchicalScheduler( #CREATES THE SCHEDULE OVER WHICH WE ADD NOISE VS REDUCE IMAGE SIZE (HOW OFTEN TO DO HIERARCHICAL VS DENOISING?)
        full_size=(144,272),
        total_steps=SAMPLING_STEPS_INFERENCE,
        x_over_y_ratio=(272/144), #
        noise_steps_per_split=NSE_STEPS_PER_SPLIT,
        shape_splits_per_shrink=1, #DO NOT AMEND THIS EVER - CODE DOES NOTHING HERE AND YOU COMMENTED OUT BELOW
        use_exponential=False,
        device=device
    )

    #Was returning 262 as the last shpae 
    hier_scheduler.shape_list.reverse()  # e.g. from (144,272) ->...->(1,1)





    ################################################################################
    # 2) AMENDED HISTORICAL PORTION TO USE HIERARCHICAL SAMPLER
    ################################################################################
    for sample_idx in range(1, 2): # (1, 4)
        print(f"\n=== HISTORICAL RUN #{sample_idx} ===")
        predicted_precip_list = []
        gif_frames            = []  #FOR GIF
        n_days = norm_surf_hist_torch.shape[0]

        with torch.no_grad():
            for day_idx in tqdm(range(n_days)):
                # shape => (1,4,lat,lon) = (1,4,144,272)
                inp_surf_day = norm_surf_hist_torch[day_idx:day_idx+1].to(device)

                NUM_DAYS_GIF = 5 #Generate it for the first 5 days
                # ONLY DO IT FOR THE FIRST FEW DAYS 
                # 2a) sample hierarchically
                if day_idx < NUM_DAYS_GIF:
                    x_next_residual = sample_model_hierarchical(
                        model=diffusion_model,
                        images_input=inp_surf_day,     # the normalized coarse/residual input
                        coarse=inp_surf_day,           # for simplicity, pass same if your net expects it
                        scheduler=hier_scheduler,
                        device=device,
                        num_inference_steps=num_steps,        # same as scheduler
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        rho=rho,
                        S_churn=S_churn,
                        S_min = S_min,
                        S_max = S_max,
                        S_noise=S_noise,
                        surface_mask = surface_mask,
                        dataset = dataset_val,
                        make_gif = True,
                    )



                # 2a) sample hierarchically
                x_next_residual = sample_model_hierarchical(
                    model=diffusion_model,
                    images_input=inp_surf_day,     # the normalized coarse/residual input
                    coarse=inp_surf_day,           # for simplicity, pass same if your net expects it
                    scheduler=hier_scheduler,
                    device=device,
                    num_inference_steps=num_steps,        # same as scheduler
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    S_churn=S_churn,
                    S_min = S_min,
                    S_max = S_max,
                    S_noise=S_noise,
                    surface_mask = surface_mask,
                    dataset = dataset_val,
                    make_gif = False,
                )

                # 2b) Convert residual => final 4-ch. If your training was “coarse + residual => fine”
                #    then 'x_next_residual' is (1,4,H,W).  We do:
                predicted_4ch = dataset_val.residual_to_fine_image(
                    x_next_residual.cpu(), inp_surf_day.cpu()
                )

                # 2c) Extract precipitation (channel=3 => 'tp')
                pred_precip = predicted_4ch[:, 3:4, :, :]  # shape (1,1,H,W)

                # 2d) Denormalize if needed:  pred_precip * std(tp) + mean(tp)
                #    since you already have mean_arr, std_arr, etc.:
                # e.g. if channel=3 => idx=3 in your stats array
                #tp_denorm = pred_precip.numpy() * std_arr[:,3:4] + mean_arr[:,3:4]

                #PREVIOUS RUN WAS USING THE 'RESIDUAL-TO-FINE-IMAGE' FUNCTION WITH INCORRECT STATS (CHANGED) AND WAS DOUBLE UNNORMALISING BY USING TP_DENORM
                predicted_precip_list.append(pred_precip) #tp_denorm

        # Convert predicted_precip_list => (time,1,H,W)
        predicted_precip_4d = np.concatenate(predicted_precip_list, axis=0)

        # Then do the rest: coarsen to 0.5 deg, save, plot, etc.
        predicted_precip_4d_coarse = coarsen_data(predicted_precip_4d, final_coarse_shape)
        out_nc = f"Downscaled_Products/historical_diffusion_output_{sample_idx}_hier_cerfacs_sv_n.nc"
        save_to_netcdf(predicted_precip_4d_coarse, out_nc, lat_05deg, lon_05deg, hist_times)


        # Plot sample day (day 0)
        if predicted_precip_4d.shape[0] > 0:
            sample_day = 0
            # 0.25 deg
            plot_data_sample(
                predicted_precip_4d[sample_day], 0,
                np.linspace(lat_min, lat_max, Model_shape[0]),
                np.linspace(lon_min, lon_max, Model_shape[1]),
                f"Hist Precip (Diffusion) run#{sample_idx} - 0.25deg",
                f"hist_precip_0.25deg_run{sample_idx}.png"
            )
            # 0.5 deg
            plot_data_sample(
                predicted_precip_4d_coarse[sample_day], 0,
                lat_05deg, lon_05deg,
                f"Hist Precip (Diffusion) run#{sample_idx} - 0.5deg",
                f"hist_precip_0.5deg_run{sample_idx}.png"
            )




    ###########################################


    '''

    #FUTURE - NOT USING NOW SO KEEP THE SAME
    # =========== FUTURE - 3 runs ===========
    for sample_idx in range(1, 4):
        print(f"\n=== FUTURE RUN #{sample_idx} ===")
        predicted_precip_list = []
        n_days = norm_fut_surf_torch.shape[0]

        with torch.no_grad():
            for day_idx in tqdm(range(n_days)):
                inp_surf_day = norm_fut_surf_torch[day_idx:day_idx+1].to(device)
                sigma_day    = torch.full((1,), 0.002, device=device)
                
                
                # Concatenate your coarse data (4 ch) with mask (3 ch) => 7 ch total
                # Mask needs a batch dim => (1,3,lat,lon)
                mask_3ch     = surface_mask.unsqueeze(0)  # shape (1,3,lat,lon)
                full_condition = torch.cat([inp_surf_day, mask_3ch], dim=1)  
                # shape => (1,4+3, lat,lon) = (1,7, lat,lon)
        

                pred_surface = diffusion_model(
                    x=inp_surf_day,
                    sigma=sigma_day,
                    condition_img=full_condition,
                    class_labels=None,
                    force_fp32=True
                )
                pred_precip = pred_surface[:,3:4,:,:]
                tp_denorm   = pred_precip.cpu().numpy() * std_arr[:,3:4] + mean_arr[:,3:4]
                predicted_precip_list.append(tp_denorm)

        predicted_precip_4d = np.concatenate(predicted_precip_list, axis=0)
        predicted_precip_4d_coarse = coarsen_data(predicted_precip_4d, final_coarse_shape)

        out_nc = f"Downscaled_Products/future_diffusion_output_cerfacs_{sample_idx}.nc"
        save_to_netcdf(predicted_precip_4d_coarse, out_nc, lat_05deg, lon_05deg, fut_times)

        # Plot sample day
        if predicted_precip_4d.shape[0] > 0:
            sample_day = 0
            plot_data_sample(
                predicted_precip_4d[sample_day], 0,
                np.linspace(lat_min, lat_max, Model_shape[0]),
                np.linspace(lon_min, lon_max, Model_shape[1]),
                f"Fut Precip (Diffusion) run#{sample_idx} - 0.25deg",
                f"fut_precip_0.25deg_run{sample_idx}.png"
            )
            plot_data_sample(
                predicted_precip_4d_coarse[sample_day], 0,
                lat_05deg, lon_05deg,
                f"Fut Precip (Diffusion) run#{sample_idx} - 0.5deg",
                f"fut_precip_0.5deg_run{sample_idx}.png"
            )
    '''
    print("\nAll done! Created 3 historical runs using precomputed normalization stats. No Future done for now")


if __name__ == "__main__":
    main()
