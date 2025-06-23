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
normal_stats_json    = "normalisation_stats_ALL_AUS.json"  #NEED TO CHANGE THE NORMALISATION FILE AS WELL HERE FOR 20YR# precomputed stats
diffusion_ckpt       = "Equal_Aus_MASS380-30yr.pt" #Equal_Aus_MASS119-240-30yr.pt #"Equal_Aus_MASS360-30yr.pt" #Equal_Aus_MASS199-20yr.pt" #"Equal_Aus_150-10yr-no-mass-loss.pt" #"Equal_Aus_30.pt"  #Australia over 30 epochs, 2 years data?      # your trained EDM checkpoint

# GCM specifics

GCM_PATH_MIDDLE      = 'CNRM-CERFACS/CNRM-CM5'
HISTORICAL_VERSION   = 'v20120530'
FUTURE_VERSION       = 'v20130101'
#
'''
GCM_PATH_MIDDLE = 'MIROC/MIROC5' # 'CNRM-CERFACS/CNRM-CM5'  # Replace with desired GCM
HISTORICAL_VERSION = 'v20120710'  # Version subfolder for historical data
FUTURE_VERSION = 'v20120710'     # Version subfolder for future data
'''

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
        label_dim=2,                   #2 for day/hour
        sigma_min=0.002,
        sigma_max=80.0,
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
    with open('normalisation_stats_ALL_AUS.json', 'r') as f: #THE STATS HERE RIGHT NOW ARE FOR YOUR 30-YEAR DATA MODEL
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
    
    
    # (A) Define sampling params exactly as training
    num_steps  = 50 
    sigma_min  = 0.002
    sigma_max  = 80.0
    rho        = 7
    S_churn    = 5 #20
    S_min      = 0
    S_max      = float('inf')
    S_noise    = 1 #5
    
    
    

    # =========== HISTORICAL - 3 runs ===========
    for sample_idx in range(1, 2): #BEFORE WAS (1, 4) - CHANGED HERE TO (1, 1)
        print(f"\n=== HISTORICAL RUN #{sample_idx} ===")
        predicted_precip_list = []
        n_days = norm_surf_hist_torch.shape[0]
        
        
        

        with torch.no_grad():
            for day_idx in tqdm(range(n_days)):
                # shape => (1,4,lat,lon)
                inp_surf_day = norm_surf_hist_torch[day_idx:day_idx+1].to(device)
                

                # (A) Define inputs (Done above)
                # (B) Create initial noise shape = same as your 4-ch 'inp_surf_day'
                init_noise = torch.randn_like(inp_surf_day, dtype=torch.float64)  # e.g. (1,4,H,W)
                
                # (C) Discretize the time steps
                step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
                t_steps = (sigma_max ** (1 / rho)
                          + step_indices/(num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
                # Append final t=0
                t_steps = torch.cat([diffusion_model.round_sigma(t_steps),
                                     torch.zeros_like(t_steps[:1])])
                x_next = init_noise * t_steps[0]  # multiply noise by t_0
                
                # (D) Multi-step loop
                for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    x_cur = x_next
                    
                    # Possibly add "churn" noise
                    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
                    t_hat = diffusion_model.round_sigma(t_cur + gamma * t_cur)
                    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)
                
                    # Build the 7-channel condition each iteration
                    #   4-ch running state (x_hat)
                    #   + 3-ch mask  (assuming you already have full_condition = cat([inp_surf_day, mask_3ch])
                    #   but for multi-step, you'll want to cat x_hat instead of inp_surf_day:
                    mask_3ch     = surface_mask.unsqueeze(0).to(torch.float64)  # (1,3,H,W)
                    # => shape (1,4 + 3, H,W) = (1,7,H,W) if you only want 7 total
                    #multi_condition = torch.cat([x_hat, mask_3ch], dim=1)
                    

                    #NEED TO FIX MODEL BELOW HERE SO IT APPLIES DIFFUSION STEP AS EXPECTED
                    
                    #DOUBLE CHECK - IS IT APPLYING THE BELOW STEP PER HOW IS DONE IN 'TRAIN-DIF-AUS.PY???'
                
                    # Euler step
                    #denoised = diffusion_model(multi_condition, t_hat, class_labels=None).to(torch.float64)
                    
                    #Debugging - first input is whatever stage of denoising process we are at (denoised image), second is current noise level, third is full condition and last is the condition params (day/hour scalar values passed to model etc. - don't do last here yet)
                   
                    full_condition = torch.cat([inp_surf_day,mask_3ch], dim = 1)
                    
                    denoised = diffusion_model(x_hat, t_hat, full_condition, class_labels=None).to(torch.float64)
                    
                    
                    
                    d_cur    = (x_hat - denoised) / t_hat
                    x_next   = x_hat + (t_next - t_hat)*d_cur
                
                    # 2nd-order correction
                    if i < num_steps - 1:
                        #multi_condition2 = torch.cat([x_next, mask_3ch], dim=1)
                        #denoised2 = diffusion_model(multi_condition2, t_next, class_labels=None).to(torch.float64)
                        #multi_condition2 = torch.cat([x_next, mask_3ch], dim=1)
                        denoised2 = diffusion_model(x_hat, t_hat, full_condition, class_labels=None).to(torch.float64)
                        
                        d_prime   = (x_next - denoised2) / t_next
                        x_next    = x_hat + (t_next - t_hat)*0.5*(d_cur + d_prime)
                
                # (E) Now x_next is your final 4-ch residual. Convert to precipitation:
                predicted_4ch = dataset_val.residual_to_fine_image(x_next.float().cpu(), inp_surf_day.cpu()) #USING DATASET_VAL HERE BUT THE NORMED VALUES WILL BE THE SAVED ONES FROM THE .JSON FILE #dataloader.dataset.residual_to_fine_image(x_next.float().cpu(), inp_surf_day.cpu())
                pred_precip   = predicted_4ch[:, 3:4, :, :]  # channel 3 => 'tp'
                
                # (F) Append to list 
                #tp_denorm = pred_precip.numpy() * std_arr[:,3:4] + mean_arr[:,3:4]
                predicted_precip_list.append(pred_precip) #tp_denorm
                # --------------------------------------------------------------
    
    
    
    
                
    
        


        # (time,1,lat,lon)
        predicted_precip_4d = np.concatenate(predicted_precip_list, axis=0)

        # Coarsen to 0.5 deg
        predicted_precip_4d_coarse = coarsen_data(predicted_precip_4d, final_coarse_shape)

        # Save
        out_nc = f"Downscaled_Products/historical_diffusion_output_cerfacs_EDM{sample_idx}.nc"
        os.makedirs("Downscaled_Products", exist_ok=True)
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


    #FUTURE - NOT USING NOW SO KEEP THE SAME
    # =========== FUTURE - 3 runs ===========
    for sample_idx in range(1, 1): #BEFORE WAS (1, 4) - CHANGED HERE TO (1, 1)
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

    print("\nAll done! Created 3 historical + 3 future runs using precomputed normalization stats.")


if __name__ == "__main__":
    main()
