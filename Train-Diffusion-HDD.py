# -*- coding: utf-8 -*-
###########################

import pkg_resources

installed_packages = pkg_resources.working_set
for package in installed_packages:
    print(f"{package.key} ({package.version})")



###########################


#Import eval metrics from eval diffusion
from Eval_diffusion import calculate_rmse, calculate_psnr, calculate_ssim, sample_model



import torch
#from src.Network_FPN import EDMPrecond
#from src.Network_FPN import EDMPrecond #Try seeing if you can just up the network size and make it the same as the others by making it the same size via importing JUST network rather than network_FPN

#CHANGED ABOVE TO BELOW - SHOULD STILL WORK AS NETWORKS ARE THE SAME
from src.Network import EDMPrecond

#import src.Network #Just import lots of the individual functions rather than 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from src.DatasetAUS import UpscaleDataset #src.DatasetUS import UpscaleDataset
from torch.utils.tensorboard import SummaryWriter
import os
#import dill as pickle







# Loss class taken from EDS_Diffusion/loss.py
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None,
                 augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        
        
        
        # **Add print statements here**
        #print(f"Inside EDMLoss __call__:")
        #print(f"images.shape: {images.shape}")                          # Expected: [batch_size, 4, H, W]
        #print(f"conditional_img.shape: {conditional_img.shape if conditional_img is not None else 'None'}")
        #print(f"labels.shape: {labels.shape if labels is not None else 'None'}")
        #print(f"y.shape: {y.shape}")                                    # Should match images.shape
        #print(f"n.shape: {n.shape}")                                    # Should match y.shape
        #print(f"(y + n).shape: {(y + n).shape}")                        # Input to the network
        #print(f"sigma.shape: {sigma.shape}")
        
        
        
        D_yn = net(y + n, sigma, conditional_img, labels,
                   augment_labels=augment_labels)
                   
                   
        #Previously, loss is computed here amongst all variables - precipitation seems to perform much better than the others (?)
        #loss = weight * ((D_yn - y) ** 2)
                   
        #Have tried a change where loss is calculated equally across all variables
                   
        per_variable_loss = weight * ((D_yn - y) ** 2)
        mean_loss_per_variable = per_variable_loss.view(per_variable_loss.size(0), per_variable_loss.size(1), -1).mean(dim=2)
        
        # Compute total loss
        loss = mean_loss_per_variable.mean()
        
                   

        
        
        return loss, D_yn #Added the returning of the predicted output so we can calculate mass conservation loss
    
"""
Function for a single training step.
:param model: Instance of the Unet class
:param loss_fn: Loss function
:param optimiser: Optimiser to use
:param data_loader: Data loader
:param scaler: Scaler for mixed precision training
:param step: Current step
:param accum: Number of steps to accumulate gradients over
:param writer: Tensorboard writer
:param device: Device to use
:return: Loss value
"""

def training_step(model, loss_fn, optimiser, data_loader, scaler, step, accum=4, writer=None,surface_mask_batch=None,device="cuda"):

    model.train()
    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        step_loss = 0
        for i, batch in enumerate(data_loader):
            tq.update(1)

            image_input = batch["coarse"].to(device) #batch["inputs"].to(device)
            image_output = batch["fine"].to(device) #batch["targets"].to(device)
            day = batch["doy"].to(device)
            hour = batch["hour"].to(device)
            condition_params = torch.stack((day, hour), dim=1)
            

            #DEC - PRINT STATEMENTS            
            #print(f"Batch {i}:")
            #print(f"image_input.shape: {image_input.shape}")    # Expected: [batch_size, 4, H, W]
            #print(f"image_output.shape: {image_output.shape}")  # Expected: [batch_size, 4, H, W]
            #print(f"condition_params.shape: {condition_params.shape}")  # Expected: [batch_size, 2]
            
            
            # Print min/max values of inputs
            print(f"Batch {i}:")
            print(f"image_input min: {image_input.min().item()}, max: {image_input.max().item()}")
            print(f"image_output min: {image_output.min().item()}, max: {image_output.max().item()}")
            print(f"condition_params min: {condition_params.min().item()}, max: {condition_params.max().item()}")

            
            
            full_condition = torch.cat([image_input, surface_mask_batch], dim=1)
            

            # forward unet
            with torch.cuda.amp.autocast():
                loss, D_yn = loss_fn(net=model, images=image_output, #CHANGED FROM 'loss_fn' to 'hierarchical_loss_fn' here
                               conditional_img=full_condition, #image_input
                               labels=condition_params)
                loss = torch.mean(loss)
            
            ###
            #Old    
            #mass_criterion = PixelWiseMassConservationLoss(scale_factor=2, weight=0.1).to(device)
            #mass_loss = mass_criterion(D_yn, image_input)
            ###
            #Define conservation of mass loss function
            mass_criterion = MassConservationLoss(weight=0.0000002).to(device) #100 #100000
            
            #############
            # Compute mass conservation loss (DO ONLY FOR SURFACE VARIABLES FOR NOW)
            print(f"Input size: {image_input.size()}") #TEST IT IS THE CORRECT SIZE
            print(f"Output size: {image_output.size()}")
            print(f"Output size of pred: {D_yn.size()}")
            #print(f"Input size: {image_input.shape()}") #TEST IT IS THE CORRECT SIZE
            #print(f"Output size: {image_output.shape()}")
            
            input_mass = calculate_mass(image_input) ###
            output_mass = calculate_mass(D_yn) ###
            
            print(input_mass.shape)
            print(output_mass.shape)
            
            mass_loss2 = mass_criterion(input_mass, output_mass) ###
            
            #total_loss = mse_loss + mass_loss ###
            #############
            
            

            total_loss = loss #+  mass_loss2 #Don't even take the first pixelwise mass conservation into account anymore: + mass_loss +
            
            #REMOVING MASS LOSS FROM LOSS FUCNTION FOR NOW - TEST HOW IT HAPPENS

            #What is driving the loss for the reverse noise UNET? 
            print(f"MSE loss is {loss}")
            print(f"Mass loss is {mass_loss2}")
            
            

            # backpropagation
            scaler.scale(total_loss).backward()
            step_loss += total_loss.item()

            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                if writer is not None:
                    writer.add_scalar("Loss/train", step_loss / accum,
                                      step * len(data_loader) + i)
                step_loss = 0

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")
    return mean_loss


@torch.no_grad()
def sample_model(model, dataloader, num_steps=10, sigma_min=0.002, #Keeping the num_steps as very low here - can compare to the 500 used in the sample_model_val()
                 sigma_max=80, rho=7, S_churn=40, S_min=0,
                 S_max=float('inf'), S_noise=1, device="cuda", surface_mask_batch=None): #S_churn=40 #sigma_max=80 #Noisy output debug - attempt to turn down the churn which appears to be noise added at inference and the sigma max - num steps were already increased to 100 earlier from 40

    batch = next(iter(dataloader))
    images_input = batch["coarse"].to(device) #batch["inputs"]
    
    coarse, fine = batch["coarse"], batch["fine"] #DEC - CHANGED TO MATCH VARIABLES FROM OTHER 'UPSCALEDATASET' FUNCTION #Should you do this? Take the  
    
    #####coarse_norm = (coarse - dataloader.dataset.coarse_mean) / dataloader.dataset.coarse_std_dev ##$

    #Old solution for upscale dataset was below
    #inputs = batch["inputs"].to(device)
    #targets = batch["targets"].to(device)


    condition_params = torch.stack(
        (batch["doy"].to(device),
         batch["hour"].to(device)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 4, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device) #DEC NOTE HERE - CHANGE 3 -> 4 PREVIOUSLY IT WAS: images_input.shape[0], 3, images_input.shape[2]

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))
                 
        print(f"surface_mask_batch is {surface_mask_batch.shape}")
                 
        full_condition = torch.cat([images_input, surface_mask_batch], dim=1)

        # Euler step.
        denoised = model(x_hat, t_hat, full_condition, condition_params).to( #x_hat, t_hat, images_input,
            torch.float64) #NEED TO ADD THE HIERARCHY MODELLING HERE AS A CONDITION????
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, full_condition, #x_next, t_next, images_input,
                             condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataloader.dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse) ##$ coarse -> coarse_norm
        
        
    
    # NORMALIZED RESULTS
    print("NORMALIZED RESULTS")
    for i, var_name in enumerate(dataloader.dataset.variables):
        print(f"{var_name} - Predicted min: {predicted[0, i].min().item()}, max: {predicted[0, i].max().item()}, mean: {predicted[0, i].mean().item()}")
        print(f"{var_name} - Coarse min: {coarse[0, i].min().item()}, max: {coarse[0, i].max().item()}, mean: {coarse[0, i].mean().item()}")
        print(f"{var_name} - Fine min: {fine[0, i].min().item()}, max: {fine[0, i].max().item()}, mean: {fine[0, i].mean().item()}")

    # For plotting, denormalize fine and coarse images if needed ##$
    coarse_denorm = dataloader.dataset.inverse_normalize_coarse(coarse) #coarse_norm
    fine_denorm = dataloader.dataset.inverse_normalize_data(fine)
    #predicted = dataloader.dataset.inverse_normalize_data(predicted) ###% 27/11/24 - this won't work but try inverse normalising here
    
    # UNNORMALIZED RESULTS (I.E., CORRECT FINAL ONES)
    print("\nUNNORMALIZED RESULTS (I.E., CORRECT FINAL ONES)")
    for i, var_name in enumerate(dataloader.dataset.variables):
        print(f"{var_name} - Predicted min: {predicted[0, i].min().item()}, max: {predicted[0, i].max().item()}, mean: {predicted[0, i].mean().item()}")
        print(f"{var_name} - Coarse min: {coarse_denorm[0, i].min().item()}, max: {coarse_denorm[0, i].max().item()}, mean: {coarse_denorm[0, i].mean().item()}")
        print(f"{var_name} - Fine min: {fine_denorm[0, i].min().item()}, max: {fine_denorm[0, i].max().item()}, mean: {fine_denorm[0, i].mean().item()}")


    print(f"\nScale used in plot_batch: vmin={dataloader.dataset.vmin}, vmax={dataloader.dataset.vmax}")

    
    #Normalise them again now to pass to the plot.... lol
    # Normalize coarse_denorm
    coarse_normalized = (coarse_denorm - dataloader.dataset.coarse_mean) / dataloader.dataset.coarse_std_dev
    coarse_normalized = coarse_normalized.float()
    
    
    # Normalize fine_denorm
    fine_normalized = (fine_denorm - dataloader.dataset.data_mean) / dataloader.dataset.data_std_dev
    fine_normalized = fine_normalized.float()
    
    # Normalize predicted
    predicted_normalized = (predicted - dataloader.dataset.data_mean) / dataloader.dataset.data_std_dev
    predicted_normalized = predicted_normalized.float()
    
    print("Coarse_OUT")
    print(dataloader.dataset.coarse_mean)
    print(dataloader.dataset.coarse_std_dev)
    
    print("Fine_OUT")
    print(dataloader.dataset.data_mean)
    print(dataloader.dataset.data_std_dev)
    
    print("residual_OUT")
    print(dataloader.dataset.residual_mean)
    print(dataloader.dataset.residual_std_dev)


    fig, ax = dataloader.dataset.plot_batch(coarse_normalized, fine_normalized, predicted_normalized) ##$ (plotting the denormed ones now)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    base_error = torch.mean(torch.abs(fine - coarse))
    pred_error = torch.mean(torch.abs(fine - predicted))

    return (fig, ax), (base_error.item(), pred_error.item())
#

'''
#AMENDED FOR NCI
def main(downscale_factor=8):
    batch_size = 1 #Reduced from 8 to 2 given memory constraints
    learning_rate = 1e-4
    num_epochs = 5
    accum = 8

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = EDMPrecond((721, 1440), 8, 4, label_dim=2)  # Update to match new data shape and variable count
    network.to(device)

    # Define the datasets
    datadir = '/g/data/rt52/era5/single-levels/reanalysis/' #datadir = os.getenv('ROOT_DIR')  # Use the NCI directory

    # Calculate input and output shapes
    base_shape = (721, 1440)  # Updated shape to reflect the NCI dataset resolution
    downscale_factor = 2  # Assuming downscaling by a factor of 2
    in_shape = (base_shape[0] // downscale_factor, base_shape[1] // downscale_factor)  # Coarse shape
    out_shape = base_shape  # High-resolution shape remains the same

    # Use the updated UpscaleDataset function (shown below)
    dataset_train = UpscaleDataset(datadir, year_start=2000, year_end=2001,
                                   in_shape=in_shape, out_shape=out_shape,
                                   variables=['10u', '10v', '2t', 'tp'],  # Matching the variables used in NCI script
                                   months=[1],  # Define months to include
                                   aggregate_daily=True)  # Aggregate to daily

    dataset_test = UpscaleDataset(datadir, year_start=2000, year_end=2001,
                                  in_shape=in_shape, out_shape=out_shape,
                                  variables=['10u', '10v', '2t', 'tp'],  # Same variables as in the train dataset
                                  months=[1],  # Same months for consistency
                                  aggregate_daily=True)
'''



##################################
#MASS CONSERVATION LOSS

import torch
import torch.nn as nn

'''
class PixelWiseMassConservationLoss(nn.Module):
    def __init__(self, scale_factor=2, weight=1.0):
        """
        Initialize the pixel-wise mass conservation loss.
        :param scale_factor: The downscale factor (e.g., 2 for 2x).
        :param weight: Weight of the mass conservation term in the total loss.
        """
        super(PixelWiseMassConservationLoss, self).__init__()
        self.scale_factor = scale_factor
        self.weight = weight

    def forward(self, hr_pred, lr):
        """
        Compute the mass conservation loss at pixel level.
        :param hr_pred: HR prediction tensor of shape (N, C, H_hr, W_hr).
        :param lr: LR tensor of shape (N, C, H_lr, W_lr).
        :return: Scalar mass loss value.
        """
        N, C, H_hr, W_hr = hr_pred.shape
        _, _, H_lr, W_lr = lr.shape

        # Reshape HR to group each block of sxs pixels corresponding to one LR pixel
        # shape: (N, C, H_lr, scale_factor, W_lr, scale_factor)
        hr_reshaped = hr_pred.view(N, C, H_lr, self.scale_factor, W_lr, self.scale_factor)

        # Sum over the sxs block
        hr_sum = hr_reshaped.sum(dim=(3, 5))  # sum over the scale_factor dimensions

        # hr_sum now has shape (N, C, H_lr, W_lr) matching lr
        mass_loss = torch.mean((hr_sum - lr) ** 2)
        return self.weight * mass_loss
'''



class PixelWiseMassConservationLoss(nn.Module):
    def __init__(self, scale_factor=2, weight=1.0): ##% scale_factor=2, - CHANGING TO 6 DOESN'T WORK - WON'T DIVIDE PROPERLY ON 2ND DIMENSION (CHANGED FROM 2 -> 6 - FOR AUS DATASET(APPROXIMATE - NOT ONE-TO-ONE FOR (144,272) -> (24,46)) 
        super(PixelWiseMassConservationLoss, self).__init__()
        self.scale_factor = scale_factor
        self.weight = weight

    def forward(self, hr_pred, lr_upsampled):
        """
        hr_pred: (N, C, H_hr, W_hr) - model HR prediction
        lr_upsampled: (N, C, H_hr, W_hr) - LR data after being upsampled back to HR size
        We need to down-block-average lr_upsampled back to original LR shape to get LR pixels.
        """
        N, C, H_hr, W_hr = hr_pred.shape
        # Compute original LR dimensions
        H_lr = H_hr // self.scale_factor
        W_lr = W_hr // self.scale_factor

        # Block-average the upsampled LR to recover original LR values
        # shape: (N, C, H_lr, scale_factor, W_lr, scale_factor)
        lr_block = lr_upsampled.view(N, C, H_lr, self.scale_factor, W_lr, self.scale_factor)
        lr_recovered = lr_block.mean(dim=(3,5))  # shape: (N, C, H_lr, W_lr)

        # Sum the HR predictions in the corresponding sxs block
        hr_block = hr_pred.view(N, C, H_lr, self.scale_factor, W_lr, self.scale_factor)
        hr_sum = hr_block.sum(dim=(3,5))  # shape: (N, C, H_lr, W_lr)

        mass_loss = torch.mean((hr_sum - lr_recovered) ** 2)
        return self.weight * mass_loss


#################################





##########################
#ADD CONSERVATION OF MASS CONSTRAINTS
#(Above implementation is very convoluted & messy - try this one below)

def calculate_mass(precipitation_data):
    """
    Calculate the total mass of precipitation.
    :param precipitation_data: Tensor of precipitation data, shape (N, C, H, W)
    :return: Tensor of total mass, shape (N,)
    """
    # Sum over the spatial dimensions (H, W) for each sample
    mass = torch.sum(precipitation_data, dim=(2, 3))
    return mass
 
class MassConservationLoss(nn.Module):
    def __init__(self, weight=0.2):
        """
        Initialize the MassConservationLoss.
        :param weight: Weight of the mass conservation term in the total loss
        """
        super(MassConservationLoss, self).__init__()
        self.weight = weight
 
    def forward(self, input_mass, output_mass):
        """
        Compute the mass conservation loss.
        :param input_mass: Total mass of the input, shape (N,)
        :param output_mass: Total mass of the output, shape (N,)
        :return: Scalar loss value
        """
        # Calculate the mass conservation loss
        mass_loss = torch.mean((output_mass - input_mass) ** 2)
        return self.weight * mass_loss


##########################








######################################
#PLOT THE VARIABLE DISTRIBUTIONS ETC.


'''
def plot_all_distributions(dataset, save_dir="./distribution_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Extract arrays from dataset
    raw_fine = dataset.data
    raw_coarse = dataset.coarse_data
    # raw_residual is the difference between fine (raw) and coarse (raw)
    # if dataset.residuals is the raw residual, use that. Otherwise:
    raw_residual = dataset.data - dataset.inputs  # or dataset.residuals if stored as raw

    norm_fine = dataset.fine
    norm_coarse = dataset.coarse
    norm_residual = dataset.residuals

    variables = dataset.variables
    print(variables)

    def flatten_data(data):
        return data.reshape(data.shape[0]*data.shape[2]*data.shape[3], data.shape[1]).T

    raw_fine_flat = flatten_data(raw_fine)
    raw_coarse_flat = flatten_data(raw_coarse)
    raw_residual_flat = flatten_data(raw_residual)

    norm_fine_flat = flatten_data(norm_fine)
    norm_coarse_flat = flatten_data(norm_coarse)
    norm_residual_flat = flatten_data(norm_residual)

    for i, var in enumerate(variables):
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        # Rows: Fine, Coarse, Residual
        # Cols: Raw, Normalized

        # Fine raw
        axs[0, 0].hist(raw_fine_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[0, 0].set_title(f'{var} - Fine (Raw)')

        # Fine normalized
        axs[0, 1].hist(norm_fine_flat[i], bins=1000, color='green', alpha=0.7)
        axs[0, 1].set_title(f'{var} - Fine (Normalized)')

        # Coarse raw
        axs[1, 0].hist(raw_coarse_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[1, 0].set_title(f'{var} - Coarse (Raw)')

        # Coarse normalized
        axs[1, 1].hist(norm_coarse_flat[i], bins=1000, color='green', alpha=0.7)
        axs[1, 1].set_title(f'{var} - Coarse (Normalized)')

        # Residual raw
        axs[2, 0].hist(raw_residual_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[2, 0].set_title(f'{var} - Residual (Raw)')

        # Residual normalized
        axs[2, 1].hist(norm_residual_flat[i], bins=1000, color='green', alpha=0.7)
        axs[2, 1].set_title(f'{var} - Residual (Normalized)')

        for row in range(3):
            for col in range(2):
                axs[row, col].set_xlabel('Value')
                axs[row, col].set_ylabel('Frequency')

        plt.suptitle(f'Distributions of {var} (Fine, Coarse, Residual)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(os.path.join(save_dir, f'distribution_{var}.png'), dpi=300)
        plt.close(fig)
'''

def plot_all_distributions(dataset, save_dir="./distribution_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Extract arrays from dataset
    raw_fine = dataset.data
    raw_coarse = dataset.coarse_data
    # raw_residual is the difference between fine (raw) and coarse (raw)
    # if dataset.residuals is the raw residual, use that. Otherwise:
    raw_residual = dataset.data - dataset.inputs  # or dataset.residuals if stored as raw

    norm_fine = dataset.fine
    norm_coarse = dataset.coarse
    norm_residual = dataset.residuals

    variables = dataset.variables
    print(variables)

    def flatten_data(data):
        return data.reshape(data.shape[0]*data.shape[2]*data.shape[3], data.shape[1]).T

    raw_fine_flat = flatten_data(raw_fine)
    raw_coarse_flat = flatten_data(raw_coarse)
    raw_residual_flat = flatten_data(raw_residual)

    norm_fine_flat = flatten_data(norm_fine)
    norm_coarse_flat = flatten_data(norm_coarse)
    norm_residual_flat = flatten_data(norm_residual)


    for i, var in enumerate(variables):
        # RAW stats
        fine_min, fine_max, fine_mean = float(raw_fine[:, i].min()), float(raw_fine[:, i].max()), float(raw_fine[:, i].mean())
        coarse_min, coarse_max, coarse_mean = float(raw_coarse[:, i].min()), float(raw_coarse[:, i].max()), float(raw_coarse[:, i].mean())
        residual_min, residual_max, residual_mean = float(raw_residual[:, i].min()), float(raw_residual[:, i].max()), float(raw_residual[:, i].mean())

        # NORMALIZED stats
        nf_min, nf_max, nf_mean = float(norm_fine[:, i].min()), float(norm_fine[:, i].max()), float(norm_fine[:, i].mean())
        nc_min, nc_max, nc_mean = float(norm_coarse[:, i].min()), float(norm_coarse[:, i].max()), float(norm_coarse[:, i].mean())
        nr_min, nr_max, nr_mean = float(norm_residual[:, i].min()), float(norm_residual[:, i].max()), float(norm_residual[:, i].mean())

        print(f"NON FLATTENED Variable: {var}")
        print("Raw Fine:", f"min={fine_min}, max={fine_max}, mean={fine_mean}")
        print("Raw Coarse:", f"min={coarse_min}, max={coarse_max}, mean={coarse_mean}")
        print("Raw Residual:", f"min={residual_min}, max={residual_max}, mean={residual_mean}")
        print("Normalized Fine:", f"min={nf_min}, max={nf_max}, mean={nf_mean}")
        print("Normalized Coarse:", f"min={nc_min}, max={nc_max}, mean={nc_mean}")
        print("Normalized Residual:", f"min={nr_min}, max={nr_max}, mean={nr_mean}\n")



    for i, var in enumerate(variables):
        # Print stats for raw fine
        print(f"Variable: {var}")
        print("Raw Fine:")
        print(f"  min={raw_fine_flat[i].min()}, max={raw_fine_flat[i].max()}, mean={raw_fine_flat[i].mean()}")

        # Print stats for raw coarse
        print("Raw Coarse:")
        print(f"  min={raw_coarse_flat[i].min()}, max={raw_coarse_flat[i].max()}, mean={raw_coarse_flat[i].mean()}")

        # Print stats for raw residual
        print("Raw Residual:")
        print(f"  min={raw_residual_flat[i].min()}, max={raw_residual_flat[i].max()}, mean={raw_residual_flat[i].mean()}")

        # Print stats for normalized fine
        print("Normalized Fine:")
        print(f"  min={norm_fine_flat[i].min()}, max={norm_fine_flat[i].max()}, mean={norm_fine_flat[i].mean()}")

        # Print stats for normalized coarse
        print("Normalized Coarse:")
        print(f"  min={norm_coarse_flat[i].min()}, max={norm_coarse_flat[i].max()}, mean={norm_coarse_flat[i].mean()}")

        # Print stats for normalized residual
        print("Normalized Residual:")
        print(f"  min={norm_residual_flat[i].min()}, max={norm_residual_flat[i].max()}, mean={norm_residual_flat[i].mean()}\n")

        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        # Rows: Fine, Coarse, Residual
        # Cols: Raw, Normalized

        # Fine raw
        axs[0, 0].hist(raw_fine_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[0, 0].set_title(f'{var} - Fine (Raw)')

        # Fine normalized
        axs[0, 1].hist(norm_fine_flat[i], bins=1000, color='green', alpha=0.7)
        axs[0, 1].set_title(f'{var} - Fine (Normalized)')

        # Coarse raw
        axs[1, 0].hist(raw_coarse_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[1, 0].set_title(f'{var} - Coarse (Raw)')

        # Coarse normalized
        axs[1, 1].hist(norm_coarse_flat[i], bins=1000, color='green', alpha=0.7)
        axs[1, 1].set_title(f'{var} - Coarse (Normalized)')

        # Residual raw
        axs[2, 0].hist(raw_residual_flat[i], bins=1000, color='blue', alpha=0.7)
        axs[2, 0].set_title(f'{var} - Residual (Raw)')

        # Residual normalized
        axs[2, 1].hist(norm_residual_flat[i], bins=1000, color='green', alpha=0.7)
        axs[2, 1].set_title(f'{var} - Residual (Normalized)')

        for row in range(3):
            for col in range(2):
                axs[row, col].set_xlabel('Value')
                axs[row, col].set_ylabel('Frequency')

        plt.suptitle(f'Distributions of {var} (Fine, Coarse, Residual)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(os.path.join(save_dir, f'distribution_{var}.png'), dpi=300)
        plt.close(fig)
        
        
#########################################



###########################################################
#Define a separate 'sample_model' function which just returns the predicted tensor, rather than the images as well

@torch.no_grad()
def sample_model_val(model, batch, dataset, device, surface_mask_batch=None,num_steps=100, sigma_min=0.002, sigma_max=80, rho=7,
                 S_churn=40, S_min=0, S_max=float('inf'), S_noise=1): #### Try editing some of the samp;ling parameters here as well!! (S_churn = 40, simga max = 80)

    images_input = batch["inputs"].to(device)
    coarse = batch["coarse"].to(device)
    condition_params = torch.stack((batch["doy"].to(device), batch["hour"].to(device)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 4, images_input.shape[2], images_input.shape[3]),
                             dtype=torch.float64, device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))
                 
                 
        full_condition = torch.cat([images_input, surface_mask_batch], dim=1) ###$$$

        denoised = model(x_hat, t_hat, full_condition, condition_params).to(torch.float64) #model(x_hat, t_hat, images_input, condition_params).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = model(x_next, t_next, full_condition, condition_params).to(torch.float64) #model(x_next, t_next, images_input, condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataset.residual_to_fine_image(x_next.detach().cpu(), coarse.cpu())
    # 'predicted' is now a tensor you can call .numpy() on if desired.
    return predicted #the 'residual_to_fine_image' function above is returning an image in the correct scale (Not normalised) so no need to denorm again - just use output


###########################################################



##################
#PLOT THE RESULTS OF YOUR VALIDATED DATA
import os
import matplotlib.pyplot as plt

def plot_coarse_fine_pred(coarse_denorm, fine_denorm, pred_denorm, variables, 
                          batch_idx, output_dir="./results_validation"):
    """
    Plot and save side-by-side images for coarse, fine, and predicted for each variable.
    
    :param coarse_denorm: Tensor [B, C, H, W] (denormalized coarse)
    :param fine_denorm:   Tensor [B, C, H, W] (denormalized fine/ground truth)
    :param pred_denorm:   Tensor [B, C, H, W] (denormalized model prediction)
    :param variables:      List of variable names corresponding to channels in the above tensors
    :param batch_idx:      Index of the current validation batch
    :param output_dir:     Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Assume batch size == 1 for simplicity; if bigger, you may want an outer loop
    b = 0  # We'll just plot the first (or only) sample in the batch
    
    # Move data to CPU numpy if still on GPU
    coarse_np = coarse_denorm[b].cpu().numpy()
    fine_np   = fine_denorm[b].cpu().numpy()
    pred_np   = pred_denorm[b].cpu().numpy()
    
    # Each variable is a channel
    for var_idx, var_name in enumerate(variables):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        # Coarse
        im0 = axes[0].imshow(coarse_np[var_idx, :, :], cmap='viridis',interpolation='none') #,interpolation='nearest' (none?)
        axes[0].set_title(f"Coarse: {var_name}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Fine
        im1 = axes[1].imshow(fine_np[var_idx, :, :], cmap='viridis',interpolation='none')
        axes[1].set_title(f"Fine (GT): {var_name}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Predicted
        im2 = axes[2].imshow(pred_np[var_idx, :, :], cmap='viridis',interpolation='none')
        axes[2].set_title(f"Predicted: {var_name}")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        outpath = os.path.join(output_dir, f"val_batch{batch_idx}_{var_name}.png")
        plt.savefig(outpath, dpi=150)
        plt.close(fig)





##################


def load_surface_inputs(batch_size,device):

    #############
    #SURFACE MASKS
    soil_type = np.load(f'/g/data/hn98/Declan/soil_type.npy')
    topography = np.load(f'/g/data/hn98/Declan/topography.npy')
    land_sea = np.load(f'/g/data/hn98/Declan/land_mask.npy')  # Load the third array
    
    
    ######
    # Create latitude and longitude arrays corresponding to the data grid
    latitudes = np.linspace(90, -90, 721)  # 0.25-degree steps from 90N to -90S
    longitudes = np.linspace(0, 360, 1440, endpoint=False)  # 0.25-degree steps from 0E to 359.75E
    
    # Define the latitude and longitude ranges for Australia
    lat_min, lat_max = -7.75, -43.5 #-11.75 -> -7.75
    lon_min, lon_max = 109.0, 176.75
    #ALREADY DEFINED THESE VARIABLES IN THE VERY BEGINNING
    
    # Find the indices corresponding to the Australian region
    #lat_indices = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    lon_indices = np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]
    
    if latitudes[0] > latitudes[-1]:  # Decreasing latitudes
        lat_indices = np.where((latitudes <= lat_min) & (latitudes >= lat_max))[0]
    else:
        lat_indices = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    
    
    # Subset the arrays to the Australian region
    soil_type_subset = soil_type[np.ix_(lat_indices, lon_indices)]
    topography_subset = topography[np.ix_(lat_indices, lon_indices)]
    land_sea_subset = land_sea[np.ix_(lat_indices, lon_indices)]
    
    print("lat_indices size:", lat_indices.size)
    print("lon_indices size:", lon_indices.size)
    print("soil_type_subset shape:", soil_type_subset.shape)
    print("topography_subset shape:", topography_subset.shape)
    print("land_sea_subset shape:", land_sea_subset.shape)
    
    
    soil_type = soil_type_subset
    topography = topography_subset
    land_sea = land_sea_subset
    ######
    
    # Stack the arrays along a new axis (e.g., axis=0 or axis=-1)
    stacked_array = np.stack((soil_type, topography, land_sea), axis=-1)
    
    # Convert the stacked array to a PyTorch tensor
    tensor = torch.tensor(stacked_array)
    
    # Print the shape of the resulting tensor to verify
    print(tensor.shape)
    
    surface_mask = tensor
    surface_mask = surface_mask.permute(2, 0, 1)
    print(surface_mask.shape)
    ##############
    
    surface_mask_batch = surface_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device) #padding the batch dimension to match other data
    print(surface_mask_batch.shape)
    
    return surface_mask_batch






############################################
#CHANGING THE NOISE SCHEDULING TO BE HIERARCHICAL AS WELL

import math
import imageio  # For saving video frames if desired

class HierarchicalScheduler:
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
                
                
                
            '''
                
            #REPLACED THE PORTION BELOW WITH THIS - SHOULD ALLOW FOR MULTIPLE SHRINKS PER NOISE STEP NOW 
            if shrink_now and (not done_h or not done_w):
                # Perform multiple splits this time if shape_splits_per_shrink > 1
                for _ in range(self.shape_splits_per_shrink):
                    if not done_h and not done_w:
                        # Decide which dimension to shrink, same ratio logic
                        ratio_now = cur_w / float(cur_h)
                        if ratio_now > self.x_over_y_ratio:
                            cur_w -= 1
                        else:
                            cur_h -= 1
                    elif not done_h:
                        # w is done => must shrink h if possible
                        cur_h -= 1
                    elif not done_w:
                        # h is done => must shrink w if possible
                        cur_w -= 1
            
                    # Clamp to 1, mark done if they hit 1
                    if cur_h < 1: 
                        cur_h = 1
                        done_h = True
                    if cur_w < 1:
                        cur_w = 1
                        done_w = True
                        
                    shape_list.append((cur_h, cur_w))
            
                    # If both finished, no need to keep looping
                    if done_h and done_w:
                        break
                        
            '''
            
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


class HierarchicalEDMLoss:
    def __init__(self,
                 scheduler: HierarchicalScheduler,
                 P_mean=-1.2,
                 P_std=1.2,
                 sigma_data=1.0):
        """
        P_mean, P_std, sigma_data: same as your old EDMLoss
        scheduler: a HierarchicalScheduler instance
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.scheduler = scheduler  # the shape-schedule engine

    def __call__(self, net, images, conditional_img=None, labels=None,
                 augment_pipe=None):
        """
        images: shape (B,C,H,W) e.g. (B,4,144,272).
        1) We pick a random t in [0, scheduler.total_steps-1].
        2) We forward-noise from x0 -> x_t according to the hierarchical chain.
        3) Then upsample x_t back to (144,272) for the UNet.
        4) We sample log_sigma from N(P_mean,P_std^2) => sigma
        5) As usual in EDM, we do  D_yn = net( (y + n), sigma ), etc.
        6) Return \(\mathcal{L}\).
        """
        device = images.device
        batch_size = images.shape[0]

        # Possibly augment
        y, augment_labels = augment_pipe(images) if augment_pipe else (images, None)

        # 1) random t
        t_int = torch.randint(
            low=0,
            high=self.scheduler.total_steps,
            size=(batch_size,),
            device=device
        )

        # 2) build or use the forward chain
        #    Because building the entire chain for each sample is expensive,
        #    well do a small trick: we build the entire chain ONCE per batch,
        #    then each sample picks x_{t_i} from it.  If your batch size is big
        #    and memory is an issue, you can do it sample-by-sample in a loop. #HONESTLY DOESN'T EVEN MATTER BECAUSE BATCH SIZE IS ONE - MEMORY ISSUE
        chain = self.scheduler.forward_noising_chain(y, add_noise=True)
        # chain is a list of length T = scheduler.total_steps
        # chain[k].shape = (B, C, smallerH, smallerW)

        # 3) gather x_{t_i} for each sample i
        #    e.g. x_t[i] = chain[t_int[i]][i, ...]
        #    but chain[t] is (B,C, h,w). We want to gather the correct t for each i.
        #    Easiest approach: stack them shape (T,B,C,...) => pick with advanced indexing
        
        '''
        chain_tensor = torch.stack(chain, dim=0)  # => shape (T, B, C, hi, wi)
        
        # now gather
        indices = t_int.view(1, -1, 1, 1, 1).expand(-1, -1, chain_tensor.shape[2], 1, 1)
        # shape: (1,B,C,1,1)
        # but we must gather along dim=0, so do gather with a range style.
        # simpler: do a custom function that picks chain_tensor[t_i,i]
        x_t = []
        for i in range(batch_size):
            t_i = t_int[i].item()
            # chain_tensor[t_i, i, ...]
            x_t.append(chain_tensor[t_i, i, ...].unsqueeze(0))
        x_t = torch.cat(x_t, dim=0)  # shape (B,C,h_t_i,w_t_i)
        '''
        
        #With batch size one, change to as follows as won't stack differing sizes correctly
        t_i = t_int[0].item()
        x_t = chain[t_i]

        # 3a) Upsample x_t to (144,272)
        x_t_up = []
        for i in range(batch_size):
            xsmall = x_t[i:i+1]  # shape (1,C,h,w)
            xbig   = self.scheduler.upsample_to_full(xsmall)
            x_t_up.append(xbig)
        x_t_up = torch.cat(x_t_up, dim=0)  # shape (B,C,144,272)

        # 4) Sample log_sigma for EDM
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2

        # 5) add noise n
        n = torch.randn_like(x_t_up) * sigma

        # If user wants to pass shape (i_t, j_t) to net as labels => we do that here
        # We have t_int => shape[t_int], let's compute them:
        shape_t = []
        for i in range(batch_size):
            st = self.scheduler.shape_list[t_int[i]]
            shape_t.append(st)
        # shape_t is a list of (h,w). We can store them in a 2D tensor
        shape_cond = torch.tensor(shape_t, dtype=torch.float16, device=device) ##torch.float32
        # combine with users 'labels' if you want => e.g. final_labels = torch.cat([labels, shape_cond], dim=1)

        # Because your old net expects labels=..., just do:
        if labels is not None:
            final_labels = torch.cat([labels, shape_cond], dim=1)  # shape (B, 2 + original_label_dim)
        else:
            final_labels = shape_cond  # shape (B,2)

        # 6) Do the usual EDM forward
        # y + n => x_t_up + n
        # network tries to predict D_yn => (approx) x_0
        D_yn = net(x_t_up + n, sigma, conditional_img, #conditional_img=conditional_img
                   final_labels, augment_labels=augment_labels) #REMOVED THE 'LABEL=' PART labels=final_labels

        # same weighting as old code
        per_var_loss = weight * ((D_yn - x_t_up)**2)
        # reduce
        mean_loss_per_var = per_var_loss.view(per_var_loss.size(0),
                                              per_var_loss.size(1),
                                              -1).mean(dim=2)
        loss = mean_loss_per_var.mean()

        return loss, D_yn


#NEW MODEL SAMPLING PROCESS 






import torch
import torch.nn.functional as F
import numpy as np
import imageio

@torch.no_grad()
def sample_model_hierarchical(
    model,
    batch,
    dataset,
    device,
    scheduler,              # <--- Your existing HierarchicalScheduler instance
    surface_mask_batch=None,
    num_inference_steps=10,
    sigma_min=0.002,
    sigma_max=30,
    rho=4,
    S_churn=1,
    S_min=0,
    S_max=float('inf'),
    S_noise=1,
    # VIDEO-RELATED:
    create_video=False,
    video_path="./hierarchical_denoise.gif"
):
    """ 
    Hierarchical + EDM sampler with video creation:
      - Uses scheduler.shape_list (which goes from (H,W) down to (1,1)),
        then we reverse it so we effectively grow from (1,1) up to (H,W).
      - Each iteration does an EulerHeun step with S_churn, etc.
      - We *interpolate* the partial latent to (144,272) before the model call,
        because your model expects a fixed input resolution.
      - If create_video=True, we collect frames and write them to an MP4 at the end.
    """

    # ---------------------------------------------------------------------
    # 1) Basic setup (same as your original sample_model_val)
    # ---------------------------------------------------------------------
    images_input = batch["inputs"].to(device)
    coarse = batch["coarse"].to(device)
    fine   = batch["fine"].to(device)  ###
    condition_params = torch.stack(
        (batch["doy"].to(device), batch["hour"].to(device)), dim=1
    )

    # We assume your final desired resolution is just images_input.shape
    B, _, H, W = images_input.shape

    # Clamp sigma_min / sigma_max
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    # ---------------------------------------------------------------------
    # 2) Reverse the scheduler�s shape_list to go small->large
    # ---------------------------------------------------------------------
    full_shape_list = list(scheduler.shape_list)  # e.g. (H,W)->...->(1,1)
    full_shape_list.reverse()                     # now (1,1)->...->(H,W)
    
    print(full_shape_list.reverse())
    

    # For simplicity, assume scheduler.total_steps == num_inference_steps
    shape_list_reverse = full_shape_list[:num_inference_steps+1]
    
    print(shape_list_reverse)

    # ---------------------------------------------------------------------
    # 3) EDM time steps (just like sample_model_val)
    # ---------------------------------------------------------------------
    step_indices = torch.arange(num_inference_steps, dtype=torch.float64, device=device)
    t_steps = (
        (sigma_max ** (1 / rho)
         + step_indices/(num_inference_steps - 1)
           * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    )
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # final t_N=0

    # ---------------------------------------------------------------------
    # 4) Initialize random latent at shape (1,1) scaled by t_steps[0]
    # ---------------------------------------------------------------------
    (tiny_h, tiny_w) = shape_list_reverse[0]
    init_noise = torch.randn(
        (B, 4, tiny_h, tiny_w), dtype=torch.float64, device=device
    )
    x_next = init_noise * t_steps[0]

    # ---------------------------------------------------------------------
    # 5) VIDEO: set up a frames list if we want to record the sampling
    # ---------------------------------------------------------------------
    frames = []  # We'll store one grayscale frame (channel 0) per iteration
    frames_residual = []
    
    #Get colour for coarse/fine ###
    coarse_color = coarse[0, :3, :, :].detach().cpu().numpy().transpose(1,2,0)
    fine_color   = fine[0, :3, :, :].detach().cpu().numpy().transpose(1,2,0)

    ###
    def normalize_to_01(img_3ch):
        mn = img_3ch.min()
        mx = img_3ch.max()
        if mx > mn:
            return (img_3ch - mn) / (mx - mn)
        else:
            return np.zeros_like(img_3ch)

    coarse_color = normalize_to_01(coarse_color)
    fine_color   = normalize_to_01(fine_color)
    

    # ---------------------------------------------------------------------
    # 6) Main diffusion loop
    # ---------------------------------------------------------------------
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # (a) Interpolate x_next to the "desired_shape" in shape_list_reverse
        desired_shape = shape_list_reverse[i]
        x_cur = F.interpolate(
            x_next, size=desired_shape, mode='bilinear', align_corners=False
        ).to(torch.float16)

        # (b) Compute S_churn gamma
        if (S_min <= t_cur <= S_max):
            gamma = min(S_churn / num_inference_steps, np.sqrt(2) - 1)
        else:
            gamma = 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)

        # (c) If gamma>0, add noise
        if gamma > 0:
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)
        else:
            x_hat = x_cur

        # (d) Build the full_condition (also upsample surface_mask_batch if it exists)
        
        # (d) Build the full_condition (upsample surface_mask_batch to model's input size)
        if surface_mask_batch is not None:
            # Upscale mask to (H, W) instead of desired_shape
            mask_up = F.interpolate(
                surface_mask_batch, size=(H, W),  # Changed desired_shape to (H, W)
                mode='bilinear', align_corners=False
            ).to(torch.float64)
            full_condition = torch.cat([images_input, mask_up], dim=1)
        else:
            full_condition = images_input
    
    
        #if surface_mask_batch is not None:
        #    mask_up = F.interpolate(
        #        surface_mask_batch, size=desired_shape,
        #        mode='bilinear', align_corners=False
        #    ).to(torch.float64)
        #    full_condition = torch.cat([images_input, mask_up], dim=1)
        #else:
        #    full_condition = images_input

        # Also build shape-based labels if you want them
        # (just as an example, you can incorporate shape or not)
        shape_cond = torch.tensor([list(desired_shape)]*B, 
                                  dtype=torch.float16, device=device)
        final_labels = torch.cat([condition_params, shape_cond], dim=1)  # shape (B,4)

        # (e) The model expects (144,272) as input size, so let's upsample x_hat
        #     to that fixed shape. We'll call it 'x_hat_up'.
        x_hat_up = F.interpolate(
            x_hat, size=(H, W), mode='bilinear', align_corners=False
        )

        # likewise, upsample full_condition (if you want it shape=(B,8,H,W), etc.)
        # but if your images_input was already (B,4,H,W), we can just use that.
        # If your surface_mask was included, also upsampled it to (H,W).
        # Let's do that quickly:
        cond_up = F.interpolate(
            full_condition, size=(H, W), mode='bilinear', align_corners=False
        ).to(torch.float16)

        # The model might need half precision:
        #model = model.half()
        #x_hat_up = x_hat_up.half()
        #cond_up = cond_up.half()
        #final_labels = final_labels.half()
        #t_hat = t_hat.half()

        print("\nDEBUG SHAPES BEFORE MODEL CALL:")
        print(f"x_hat shape: {x_hat.shape} | dtype: {x_hat.dtype} | device: {x_hat.device}")
        print(f"t_hat shape: {t_hat.shape} | dtype: {t_hat.dtype} | device: {t_hat.device}")
        print(f"full_condition shape: {full_condition.shape} | dtype: {full_condition.dtype}")
        print(f"final_labels shape: {final_labels.shape} | dtype: {final_labels.dtype}")
        print(f"Current VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")


        # (f) EulerHeun steps:
        denoised = model(x_hat_up, t_hat, cond_up, final_labels).to(torch.float16)
        d_cur = (x_hat_up - denoised) / t_hat
        # Euler step
        x_euler_up = x_hat_up + (t_next - t_hat)*d_cur

        if i < num_inference_steps - 1:
            # second pass
            denoised_euler = model(x_euler_up, t_next, cond_up, final_labels).to(torch.float16) #t_next.half()
            d_prime = (x_euler_up - denoised_euler) / t_next
            x_next_up = x_hat_up + (t_next - t_hat) * 0.5 * (d_cur + d_prime)
        else:
            x_next_up = x_euler_up

        # Now we "downsample" x_next_up back to the smaller shape for iteration i+1,
        # unless we're on the last iteration. Let's keep consistent with the loop:
        if i < num_inference_steps - 1:
            # next_shape is shape_list_reverse[i+1]
            next_shape_hw = shape_list_reverse[i+1]
            print(f"NEXT SHAPE: {next_shape_hw}")
            x_next = F.interpolate(
                x_next_up, size=next_shape_hw, mode='bilinear', align_corners=False
            ).to(torch.float64)
        else:
            # last iteration => no next shape
            x_next = x_next_up.to(torch.float64)

        # (g) If creating a video, store a frame
        # Let's store channel 0 of x_next_up (or denoised) as grayscale
        
        #Change from just first channel to all channels
        '''
        if create_video:
            # 'x_next_up' is shape (B,4,H,W). We'll just take the first sample, channel 0
            frame_array = x_next_up[0, 0].detach().cpu().numpy()  # (H,W)
            # normalize 0..1
            fmin, fmax = frame_array.min(), frame_array.max()
            if fmax > fmin:  # avoid div-by-zero
                frame_norm = (frame_array - fmin)/(fmax - fmin)
            else:
                frame_norm = frame_array * 0
            frame_uint8 = (frame_norm * 255).astype(np.uint8)
            frames.append(frame_uint8)
        '''    

        ### The one here VVVV works well, just in greyscale
        '''
        # In the frame creation section of sample_model_hierarchical:
        if create_video:
            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f'Step {i+1}/{num_inference_steps}', fontsize=14)
            
            # Plot all 4 variables
            for var_idx in range(4):
                ax = axs[var_idx//2, var_idx%2]
                var_data = x_next_up[0, var_idx].detach().cpu().numpy()
                
                # Normalize using percentiles for better visualization
                vmin = np.percentile(var_data, 1)
                vmax = np.percentile(var_data, 99)
                
                im = ax.imshow(var_data, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f'Variable {var_idx+1}')
                fig.colorbar(im, ax=ax)
            
            plt.tight_layout()
            
            # Convert plot to numpy array
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer._renderer)[..., :3]  # RGB only
            frames.append(frame)
            plt.close(fig)
            
        '''
        
        ################################################################
        # NEW PLOTTING CODE: Show all 4 variables in separate subplots.
        # Each row => [coarse, fine, evolving prediction].
        ################################################################
        if create_video:
            fig, axs = plt.subplots(4, 3, figsize=(12, 16))
            fig.suptitle(f"Step {i+1} / {num_inference_steps}", fontsize=16)

            #Get residual as well
            fig_res, axs_res = plt.subplots(4, 3, figsize=(12, 16)) #plt.subplots(nrows=4, ncols=1, figsize=(5, 16))
            fig_res.suptitle(f"Residual Only :: Step {i+1}/{num_inference_steps}", fontsize=15)


            # For each of the 4 variables
            for var_idx in range(4):
                # -- 1) Extract Coarse --
                coarse_var = coarse[0, var_idx].detach().cpu().numpy()
                vmin_c = np.percentile(coarse_var, 1)
                vmax_c = np.percentile(coarse_var, 99)

                im0 = axs[var_idx, 0].imshow(
                    coarse_var,
                    cmap="jet", #cmap='viridis',
                    vmin=vmin_c,
                    vmax=vmax_c,
                    origin="upper",
                )
                axs[var_idx, 0].set_title(f"Coarse var {var_idx+1}")
                axs[var_idx, 0].axis("off")
                fig.colorbar(im0, ax=axs[var_idx, 0], fraction=0.046, pad=0.04)

                # -- 2) Extract Fine (GT) --
                fine_var = fine[0, var_idx].detach().cpu().numpy()
                vmin_f = np.percentile(fine_var, 1)
                vmax_f = np.percentile(fine_var, 99)

                im1 = axs[var_idx, 1].imshow(
                    fine_var,
                    cmap="jet", #cmap='viridis',
                    vmin=vmin_f,
                    vmax=vmax_f,
                    origin="upper",
                )
                axs[var_idx, 1].set_title(f"Fine (GT) var {var_idx+1}")
                axs[var_idx, 1].axis("off")
                fig.colorbar(im1, ax=axs[var_idx, 1], fraction=0.046, pad=0.04)

                # -- 3) Extract Evolving Prediction for var_idx --
                #x_next_up shape = (B, 4, H, W); pick [0, var_idx, ...]
                #pred_var = x_next_up[0, var_idx].detach().cpu().numpy()
                #add in below to make it not just the residual - but the actual image:
                #print(f"pred_var shape: {pred_var.shape}") 
                #print(f"%%%x_next_detach is of shape: {x_next.detach().cpu().shape}")
                #print(f"x_next_detach is of shape: {coarse.cpu().shape}")
                pred_var_unnorm = dataset.residual_to_fine_image(x_next_up[0, var_idx].detach().cpu(), coarse.cpu()) #x_next.detach().cpu()
                pred_var = pred_var_unnorm[0,var_idx].numpy() #Needs to be numpy for matplotlib I believe (won't take torch tensors)
                vmin_p = np.percentile(pred_var, 1)
                vmax_p = np.percentile(pred_var, 99)

                im2 = axs[var_idx, 2].imshow(
                    pred_var,
                    cmap="jet", #cmap='viridis',
                    vmin=vmin_p, #vmin_p,
                    vmax=vmax_p, #vmax_p,
                    origin="upper",
                )
                axs[var_idx, 2].set_title(f"Prediction var {var_idx+1}")
                axs[var_idx, 2].axis("off")
                fig.colorbar(im2, ax=axs[var_idx, 2], fraction=0.046, pad=0.04)

                ######
                #Above is for full image, now just look at the residual here

                #First, save both the same coarse & fine from earlier to the new fig_res (doesn't change)
                fig_res.colorbar(im0, ax=axs_res[var_idx, 0], fraction=0.046, pad=0.04)
                fig_res.colorbar(im1, ax=axs_res[var_idx, 1], fraction=0.046, pad=0.04)

                #THIS IS THE DIRECT RESIDUAL HERE
                pred_var = x_next_up[0, var_idx].detach().cpu().numpy()

                vmin_p = np.percentile(pred_var, 1)
                vmax_p = np.percentile(pred_var, 99)

                im2_res = axs[var_idx, 2].imshow(
                    pred_var,
                    cmap="jet", #cmap='viridis',
                    vmin=vmin_p, #vmin_p,
                    vmax=vmax_p, #vmax_p,
                    origin="upper",
                )
                axs_res[var_idx, 2].set_title(f"Prediction var {var_idx+1}")
                axs_res[var_idx, 2].axis("off")
                fig_res.colorbar(im2_res, ax=axs[var_idx, 2], fraction=0.046, pad=0.04) #SHOULD HAVE EVERYTHING SAVED TO FIG_RES NOW

            plt.tight_layout()

            # Convert the figure to a NumPy array
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer._renderer)[..., :3]  # keep RGB
            frames.append(frame)
            plt.close(fig)

            # Convert the residual figure to a NumPy array
            fig_res.canvas.draw()
            frame_res = np.array(fig_res.canvas.renderer._renderer)[..., :3]  # keep RGB
            frames_residual.append(frame_res)
            plt.close(fig_res)
        ################################################################


    # ---------------------------------------------------------------------
    # 7) Done with the loop. x_next is final in shape (H,W).
    #    Convert residual -> final image:
    # ---------------------------------------------------------------------
    predicted = dataset.residual_to_fine_image(x_next.detach().cpu(), coarse.cpu())

    # ---------------------------------------------------------------------
    # 8) If we recorded frames, write out the video
    # ---------------------------------------------------------------------
    
    #This is saving an .mp4 but you are going to use .gif instead (due to your packages installed rn)
    #if create_video and len(frames) > 1:
    #    imageio.mimsave(video_path, frames, fps=5)
    #    print(f"[sample_model_hierarchical] Wrote video to {video_path}")
        
    if create_video and len(frames) > 1:
        # Convert frames to uint8 if needed
        frames_uint8 = [frame.astype(np.uint8) for frame in frames]
    
        # Save as GIF with 5 FPS
        imageio.mimsave(video_path, frames_uint8, fps=6, format='GIF')

    if create_video and len(frames_residual) > 1:
        # Convert frames to uint8 if needed
        frames_residual_uint8 = [f.astype(np.uint8) for f in frames_residual]
        imageio.mimsave("residual_only.gif", frames_residual_uint8, fps=6, format='GIF')
        print("Saved a second GIF showing only the *predicted residual* at each step!")

    return predicted







#UPDATED BASE SHAPE TO MAKE THE MODELS TAKE A LAT OF 736 (RATHER THAN 721) WHICH IS THE SMALLEST NUMBER ABOVE 721 THAT IS DIVISIBLE BY 2^4 = 16 FOR THE CONV LAYERS IN THE UNET - ALSO MODIFIED DATA LOADING FUNCTION TO PAD THE DATA UP TO THIS SIZE IN 'DATASETUS.PY'
def main(downscale_factor=8):
    import torch
    batch_size = 1  # Adjusted batch size
    learning_rate = 1e-5
    num_epochs = 150
    accum = 8
    
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    
    network = EDMPrecond((144, 272), 11, 4, label_dim=4) #UPPED LABEL_DIM =2 TO LABEL_DIM=4 FOR I/J OF SHAPE AS WELL #UPPED CONDITIONAL CHANNELS FROM 8 -> 11 # Updated image resolution #736, 1440 #
    network.to(device)
    
    #NEW - LOAD CHECKPOINTS IF YOU DEFINE ONE
    #checkpoint = "./Checkpoints/Diffusion/Equal_Aus_FPN140-10yr.pt"   #Equal_Aus_FPN70-10yr.pt" #None
    checkpoint = None #Does this solve it? 
    if checkpoint is not None:
        network.load_state_dict(torch.load(checkpoint))
    #torch.save(network.state_dict(), f"./Checkpoints/Diffusion/Equal_Aus_FPN{step}.pt") #f"./Model/{step}.pt") Checkpoints/Diffusion/

    
    
    # Define the datasets
    datadir = '/g/data/rt52/era5/single-levels/reanalysis/'

    # Calculate input and output shapes
    #base_shape = (736, 1440)  # Updated base shape
    #downscale_factor = 16  # Adjust as necessary
    #in_shape = (base_shape[0] // downscale_factor, base_shape[1] // downscale_factor)  # Coarse shape
    #out_shape = base_shape  # High-resolution shape remains the same


    base_shape = (144, 272)   # HR shape
    in_shape   = (24, 46)     # Coarse shape
    out_shape = base_shape
    
    
    starting_year = 2019 #1990
    ending_year = 2019 #2010
    train_months = [1] #[1,2,3,4,5,6,7,8,9,10,11,12] #[1,2,3]

    # Use the updated UpscaleDataset
    dataset_train = UpscaleDataset(datadir, year_start=starting_year, year_end=ending_year, #year_start=1996, year_end=2000,
                                   in_shape=in_shape, out_shape=out_shape,
                                   variables=['10u', '10v', '2t', 'tp'], #CHANGE THE ORDERING OF THESE TO JUST SEE IF THIS AFFECTS WHAT IS INCORRECTLY NORMALISED: # '10u', '10v', '2t', 'tp'
                                   months=train_months, #1,2,3,4,5,6,7,8,9
                                   aggregate_daily=True)


    #Load in your constant surface variables (soil, land-sea-mask, etc.)
    surface_mask_batch = load_surface_inputs(batch_size,device)
    

    #Plot the norm etc. of thesee data distributions
    plot_all_distributions(dataset_train, save_dir="./results/distributions")


    # Get all normalization stats
    all_stats = dataset_train.get_all_normalization_stats()

    # Convert numpy data to serializable types
    for key in all_stats:
        all_stats[key] = {k: float(v) for k, v in all_stats[key].items()}

    import json
    # Save to a file
    with open('normalisation_stats_ALL_AUS_30yr.json', 'w') as f:
        json.dump(all_stats, f)

    

    # Get and save normalization stats (for reverse norm process)
    #means, stds = dataset_train.get_normalization_stats()
    # Convert means and stds to serializable types
    #means = {k: float(v) for k, v in means.items()}
    #stds = {k: float(v) for k, v in stds.items()}
    # Save to a file
    #import json
    #with open('normalization_stats.json', 'w') as f:
    #    json.dump({'means': means, 'stds': stds}, f)
    
    
    #Load whatever mean/std-devs were printed to use these
    import json
    
    # Load normalization stats
    with open('normalisation_stats_ALL_AUS_30yr.json', 'r') as f:
        precomputed_stats = json.load(f)
    

    dataset_test = UpscaleDataset(datadir, year_start=2000, year_end=2000,
                                  in_shape=in_shape, out_shape=out_shape,
                                  variables=['10u', '10v', '2t', 'tp'], #'10u', '10v', '2t', 'tp'
                                  months=[10],
                                  aggregate_daily=True,
                                  precomputed_stats=precomputed_stats)

    # Create DataLoaders for training and testing
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    scaler = torch.cuda.amp.GradScaler()

    # Define the optimizer
    optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    # Define the tensorboard writer
    writer = SummaryWriter("./runs")

    # Define the loss function
    
    #########################
    #CHANGING TO THE HIERARCHICAL LOSS FUNCTION
    
    #Changing to hierarchical one #loss_fn = EDMLoss()
    
    
    #IMPORTANT NOTE HERE ON THE DENOISING/RESIZING STEPS. ALL OF THE FOLLOWING NEED TO BE THE SAME:
    # - THE SCHEDULER
    # - THE NUMBER OF STEPS USED WHEN SAMPLING AT INFERENCE (AT YOUR MID-EVAL POINTS IN THE TRAINING)
    # - THE NUMBER OF STEPS USED WHEN SAMPLING AT INFERENCE (AT YOUR FINAL EVAL)    
    
    SAMPLING_STEPS_INFERENCE = 50
    NSE_STEPS_PER_SPLIT = 2
    
    scheduler = HierarchicalScheduler( #CREATES THE SCHEDULE OVER WHICH WE ADD NOISE VS REDUCE IMAGE SIZE (HOW OFTEN TO DO HIERARCHICAL VS DENOISING?)
        full_size=(144,272),
        total_steps=SAMPLING_STEPS_INFERENCE,
        x_over_y_ratio=(272/144), #
        noise_steps_per_split=NSE_STEPS_PER_SPLIT,
        shape_splits_per_shrink=1, #DO NOT AMEND THIS EVER - CODE DOES NOTHING HERE AND YOU COMMENTED OUT BELOW
        use_exponential=False,
        device=device
    )
    
    
    print("\n=== Full Hierarchical Scheduler Shape Schedule ===")
    for i, (h, w) in enumerate(scheduler.shape_list):
        print(f"Step {i:3d} | shape=({h}, {w})")
    print("==================================================\n")
    
    loss_fn = HierarchicalEDMLoss(
        scheduler=scheduler,  # pass the above
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=1.0
    )
    
    ##################
    #DEFINE THE VALIDATION DATALOADER:
    

    # Define your validation period (adjust as needed)
    val_years = [2020,2021,2022,2023,2024]  # example, change as needed
    val_months = [1,2,3,4,5,6,7,8,9,10,11] # example, change as needed 

    print("\n=== Running Validation Evaluation ===")

    # Load validation dataset
    dataset_val = UpscaleDataset(
        datadir,
        year_start=min(val_years),
        year_end=max(val_years),
        in_shape=in_shape,
        out_shape=out_shape,
        variables=['10u', '10v', '2t', 'tp'],  # same variables as train/test
        months=val_months,
        aggregate_daily=True,
        precomputed_stats=precomputed_stats  # use the same normalization stats
    )

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Evaluation metrics storage
    variables = ['10u', '10v', '2t', 'tp']  # ensure same variable order
    variable_indices = {var: i for i, var in enumerate(variables)}
    total_rmse_per_var = {var: 0.0 for var in variables}
    total_psnr_per_var = {var: 0.0 for var in variables}
    total_ssim_per_var = {var: 0.0 for var in variables}
    num_samples_val = 0
    
    network.eval()  # ensure model is in eval mode
    ##################



    




    # Train the model
    losses = []
    for step in range(0, num_epochs):
        epoch_loss = training_step(network, loss_fn, optimiser, dataloader_train, scaler, step, accum, writer,surface_mask_batch = surface_mask_batch) ###$$$
        losses.append(epoch_loss)

    
        if (step + 0) % 5 == 0:

            ###$
            #29/11/24
            print("Coarse_IN")
            print(dataloader_test.dataset.coarse_mean)
            print(dataloader_test.dataset.coarse_std_dev)
    
            print("Fine_IN")
            print(dataloader_test.dataset.data_mean)
            print(dataloader_test.dataset.data_std_dev)
    
            print("residual_IN")
            print(dataloader_test.dataset.residual_mean)
            print(dataloader_test.dataset.residual_std_dev)


            #REMOVING THIS STEP ENTIRELY AS YOUR HIERARCHICAL SAMPLE MODEL IS NOT DOING THE SAME AS THE ORIGINAL WHERE IT PRINTS THE IMAGES
            '''
            (fig, ax), (base_error, pred_error) = sample_model_hierarchical(model=network, dataset = dataloader_test, scheduler=scheduler,batch_size=1,device=device,num_inference_steps=500, create_video=False, video_path="./hier_denoise.mp4", ,surface_mask_batch = surface_mask_batch) #sample_model(network, dataloader_test,surface_mask_batch = surface_mask_batch) #sample_model(network, dataloader_test)
            fig.savefig(f"./results/{step}.png", dpi=300)
            plt.close(fig)

            writer.add_scalar("Error/base", base_error, step)
            writer.add_scalar("Error/pred", pred_error, step)
            '''
            ####
            
        ###################################

        if step % 10 == 0 and step > 0:
            print("\n=== Running Validation Evaluation ===")
            network.eval()
            num_samples_val = 0
            total_rmse_per_var = {var: 0.0 for var in variables}
            total_psnr_per_var = {var: 0.0 for var in variables}
            total_ssim_per_var = {var: 0.0 for var in variables}
    
            for batch in dataloader_val:
                #FOR EVERY 10 EPOCHS, YOU EVALUATE USING THE LLINE DIRECTLY BELOW (THEN AGAIN FURTHER DOWN, YOU EVALUATE ONCE MORE)
                predicted = sample_model_hierarchical(model=network, dataset = dataset_val, batch=batch, scheduler=scheduler,device=device,num_inference_steps=SAMPLING_STEPS_INFERENCE, create_video=False, video_path="./hier_denoise.mp4", surface_mask_batch = surface_mask_batch) #NUM_STEPS WAS 500 - TRY SMALLER #sample_model_val(network, batch, dataset_val, device=device,surface_mask_batch = surface_mask_batch) ###$$$ #REPLACED WITH THE NEW HIERARCHICAL SAMPLING
                images_output = batch["fine"]
                fine_denorm = dataset_val.inverse_normalize_data(images_output)
                predicted_data = predicted.numpy()
    
                predicted_tensor = torch.tensor(predicted_data)
                images_output_tensor = torch.tensor(fine_denorm)
    
                for var in variables:
                    var_index = variable_indices[var]
                    pred_var = predicted_tensor[:, var_index, :, :]
                    true_var = images_output_tensor[:, var_index, :, :]
    
                    rmse = calculate_rmse(pred_var, true_var)
                    psnr = calculate_psnr(pred_var, true_var)
                    ssim_value = calculate_ssim(pred_var.unsqueeze(1), true_var.unsqueeze(1))
    
                    total_rmse_per_var[var] += rmse
                    total_psnr_per_var[var] += psnr
                    total_ssim_per_var[var] += ssim_value
    
                num_samples_val += 1
    
            avg_rmse_val = {var: total_rmse_per_var[var] / num_samples_val for var in variables}
            avg_psnr_val = {var: total_psnr_per_var[var] / num_samples_val for var in variables}
            avg_ssim_val = {var: total_ssim_per_var[var] / num_samples_val for var in variables}
    
            print("Validation Metrics per Variable:")
            for var in variables:
                print(f"Variable {var}:")
                print(f"  RMSE: {avg_rmse_val[var]}")
                print(f"  PSNR: {avg_psnr_val[var]}")
                print(f"  SSIM: {avg_ssim_val[var]}")
    
            print("=== Validation Evaluation Complete ===\n")
    
        #################################
            


        # Save the model
        #if losses[-1] == min(losses): #(JUST SAVE THE MODEL REGARDLESS AND CHECK YOUR LOSS PLOT
        torch.save(network.state_dict(), f"./Checkpoints/Diffusion/Equal_Aus_FPN{step}.pt") #f"./Model/{step}.pt") Checkpoints/Diffusion/
        print("Print out model step that has been saved")
        print(step)
        
        ###% 27/11/24
        #print(network.model)
        #from torchsummary import summary
        #summary(network.model, input_size=(network.in_channels, *network.img_resolution))
        
        # Save loss plot every epoch
        plt.figure()
        plt.plot(range(len(losses)), losses, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'./results/loss_curve_epoch_{step}.png', dpi=300)
        plt.close()
        
        
    ##########################################
    #VALIDATION EVALUATION
    
    '''
    # Define your validation period (adjust as needed)
    val_years = [2001]  # example, change as needed
    val_months = [2, 3] # example, change as needed

    print("\n=== Running Validation Evaluation ===")

    # Load validation dataset
    dataset_val = UpscaleDataset(
        datadir,
        year_start=min(val_years),
        year_end=max(val_years),
        in_shape=in_shape,
        out_shape=out_shape,
        variables=['10u', '10v', '2t', 'tp'],  # same variables as train/test
        months=val_months,
        aggregate_daily=True,
        precomputed_stats=precomputed_stats  # use the same normalization stats
    )

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Evaluation metrics storage
    variables = ['10u', '10v', '2t', 'tp']  # ensure same variable order
    variable_indices = {var: i for i, var in enumerate(variables)}
    total_rmse_per_var = {var: 0.0 for var in variables}
    total_psnr_per_var = {var: 0.0 for var in variables}
    total_ssim_per_var = {var: 0.0 for var in variables}
    num_samples_val = 0

    network.eval()  # ensure model is in eval mode
    '''
    
    i=0 #Create loop variable for batch number
    
    for batch in dataloader_val:
        #%%%predicted = sample_model_hierarchical(model=network, dataset = dataset_val, batch=batch, scheduler=scheduler,device=device,num_inference_steps=500, create_video=False, video_path="./hier_denoise.gif", surface_mask_batch = surface_mask_batch)#sample_model_val(network, batch, dataset_val, device=device, surface_mask_batch = surface_mask_batch) ###$$$ # Using sample_model as defined above
        ### ^^^ create_video=True - changed to false to see if this improves runtime - you have a large validation set rn so taking a long time over this I think
        #%%%i = i+1
        
        ###&&&&
        if i == len(dataloader_val)-1:
            predicted = sample_model_hierarchical(model=network, dataset = dataset_val, batch=batch, scheduler=scheduler,device=device,num_inference_steps=SAMPLING_STEPS_INFERENCE, create_video=True, video_path="./hier_denoise.gif", surface_mask_batch = surface_mask_batch)#sample_model_val(network, batch, dataset_val, device=device, surface_mask_batch = surface_mask_batch) ###$$$ # Using sample_model as defined above
            #Changed 500 steps - > 50
        else:
            predicted = sample_model_hierarchical(model=network, dataset = dataset_val, batch=batch, scheduler=scheduler,device=device,num_inference_steps=SAMPLING_STEPS_INFERENCE, create_video=False, video_path="./hier_denoise.gif", surface_mask_batch = surface_mask_batch)#sample_model_val(network, batch, dataset_val, device=device, surface_mask_batch = surface_mask_batch) ###$$$ # Using sample_model as defined above
            #Changed 500 steps - > 50
        i = i+1 
        ###&&&&
        
        # Denormalize data for plotting
        coarse_denorm = dataset_val.inverse_normalize_coarse(batch["coarse"])
        fine_denorm   = dataset_val.inverse_normalize_data(batch["fine"])
        #pred_denorm   = dataset_val.inverse_normalize_data(predicted) #22/01 - remove denormalise - predicted image from sample_model_val() should be correct already
        pred_denorm = predicted


        #print(f"Batch number is {batch}")
        # ---- PLOTTING STEP: coarse vs fine vs predicted per variable ----
        plot_coarse_fine_pred(
            coarse_denorm=coarse_denorm,
            fine_denorm=fine_denorm,
            pred_denorm=pred_denorm,
            variables=variables,
            batch_idx=f"AUS_Batch_{i}",  # This will show up in your saved filenames
            output_dir="./results_validation"
        )
        
        
        
        
        # Retrieve ground truth
        images_output = batch["fine"]
        fine_denorm = dataset_val.inverse_normalize_data(images_output)
        predicted_data = predicted.numpy()
        
        predicted_tensor = torch.tensor(predicted_data)
        images_output_tensor = torch.tensor(fine_denorm)

        # Compute metrics per variable
        for var in variables:
            var_index = variable_indices[var]
            pred_var = predicted_tensor[:, var_index, :, :]
            true_var = images_output_tensor[:, var_index, :, :]

            rmse = calculate_rmse(pred_var, true_var)
            psnr = calculate_psnr(pred_var, true_var)
            ssim_value = calculate_ssim(pred_var.unsqueeze(1), true_var.unsqueeze(1))

            total_rmse_per_var[var] += rmse
            total_psnr_per_var[var] += psnr
            total_ssim_per_var[var] += ssim_value

        num_samples_val += 1

    # Compute average metrics per variable for validation
    avg_rmse_val = {var: total_rmse_per_var[var] / num_samples_val for var in variables}
    avg_psnr_val = {var: total_psnr_per_var[var] / num_samples_val for var in variables}
    avg_ssim_val = {var: total_ssim_per_var[var] / num_samples_val for var in variables}

    print("Validation Metrics per Variable:")
    for var in variables:
        print(f"Variable {var}:")
        print(f"  RMSE: {avg_rmse_val[var]}")
        print(f"  PSNR: {avg_psnr_val[var]}")
        print(f"  SSIM: {avg_ssim_val[var]}")
    
    print("=== Validation Evaluation Complete ===")
    
    
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total number of parameters in the model: {total_params}")
    
    print(f"Training start year: {starting_year} | Training end year: {ending_year} | Training months: {train_months}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Validation years: {val_years} | Validation Months: {val_months}")
    
    print(f"Sampling steps: {SAMPLING_STEPS_INFERENCE}")
    print(f"Noise steps per split: {NSE_STEPS_PER_SPLIT}")

    print("==== FULL MODEL STRUCTURE ====")
    print(network)
    print("==============================\n")


#
if __name__ == "__main__":
    main(downscale_factor=8)
    




