"""
Created on Apr 6, 2021

This script performs inference via the sliding-window approach.

"""

import os
import sys
PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extract directory of PyCharm project
sys.path.append(PROJECT_HOME)  # this line is needed to recognize the dir as a python package
import nibabel as nib
import numpy as np
import tensorflow as tf
import shutil
import time
from tqdm import tqdm
from inference.utils_inference import load_nifti_and_resample, create_tf_dataset, create_output_folder, \
    sanity_check_inputs, str2bool, load_config_file
from training.network_training import create_compiled_unet
from show_results.utils_show_results import round_half_up
from dataset_creation.utils_dataset_creation import  print_running_time


def inference_one_subject(input_nii_path: str,
                         output_dir: str,
                         unet_checkpoint_path: str,
                         unet_patch_side: int = 64,
                         unet_batch_size: int = 1,
                         unet_threshold: float = 0.3,
                         new_spacing: tuple = (0.39, 0.39, 0.55),
                         unet=None,
                         overlapping: float = 0.5,
                         test_time_augmentation: bool = True):
    """Performs sliding-window inference on a single .nii file
    Args:
        input_nii_path (str): Path to input .nii file
        output_dir (str): Path to save results 
        unet_checkpoint_path (str): Path to model weights
        unet_patch_side (int): Size of patch side (default 64)
        unet_batch_size (int): Batch size for inference (default 1)
        unet_threshold (float): Threshold for predictions (default 0.3)
        new_spacing (tuple): Desired voxel spacing (default (0.39, 0.39, 0.55))
        unet: Pre-loaded model (optional)
        overlapping (float): Patch overlap ratio (default 0.5)
        test_time_augmentation (bool): Whether to use test time augmentation (default True)
    Returns:
        str: Path to the result.nii.gz file
    """
    # Start timer
    start = time.time()

    # Get base filename without extension
    base_filename = os.path.splitext(os.path.splitext(os.path.basename(input_nii_path))[0])[0]
    
    # Create output directory with filename
    output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary directory
    tmp_path = os.path.join(output_dir, "tmp_processing")
    os.makedirs(tmp_path, exist_ok=True)
    
    # Load input volume
    nii_obj = nib.load(input_nii_path)
    nii_data = np.asanyarray(nii_obj.dataobj)
    
    # Resample volume to desired spacing
    out_name = "resampled.nii.gz"
    nii_resampled_sitk, nii_obj_resampled, nii_data_resampled, aff_resampled = load_nifti_and_resample(
        input_nii_path, tmp_path, out_name, new_spacing
    )

    # Setup sliding window parameters
    shift_scale = unet_patch_side // 2
    rows, cols, slices = nii_data_resampled.shape
    step = int(round_half_up((1 - overlapping) * unet_patch_side))

    # Collect patches
    patches = []
    patch_centers = []
    total_steps = ((rows - 2*shift_scale)//step) * ((cols - 2*shift_scale)//step) * ((slices - 2*shift_scale)//step)
    print(f"\nCollecting patches from volume of shape {nii_data_resampled.shape}")
    pbar = tqdm(total=total_steps, desc="Processing patches")
    
    for i in range(shift_scale, rows, step):
        for j in range(shift_scale, cols, step):
            for k in range(shift_scale, slices, step):
                if (i - shift_scale >= 0 and i + shift_scale < rows and 
                    j - shift_scale >= 0 and j + shift_scale < cols and 
                    k - shift_scale >= 0 and k + shift_scale < slices):
                    
                    patch = nii_data_resampled[
                        i - shift_scale:i + shift_scale,
                        j - shift_scale:j + shift_scale,
                        k - shift_scale:k + shift_scale
                    ]
                    patch = tf.image.per_image_standardization(patch)
                    patches.append(patch)
                    patch_centers.append([i, j, k])
                    pbar.update(1)
    
    pbar.close()
    print(f"Collected {len(patches)} patches")

    # Create dataset for inference
    dataset = create_tf_dataset(patches, unet_batch_size)

    # Create output folder and run inference
    result_path = os.path.join(f"{output_dir}/{time.time()}", "result.nii.gz")
    _ = create_output_folder(
        dataset,
        output_dir,
        unet_threshold,
        unet,
        nii_data_resampled,
        False,  # reduce_fp_with_volume
        0,      # min_aneurysm_volume 
        nii_obj_resampled,
        patch_centers,
        shift_scale,
        nii_resampled_sitk,
        aff_resampled,
        tmp_path,
        False,  # reduce_fp
        5,      # max_fp
        False,  # remove_dark_fp
        0,      # dark_fp_threshold
        nii_data,
        "inference",
        test_time_augmentation,
        unet_batch_size
    )

    # Cleanup
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    end = time.time()
    print_running_time(start, end, "Inference complete")

    return result_path


def main(input_nii_path):
    """Main function to run inference on a single subject"""
    config_dict = load_config_file()

    # Extract basic inference parameters
    unet_patch_side = config_dict['unet_patch_side']
    unet_batch_size = config_dict['unet_batch_size']
    unet_threshold = config_dict['unet_threshold']
    overlapping = config_dict['overlapping']
    new_spacing = tuple(config_dict['new_spacing'])
    conv_filters = tuple(config_dict['conv_filters'])
    test_time_augmentation = str2bool(config_dict['test_time_augmentation'])
    
    # Path configurations
    output_dir = config_dict['inference_outputs_path']
    unet_checkpoint_path = config_dict['training_outputs_path']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Verify model weights path exists
    weights_path = os.path.join(unet_checkpoint_path, "saved_models", "my_checkpoint")
    if not os.path.exists(weights_path + ".index"):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Please ensure the path to model weights is correct in config_inference.json"
        )

    # Create U-Net model
    inputs = tf.keras.Input(shape=(unet_patch_side, unet_patch_side, unet_patch_side, 1), name='TOF_patch')
    unet = create_compiled_unet(inputs, config_dict['lr'], config_dict['lambda_loss'], conv_filters)
    
    # Load model weights
    try:
        unet.load_weights(weights_path).expect_partial()
    except Exception as e:
        raise Exception(f"Failed to load model weights: {str(e)}")

    # Run inference with correct parameters
    result_file = inference_one_subject(
        input_nii_path=input_nii_path,
        output_dir=output_dir,
        unet_checkpoint_path=unet_checkpoint_path,
        unet_patch_side=unet_patch_side,
        unet_batch_size=unet_batch_size,
        unet_threshold=unet_threshold,
        new_spacing=new_spacing,
        unet=unet,
        overlapping=overlapping,
        test_time_augmentation=test_time_augmentation
    )

    print(f"Results saved to: {result_file}")
    return result_file

def cleanup_before_inference():
    """Clean up cache and temporary files before running inference"""
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Clear temporary directories
    temp_dirs = [
        './Output',
        './tmp_processing'
    ]
    
    for dir in temp_dirs:
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                print(f"Cleared {dir}")
            except Exception as e:
                print(f"Could not clear {dir}: {e}")

    # Clear GPU memory if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")

if __name__ == "__main__":
    cleanup_before_inference()
    main("./Test Data/sub-022_ses-20101011_angio.nii.gz")



    