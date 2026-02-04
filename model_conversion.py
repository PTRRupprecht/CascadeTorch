# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:51:15 2026

@author: peter
"""

import numpy as np
import torch
from tensorflow.keras.models import load_model
from cascade2p import utils
import os

def convert_keras_to_pytorch(keras_model_path, pytorch_model_path, cfg):
    """
    Convert a Keras .h5 model to PyTorch .pth format
    
    Parameters
    ----------
    keras_model_path : str
        Path to the Keras .h5 file
    pytorch_model_path : str
        Path where PyTorch .pth file will be saved
    cfg : dict
        Configuration dictionary with model parameters
    """
    
    # Load Keras model
    keras_model = load_model(keras_model_path)
    
    # Create PyTorch model with same architecture
    pytorch_model = utils.define_model(
        filter_sizes=cfg["filter_sizes"],
        filter_numbers=cfg["filter_numbers"],
        dense_expansion=cfg["dense_expansion"],
        windowsize=cfg["windowsize"],
        loss_function=cfg["loss_function"],
        optimizer=cfg["optimizer"],
    )
    
    # Get Keras weights
    keras_weights = keras_model.get_weights()
    
    # Transfer weights layer by layer
    # Keras layer order: Conv1D, Conv1D, MaxPooling1D, Conv1D, MaxPooling1D, Dense, Flatten, Dense
    
    # Conv1 (weights, bias)
    pytorch_model.conv1.weight.data = torch.from_numpy(keras_weights[0].transpose(2, 1, 0))
    pytorch_model.conv1.bias.data = torch.from_numpy(keras_weights[1])
    
    # Conv2 (weights, bias)
    pytorch_model.conv2.weight.data = torch.from_numpy(keras_weights[2].transpose(2, 1, 0))
    pytorch_model.conv2.bias.data = torch.from_numpy(keras_weights[3])
    
    # Conv3 (weights, bias)
    pytorch_model.conv3.weight.data = torch.from_numpy(keras_weights[4].transpose(2, 1, 0))
    pytorch_model.conv3.bias.data = torch.from_numpy(keras_weights[5])
    
    # Dense1 (weights, bias)
    pytorch_model.dense1.weight.data = torch.from_numpy(keras_weights[6].transpose(1, 0))
    pytorch_model.dense1.bias.data = torch.from_numpy(keras_weights[7])
    
    # Dense2 (weights, bias)
    pytorch_model.dense2.weight.data = torch.from_numpy(keras_weights[8].transpose(1, 0))
    pytorch_model.dense2.bias.data = torch.from_numpy(keras_weights[9])
    
    # Save PyTorch model
    torch.save(pytorch_model.state_dict(), pytorch_model_path)
    print(f"Converted: {keras_model_path} -> {pytorch_model_path}")
    
    return pytorch_model


def convert_all_models_in_folder(model_name, model_folder="Pretrained_models"):
    """
    Convert all Keras models in a model folder to PyTorch
    
    Parameters
    ----------
    model_name : str
        Name of the model folder
    model_folder : str
        Base folder containing pretrained models
    """
    import glob
    from cascade2p import config
    
    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, "config.yaml")
    
    # Load config
    cfg = config.read_config(cfg_file)
    
    # Find all .h5 files
    keras_models = glob.glob(os.path.join(model_path, "*.h5"))
    
    print(f"Found {len(keras_models)} Keras models to convert")
    
    for keras_path in keras_models:
        # Create corresponding .pth filename
        pytorch_path = keras_path.replace('.h5', '.pth')
        
        # Convert
        convert_keras_to_pytorch(keras_path, pytorch_path, cfg)
    
    print(f"\nConversion complete! {len(keras_models)} models converted.")


model_name = r"C:\Users\peter\Desktop\CascadeTorch\CascadeTorch\Pretrained_models\GC8_EXC_30Hz_smoothing25ms_high_noise"  # Your model name
convert_all_models_in_folder(model_name)


def verify_conversion(keras_model_path, pytorch_model_path, cfg):
    """
    Verify that Keras and PyTorch models produce same outputs
    """
    import torch
    from tensorflow.keras.models import load_model
    
    # Load both models
    keras_model = load_model(keras_model_path)
    pytorch_model = utils.define_model(
        filter_sizes=cfg["filter_sizes"],
        filter_numbers=cfg["filter_numbers"],
        dense_expansion=cfg["dense_expansion"],
        windowsize=cfg["windowsize"],
        loss_function=cfg["loss_function"],
        optimizer=cfg["optimizer"],
    )
    pytorch_model.load_state_dict(torch.load(pytorch_model_path))
    pytorch_model.eval()
    
    # Create random test input
    test_input = np.random.randn(10, cfg["windowsize"], 1).astype(np.float32)
    
    # Keras prediction
    keras_output = keras_model.predict(test_input, verbose=0)
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_input = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input).numpy()
    
    # Compare outputs
    max_diff = np.max(np.abs(keras_output - pytorch_output))
    mean_diff = np.mean(np.abs(keras_output - pytorch_output))
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    if max_diff < 1e-5:
        print("✓ Conversion verified - outputs match!")
    else:
        print("✗ Warning: outputs differ significantly")
    
    return max_diff < 1e-5



from cascade2p import config
import os

# Define your model name and folder
model_name = "GC8_EXC_30Hz_smoothing25ms_high_noise"
model_folder = "Pretrained_models"

# Load the config file
model_path = os.path.join(model_folder, model_name)
cfg_file = os.path.join(model_path, "config.yaml")
cfg = config.read_config(cfg_file)

# Now verify a specific model
verify_conversion(
    keras_model_path=os.path.join(model_path, "Model_NoiseLevel_2_Ensemble_2.h5"),
    pytorch_model_path=os.path.join(model_path, "Model_NoiseLevel_2_Ensemble_2.pth"),
    cfg=cfg
)
