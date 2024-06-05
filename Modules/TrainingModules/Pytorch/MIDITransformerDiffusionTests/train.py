
import os
from tld.train import main
from tld.configs import DataConfig, ModelConfig, TrainConfig, DenoiserConfig
from accelerate import notebook_launcher
import pathlib

#os.environ["WANDB_API_KEY"]='your_key_here'
#!wandb login

# latent_path = '/content/image_latents256.npy' #@param {type:"string"}
folder = str(pathlib.Path(__file__).parent.resolve()) + '/'

# text_emb_path = '/content/orig_text_encodings256.npy'#@param {type:"string"}
latent_path = folder + 'pokemon_path/image_latents.hdf5' #@param {type:"string"}
text_emb_path = folder + 'pokemon_path/text_encodings.hdf5'#@param {type:"string"}
run_id='' #@param {type:"string"}
n_epoch=40 #@param {type:"integer"}





import h5py
import torch

# Load the HDF5 file
imFile = h5py.File(latent_path, 'r')

# Assuming you have a dataset named 'dataset' in the HDF5 file
data = imFile['image_latents'][:]

# Convert the data to a PyTorch tensor
imtensor_data = torch.tensor(data, dtype=torch.float32)

# Close the HDF5 file
imFile.close()



# Load the HDF5 file
txtFile = h5py.File(text_emb_path, 'r')

# Assuming you have a dataset named 'dataset' in the HDF5 file
data = txtFile['text_encodings'][:]

# Convert the data to a PyTorch tensor
txttensor_data = torch.tensor(data, dtype=torch.float32)

# Close the HDF5 file
txtFile.close()






data_config = DataConfig(latent_path=imtensor_data,
                        text_emb_path=txttensor_data,
                        val_path=folder + 'pokemon_path/val_encs.npy')

denoiser_config = DenoiserConfig(image_size=32)

model_cfg = ModelConfig(
    data_config=data_config,
    denoiser_config=denoiser_config,
    train_config=TrainConfig(
        n_epoch=1,
        save_model=False,
        compile=True,
        use_wandb=False
        ),
)


notebook_launcher(main, (model_cfg,), num_processes=1)

# from google.colab import runtime
# runtime.unassign()