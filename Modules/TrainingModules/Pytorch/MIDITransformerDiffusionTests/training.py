from tld.train import main
from tld.configs import ModelConfig, DataConfig, TrainConfig

import pathlib
folder = str(pathlib.Path(__file__).parent.resolve()) + '/'

# text_emb_path = '/content/orig_text_encodings256.npy'#@param {type:"string"}
latent_path = folder + 'pokemon_path/image_latents.hdf5' #@param {type:"string"}
text_emb_path = folder + 'pokemon_path/text_encodings.hdf5'#@param {type:"string"}


data_config = DataConfig(
    latent_path=latent_path, text_emb_path=text_emb_path, val_path="val_emb.npy"
)

model_cfg = ModelConfig(
    data_config=data_config,
    train_config=TrainConfig(n_epoch=100, save_model=False, compile=False, use_wandb=False),
)

main(model_cfg)

#OR in a notebook ot run the training process on 2 GPUs:
#notebook_launcher(main, model_cfg, num_processes=2)