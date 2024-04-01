from PyMIDIMusic import *
from PyMIDIMusic import MIDIToVector

import PIL
import numpy as np
from datasets import Dataset

def loadImage():
    music, test, tokens = MIDIToVector.GetTokens()
    tokens.pop()

    grid = np.array(tokens).reshape((252, 16))
    # MIDIToVector.DisplayMusicRhythm(grid)

    image = PIL.Image.fromarray(np.uint8(grid), "L")

    # image = PIL.Image.open("Assets/Datasets/Flowers102/grayscale/image_00001.jpg")


    # plt.imshow(image)
    # image = image.convert("L")

    image = image.crop((0, 0, image.width, image.width))

    print("Loading Image From MIDI:")
    print("\twidth: ", image.width)
    print("\theight: ", image.height)

    outDir = os.path.join(config.output_dir, "inputMusic")
    os.makedirs(outDir, exist_ok=True)
    image.save(f"{outDir}/{0:04d}.png")

    print("\tsaved: ", outDir)

    return image

def loadDataset(preprocess):
    image = loadImage()

    data_dict = {"image": ([image]*5000)}
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="numpy")

    def transform(examples):
        images = [preprocess(image) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    return dataset, train_dataloader