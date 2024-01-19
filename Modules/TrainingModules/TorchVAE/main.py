import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


from PyMIDIMusic import *



# music = MIDIMusic() 

# noteOnOff = NoteOnOff()
# noteOnOff.SetKey(60)
# noteOnOff.SetDuration(5)
# noteOnOff.SetVelocity(100)
# noteOnOff.SetChannel(0)
# noteOnOff.SetDeltaTime(5)

# music.AddEvent(noteOnOff)

# music.Play("/home/progz/projects/MusicGenerationFramework/Assets/Touhou.sf2")
# c = input()
# exit()





def NormalizeKey(key):
    return (key * 2 - 127) / 127

def UnNormalizeKey(key):
    return (key * 127 + 127) / 2



class Test(IMIDIEventReceiver):
    channels = []
    times = []
    notes = []
    colors = []

    def OnNoteOnOff(self, event): 
        e = NoteOnOff(event)
        self.times.append(e.GetDeltaTime())
        self.notes.append(e.GetKey())
        self.channels.append(e.GetChannel())

music = MIDIMusic() 

music.LoadFromFile("/home/progz/projects/MusicGenerationFramework/Assets/Fur_Elise.1.mid")


easyLib.MIDIMusic_FilterChannel(music.nativeObject, 9, True)
easyLib.MIDIMusic_ConvertToMonoTrack(music.nativeObject)

easyLib.MIDIMusic_ConvertToNoteOnOff(music.nativeObject)

# easyLib.MIDIMusic_ConvertAbsolute(music.nativeObject)
easyLib.MIDIMusic_FilterInstruments(music.nativeObject, 0, 7, False)

# music.Play("/home/progz/projects/MusicGenerationFramework/Assets/Touhou.sf2")

statMusic = music.Clone()
easyLib.MIDIMusic_ConvertAbsolute(statMusic.nativeObject)

test = Test()
Dispatch(statMusic, test)

# Image size and channels
width, height = 14, 14
channels = 1

# Create a black image (all zeros)
black_image = np.zeros((height, width, channels), dtype=np.uint8)

# Fill the black image manually
for y in range(height):
    for x in range(width):
        for c in range(channels):
            if (x+y*width < len(test.notes)):
                black_image[y, x, c] = float(test.notes[x+y*width])  # Set each channel to 0 (black)
            else:
                black_image[y, x, c] = 0.0
                break

# plt.imshow(black_image)
# plt.show()

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # sample = {
        #     'data': torch.tensor(self.data[idx]),
        #     # 'label': torch.tensor(self.labels[idx])
        # }
        return (NormalizeKey(torch.tensor(self.data[idx])), idx)

black_image_tensor = torch.tensor(black_image, dtype=torch.float32)






class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(width*height, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, width*height)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, width, height))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=1):
    opt = torch.optim.Adam(autoencoder.parameters())
    print("Starting training")
    for epoch in range(epochs):
        print("Starting epoch ", epoch)
        xi = 0
        for x, y in data:
            print("Current epoch : ", xi, "/", len(data), end="\r")
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            xi+=1
        print()
        print("Finishing epoch ", epoch)
    return autoencoder

latent_dims = 10
autoencoder = VariationalAutoencoder(latent_dims).to(device) # GPU

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', 
               transform=torchvision.transforms.ToTensor(), 
               download=True),
        batch_size=128,
        shuffle=True)

data = torch.utils.data.DataLoader(
        CustomDataset([black_image_tensor] * 60000),
        batch_size=128,
        shuffle=True)

autoencoder = train(autoencoder, data)

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

# plot_latent(autoencoder, data)
# plt.show()

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

# plot_reconstructed(autoencoder)
# plot_reconstructed(autoencoder, r0=(-3, 3), r1=(-3, 3))
# plt.show()

n = 1
img = np.zeros((width*n, height*n))

# z = autoencoder.decoder(data.dataset.__getitem__(0).to(device))
# x_hat = autoencoder.decoder(z)
# x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()

# x_hat = autoencoder(data.dataset.__getitem__(0)[0].to(device))

x, y = next(iter(data))
imgCpy = x[0].reshape(width, height).to('cpu').detach().numpy() * 2 + 1
img[(n-1)*width:(n-1+1)*width, 0:height] = imgCpy
plt.imshow(img)
plt.show()

x = x.to(device) # GPU
# opt.zero_grad()
x_hat = autoencoder(x)
# loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
# loss.backward()
# opt.step()
imgCpy = x_hat[0].reshape(width, height).to('cpu').detach().numpy() * 2 + 1
img[(n-1)*width:(n-1+1)*width, 0:height] = imgCpy
plt.imshow(img)

x_hat = x_hat.flatten()

music = MIDIMusic() 

noteOnOff = NoteOnOff()
noteOnOff.SetKey(0)
noteOnOff.SetDuration(200)
noteOnOff.SetVelocity(100)
noteOnOff.SetChannel(0)
noteOnOff.SetDeltaTime(100)
music.AddEvent(noteOnOff)

for i in x_hat:
    noteOnOff = NoteOnOff()
    noteOnOff.SetKey(int(UnNormalizeKey(i)))
    noteOnOff.SetDuration(1)
    noteOnOff.SetVelocity(100)
    noteOnOff.SetChannel(0)
    noteOnOff.SetDeltaTime(1)

    music.AddEvent(noteOnOff)

music.Play("/home/progz/projects/MusicGenerationFramework/Assets/Touhou.sf2")

plt.show()

while (True):
    c = input()


