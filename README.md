# Music Generation Framework

## What is the goal of this project?

- Music generation
- Good generation quality
- Customizable generation  
- Real time generation support
- Adaptive music support

## What can this framework be used for?

### Game background music
Most music in games is background music (including boss musics, theme songs, etc).
That music could change dynamically depending on events.

### Video background music
The music can be tailored to match specific events occurring in the video.

### Youtube background music
It can simply be used as a simple background music for people to listen to while studying for example.
Additionally, it can be streamed continuously, making it suitable for 24/7 playback.

## Getting Started

The project provides:

[Dynamic Libraries](Modules/RuntimeModules/README.md):\
Ready-to-use tools for your own programs.\
You should be able to use them in most languages.

[Pre-Trained Machine Learning Models](Assets/Models/README.md):\
Models that are already trained and good to go.

[ML Training Tools](Modules/TrainingModules/README.md):\
Pre-made python scripts that can help you train or fine-tune a ML model.

[Data Analysis Tools](Modules/DataAnalysisModules/README.md):\
Pre-made python scripts that can help you analyze music files.

[Asset references](Assets/README.md):\
[Datasets](Assets/Datasets/README.md) and [soundfonts](Assets/Soundfonts/README.md) can be downloaded automatically using CMake.\
If you do not have CMake installed, open the CMakeLists.txt of the corresponding folder and download assets through the link by yourself.

## Using the framework

### Prebuilt binaries

You can directly use the prebuilt binaries into your programs.\
Prebuilt binaries include:
- A collection of runtime music generation libraries
- A collection of ML models

### Building from source

#### Recommended Prerequisites

- Git
- CMake
- A C++ Compiler
- A Python interpreter

#### Download the repository

- Clone through the command line.\
Do not forget to clone the subrepositories.\
`git clone --recursive https://github.com/ThomasProg/MusicGenerationFramework.git`

- Clone through a GUI Client (GithubDesktop, GitKraken, SourceTree...).

- Download ZIP from github and extract.\
Currently not recommended because of submodules.\
[TODO] : Download submodules using CMake?

#### Installing the framework

From the root folder:

1. Generate Build files and download dependencies.\
    `cmake -B Build -DDOWNLOAD_ASSETS=ON -DBUILD_MODULES=ON -DDOWNLOAD_SUBMODULES=ON -DDOWNLOAD_MODULE_DEPENDENCIES=ON`
    
2. Build modules (in Release or Debug).\
`cmake --build Build --config Debug --target ModularMusicGenerationCore LoadAllMIDI`\
`cmake --build Build --config Release --target ModularMusicGenerationCore LoadAllMIDI`

Options:
- **BUILD_MODULES** : if enabled, build files will be generated for modules.
- **DOWNLOAD_ASSETS** : if enabled, download assets automatically.
- **DOWNLOAD_SUBMODULES** : if enabled, submodules that haven't been cloned yet will be cloned.
- **DOWNLOAD_MODULE_DEPENDENCIES** : if enabled, prebuilt binaries used as dependencies for modules will be downloaded automatically.

The first time the cmake command is ran, it will take a long time.

Downloads also take a long time, so consider disabling them if they are not needed or if you have already downloaded them.


#### Adding the framework to your CMake Project

If you are building your own program using CMake, it is recommended to just use add_subdirectory():

```cmake
# ...
# Your code
# ...

add_subdirectory(MusicGenerationFramework)

# ...
# Your code
# ...
```
