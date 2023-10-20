# Runtime Modules

## Introduction

They are the modules that are going to be used at runtime, when generating music.\
They are in C++ for performance, but export C functions so that they can be used by other languages.

## Types of Runtime Modules

Runtime modules can fall into different categories: 

- **Root Modules**\
Those are the modules that are independent from other modules of this framework.

- **Prebuilt Binaries Modules**\
Some modules are prebuilt libraries.
It can be because the source code is private, or is too complex to compile.\
All Prebuilt Binaries Modules are Root Modules.

- **Test Modules**\
They are used to test if a module is working or not.\
No Test Module is a Root Module, because it is testing another module.\
All Test Modules generate a runnable file (.exe).

- **Input Preprocessing Modules (IPM)**\
Before a Machine Learning model is used, input data has to be preprocesssed. 

- **Output Preprocessing Modules (OPM)**\
After a Machine Learning model is used, output data has to be preprocesssed.

## Creating my own Runtime Module

Minimal specifications:
- Create a directory inside this RuntimeModules folder.
- Add a CMakeLists.txt file that puts its binaries and non-module dependencies into the module binaries directory.

Good practices:
- If you are making a Root Module, consider making its own repository, and adding it as a submodule (see [EasyMidiFileParserCpp](EasyMidiFileParserCpp)).
- Add a README.md explaining what the module is about, and in what categories it falls under.
- If the module is compiled from source, split the code into Private/, Public/ and Tests/ directories.
- If it is a Prebuilt Binaries Module, download the binaries from the CMakeLists.txt.

