# Skull-stripping and Brain-tumor Segmentation System

The system strips the skull from Magnetic Resonance (MR) brain images and segments the brain tumor. Additional tools after this step allow for user correction of the segmentation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Installation

1) First, install packages from requirements.txt

```
pip install -r requirements.txt
```

2) Then install ANTs with instructions provided at https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS

3) Unpack pickles.tar.gz and put the content into the segmentation folder.

### Directory structure

- main = main.py
   This module allows you to run the GUI.
- pickles.tar.gz
   This .tar.gz archive contains the classifier. It should be unpacked and put into segmentation. However, because of GitHub memory limitations (100 MB), this file is removed from the directory.
- gui_utils
   This has every helper module that the GUI needs.
- skull-stripping
   This folder contains the files needed for skull stripping.
- segmentation
   This folder contains the files needed for the segmentation.
- requirements.txt
   This file contains the requirements needed for running the program.



### Running

In order to run the program, the user needs to enter the home directory of the project and the following command on the terminal should be run:

```
python main.py
```

In this setting, the GUI will ask you to import modality paths. The patient list on the right of the screen will then fill up. If the modality images are not skull stripped and segmented, there is a possibility to do so using the buttons "Preprocess" and "Segment" respectively. If the user wants to visualize the modules, he can click the button "Visualize".


## Authors

* Stevica Bozhinoski, M.Sc. stevica.bozhinoski@tum.de
* Enes Senel
* Yusuf Savran
* Jana Lipkova, M.Sc.
* Esther Alberts, M.Sc.
* Prof. Dr. Björn Menze

