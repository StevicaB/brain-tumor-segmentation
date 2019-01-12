# Installation:

1) First install packages from requirements.txt

```
pip install -r requirements.txt
```

2) Then install ANTs with instructions provided at https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS

3) Unpack pickles.tar.gz and put the content into the segmentation folder.

# Code

- main = main.py
   This module allows you to run the GUI.
- pickles.tar.gz
   This .tar.gz archive contains the classifier. It should be unpacked and put into segmentation.
- gui_utils
   This has every helper module that the gui needs.
- skull-stripping
   This folder contains the files needed for skull stripping.
- segmentation
   This folder contains the files needed for the segmentation.
- requirements.txt
   This file contains the requirements needed for running the program.

# Run

To run, execute main.py:

python main.py

In this setting, the gui will ask you to import modality paths. The patient list on the right of the screen will then fill up. If the modality images are not skull stripped and segmented, there is a possibility to do so using the buttons "Preprocess" and "Segment" respectivelly. If the user wants to visualize the modules, he can click the button "Visualize".


