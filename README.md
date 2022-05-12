++========================
# PhotoBox v0.0.3
==========================
#### *Author:*    Ioannis Kalfas
#### *Contact:*   kalfasyan@gmail.com , ioannis.kalfas@kuleuven.be
#### *Role:*      PhD Researcher
#### *Division:*  MeBioS, KU Leuven
--------
## Features

* Handles imaging sessions
* Capture a sticky plate image
* Spatial (applied) and color calibration (available).
* Cropping.
* Object detection on the captured image.
* Model inference on detected objects/insects.
* Validation procedure for entomology experts.
* Exporting all results in csv files.
* Creating plots with insect counts.
* Works on both a Raspberry Pi and any PC.
--------
### Installation

1. First install `pcmanfm` file manager by running this in your terminal:   
`sudo apt-get update -y`  
`sudo apt-get install -y pcmanfm`  
2. Install the `photobox` package :  
`pip install photobox`
3. Download the [package files](https://pypi.org/project/photobox/#files) (photobox-X.Y.Z.tar.gz), the [extra files](https://kuleuven-my.sharepoint.com/:f:/g/personal/ioannis_kalfas_kuleuven_be/EtfN74iqV5NJspAMBYX4b_UB3ynlQXdfRn7OC21v9T4EVA?e=LXv7HB), and the [model](https://kuleuven-my.sharepoint.com/:u:/g/personal/ioannis_kalfas_kuleuven_be/EXp7NdaetJhJhBMhq_-ax5ABWe1mrqV9VY223PTLwj4EGA?e=mIC1fd) (an unskilled model is shared for demo purposes). 
4. Extract the package files and then move the extra files and the model in the extracted directory, in the photobox folder (same dir as `photobox_app.py`).
5. From that directory run:  
   `python photobox_app.py`
  
  
### DISCLAIMER
--------
All data in this repo belong to KU Leuven.