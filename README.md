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
3. Clone this repository and download the [model](https://kuleuven-my.sharepoint.com/:u:/g/personal/ioannis_kalfas_kuleuven_be/EUBAo2_hrLdKu3Dw0bhg8NkBm_PoJ3AvV2VWOUBqvlhikg?e=ltM0a2) (unskilled model shared for demo purposes) in the photobox folder - same dir as `photobox_app.py`.
4. From that directory run:  
   `python photobox_app.py`
  
  
### DISCLAIMER
--------
All data in this repo belong to KU Leuven.
