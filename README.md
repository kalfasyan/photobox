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
## Installation

1. First install `pcmanfm` file manager by running this in your terminal:   
`sudo apt-get update -y`  
`sudo apt-get install -y pcmanfm`  
2. Install the `photobox` package :  
`pip install photobox`
3. Clone this repository and download the [model](https://kuleuven-my.sharepoint.com/:u:/g/personal/ioannis_kalfas_kuleuven_be/EUBAo2_hrLdKu3Dw0bhg8NkBm_PoJ3AvV2VWOUBqvlhikg?e=ltM0a2) (unskilled model shared for demo purposes) in the photobox folder - same dir as `photobox_app.py`.
4. From that directory run:  
   `python photobox_app.py`
---------
## Workflow
### 1. GUI operations
Here is an overview of the buttons' functionality:
![img1](./images/pbox_slide1.png?raw=True "GUI Steps")
### 2. Object detection
The software removes spatial distortion, crops and then thresholds the image to detect objects (insects). One could improve this step by adding a smart object detector, however this software was initially implemented for a Raspberry Pi and that would make it much slower. 
![img2](./images/pbox_slide2.png?raw=True "GUI Steps")
### 3. Model inference
A trained insect image classifier is fed with each insect image (150x150 pixels) and provides the user with the maximum probabilities per detection. Insects that do not belong to the "critical insect list" which is user-defined, are all shown in blue. For the "critical insects", a light green color is shown for the ones that the model showed a probability score > 75% and a red color for the rest.
![img3](./images/pbox_slide3.png?raw=True "GUI Steps")
### 4. Human verification
The user is asked to verify each "critical-insect" detection and then save the results. The session folder will then contain csv files and histogram plots with the counts per sticky plate.
![img4](./images/pbox_slide4.png?raw=True "GUI Steps")

### DISCLAIMER
--------
All data in this repo belong to KU Leuven.
