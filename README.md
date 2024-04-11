# AMLS_II_assignment23_24
Repository for ELEC0135 Project 2023/2024

### Description of the project

In this project we develop a single model for super resolution tasks. We use the Track 1 of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset with 3x bicubic downsampling. The structure 
of the project is the following:

```
AMLS_II_assignment_24/
|- figures (Figures saved from Task A)
|  |-
|  |-
|  |- ...
|- weights (Weights of last epoch of training of the model)
|  |- 249_model.pt
|  |- ...
|- Datasets/ (Datasets folder - include your dataset for training here, the order of the images must coincide for every LR and HR pair)
|  |- hr (folder for HR images)
|  |- lr (folder for LR images)
|- data.py (Data preprocessing python code): It creates a torch dataset with each image pair, which has been randomly cropped, flipped and rotated.
|- perceptual-.py (VGG model python code): It contains the code to client_loop: send disconnect: Connection reset
PS C:\Users\dani0>
|- wgan.py (Discriminiator model python code)
|- test
|- main.py (main python file)
|- README.md (Description of the project)
```

### How to run

To run the code in your local machine, first install the dependecies with `pip install -f requirements.txt`. Next, create a folder called `Datasets` as shown above and put the datasets files in 
there. Finally open the terminal in the project directory and execute the following:

`python main.py`

Some of the accepted arguments are: 

To run in testing mode, it is necessary to include the path to the saved model to load the weights, and the path to both the LR image folder and the HR resolution folder.

The code trains, validates or tests one model at a time and saves figures and reports about its results in their respective folders.
