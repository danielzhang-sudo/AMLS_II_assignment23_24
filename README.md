# AMLS_II_assignment23_24
Repository for ELEC0135 Project 2023/2024

### Description of the project

In this project we develop a single model for super resolution tasks. We use the Track 1 of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset with 4x bicubic downsampling. The structure 
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
|- test.py (Test the model with test set or custom dataset)
|- main.py (main python file)
|- README.md (Description of the project)
```

### How to run

To run the code in your local machine, first create a conda environment and install the dependecies with `conda install -f requirements.txt`. Next, create a folder called `Datasets` as shown above and put the image files in 
there. Finally open the terminal in the project directory and execute the following:

`python main.py`

This will run the model in training mode with a default of 150 epochs for pretraining and 100 epochs for finetuning.

Some of the accepted arguments are:
--augmentations (int) : with an integer determining the augmentation factor. Default is 3
--pre_epochs (int) : with an integer determining the number of pretraining epochs. Default is 150
--fine_epochs (int) : with an integer determining the number of finetuning epochs. Default is 100
--finetune : add argment when resuming training. Default is False
--path (str) : with the path to the last epoch
--test : add argument when testing the model. Default is False
--test_lr_path (str) : with path to LR test images
--test_hr_path (str) : with path to HR test images

To run in testing mode, it is necessary to include the path to the saved model to load the weights, and the path to both the LR image folder and the HR resolution folder.
