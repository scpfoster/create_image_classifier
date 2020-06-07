# create_image_classifier
Project 2 for Udacity Intro to Machine Learning with Pytorch course

The project has 2 parts:
* a jupyter notebook 
* command line application

**WIP** All code is being developed and run in a Udacity workspace that is Cuda enabled.  

# Current Status

flowers is a dummy directory that does not contain real data for training/testing.  It only contains enough data to test connectivity.

## Part 1 Jupyter Notebook
Code in the notebook works in the Udacity workspace.  An HTML output of the last time it was run is included.

## Part 2 Command Line
Running the command line to train the model:
* python train.py flowers --gpu

should result in similar results to the Jupyter notebook since the default's for all commandline options result in the same model and data being used to train the notebook. 

Currently, this is not working as expected and is being debugged.
