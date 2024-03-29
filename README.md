# Automatic Detection of Mode Interactions Using Machine Learning
**Supervised by Dr Renson**\
**Imperial College London**\
**Department of Mechanical Engineering**

## Context and introduction

As engineers embrace more complex nonlinear structures, and approach problems with methods different to usual linear approximations, they must face the wide variety of nonlinear structure responses. One of them is coupled vibration modes. Coupled vibration modes result in energy exchanges between these modes. This poses a challenging and dangerous threat to the integrity of the concerned structures. Indeed, energy transfers from a local mode of low effective mass components to a global mode with high effective mass might critically jeopardize the structure.\
Predicting and detecting these interactions is therefore a crucial security stake. Nowadays, these interactions are detected by trained-eyed physicists and engineers. They conduct frequency sweeps and inspect scalograms of the structure’s response. This time-consuming verification could be automated using well-established Machine Learning techniques.\
The aim of this project is therefore to add modal interaction detection to the long list of Machine Learning applications and to carry out the automation of that detection.

## Project objectives

This project aims to automatise the detection of mode interactions relying on established Machine Learning techniques. Therefore, the project’s first objective is to **learn how to recognise these interactions** and understand the **conditions and parameters that favour them**.\
Physicists usually rely on scalogram analysis to detect interactions. The algorithms will have to be trained with those scalograms, which entails **creating scalogram datasets** and classifying them according to whether they present modal interactions. We will also have to write scaling and pre-processing algorithms to obtain clean data and increase the accuracy and efficiency of the machine learning models.\
We will then **train and test different Machine Learning models** and techniques in order to select those best suited for this classification problem. Finally the algorithms will be **tested on real-life data** and examples.

## Instructions
The report details how to use the modules and what the parameters are. The demo files could not be uploaded on github as they are too heavy, the databases X.csv and y.csv could not be uploaded for the same reason. Some parts of the demo and parameters_test codes will not run without those files. However, their results can be found in the report as well.
