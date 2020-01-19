# RL for object grasping by robot (Franka Emika)


## Overview
 - [Prerequisites](#prerequisites)
    - [System](#system)
    - [Python](#python)
 - [Installation](#installation)
    - [Install using requirements](#install-using-requirements)
    - [Install using pip](#install-using-pip)
 - [Train and Test](#train-and-test)
 - [Tensorboard](#tensorboard)
 - [Documentation](#documentation)



## Prerequisites

**Note:** Tensorflow currently (01.2020) is not working with python in version 3.8! This was tested with python3.6.9.


### System

You'll need system packages: **CMake**, **OpenMPI** and **zlib**. Those can be installed as follows

#### Ubuntu

```
$ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```
$ brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).


### Python

This project **requires _python3 (>=3.5 and <=3.7)_** with the development headers.

Before installing the required dependencies, you may want to create a virtual environment (using [pyenv](https://github.com/pyenv/pyenv)) with specified python version and choose it:
```
$ pyenv virtualenv 3.6.9 python3.6.9
$ pyenv shell python3.6.9
```
(After work) To return to system python:
```
$ pyenv shell system
```
###### If it does not work you may need to add those lines in *.bash_profile* or .bashrc file depending on terminal for pyenv to work:
    
```
$ eval "$(pyenv init -)"
$ eval "$(pyenv virtualenv-init -)"
```



## Installation

##### Clone the repository:
```
$ git clone https://github.com/Sladzio/rl-for-object-grasping.git
$ cd rl-for-object-grasping
```


### Install using requirements
##### Install all the necessary dependencies:
```
$ pip3 install -r requirements.txt
```
**Note:** Installing the requirements will install also [Stable Baselines](https://github.com/hill-a/stable-baselines), [Pybullet](https://github.com/bulletphysics/bullet3), [Tensorflow](https://github.com/tensorflow/tensorflow) and [Gym](https://github.com/openai/gym).

OR


### Install using pip
##### Install the Stable Baselines package:
```
$ pip3 install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
$ pip3 install stable-baselines
```

##### Install the Pybullet package:
```
$ pip3 install pybullet
```

##### Install the ruamel.yaml package:
```
$ pip3 install ruamel.yaml
```

##### Install the Tensorflow 1.x (1.8 <= x <= 1.15) package, required by Stable Baselines:
```
$ pip install tensorflow==1.15
```



## Train and Test
To train model run any train_\*.py file, where time_steps is the number of steps in total the agent will do for the environment.
Temporary used models are saved in ./tmp/ folder as best_model.pkl. During training you can watch learning process with tensorboard.
After training is done, model will be saved as model.pkl in current directory, you can test it running corresponding test_\*.py file.



## Tensorboard
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
##### 1. Run tensorboard by specifying the log directory:
```
$ tensorboard --logdir ../tensorboard
  TensorBoard 1.13.1 at <url>:6006 (Press CTRL+C to quit)
```
##### 2. Enter the \<url\>:6006 into the web browser and track the mean reward per episode.



## Documentation

Stable Baselines online [documentation](https://stable-baselines.readthedocs.io/) for more details or in [pdf](https://buildmedia.readthedocs.org/media/pdf/stable-baselines/v1.0.7/stable-baselines.pdf) file.

Pybullet [documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/preview#heading=h.2ye70wns7io3) for more details.


