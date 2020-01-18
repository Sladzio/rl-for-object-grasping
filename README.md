# RL for object grasping by robot (Franka Emika)


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

**This project requires _python3 (>=3.5 and <=3.7)_** with the development headers.

##### Before installing the required dependencies, you may want to create a virtual environment (using [pyenv](https://github.com/pyenv/pyenv)) with specified python version and choose it:
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


## Instalation

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

