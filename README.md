# Visual State Represenstation Learning for Reinforcement Learning - A Framework

Framework to integrate different state representation learning methods with reinforcement learning methods.
## Installation

Depending on which Environments are intended to use go to respective section to first install the prerequisites.

1. Pull this repo with submodules
2. Create conda env using environment.yaml

### 1) Mujoco Install and Prerequisites
Mujoco is needed for the following environments:

1. Download mjpro200_linux from the MuJoCo site.
2. Create a folder .mujoco in your home directory and extract the contents of mujoco200 there.
3. Apply for license on mujoco’s site.
4. Put your license file in the .mujoco/mujoco200/bin folder. If problems put in ./mujoco folger
5. Add a line in your ~/.bashrc file to add above path to LD_LIBRARY_PATH variable: 
    ```
    # Mujoco location 
    export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH"
    ```

The first time a mujoco environment is executed, it will be compiled. There might be need for the following packages:
```
sudo apt-get update
sudo apt-get install libglfw3 libglew2.0 libosmesa6-dev
```

### 2) Pygame Install and Prerequisites

Pygame pre:
```
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

### 3) Vizdoom Install and Prerequisites
```
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm wget
```
### 4) Maze Install and Prerequisites:

```
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
```

### Troubleshouting

To use miniworld and duckietown move to the specific submodule folder and:
```
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

Eventually there are some requirements missing. Check setup.py of the specific submodule an install manualley or run (might leed to version conflicts):
```
cd gym-duckietown
pip install -e .
cd gym-duckietown
pip install -e .
cd MazeExplorer
pip install -e .
```

# Implementation
Some code is inspired by/based on the following implementations:
- SLAC: https://github.com/ku2482/slac.pytorch
- Observation Wrapper Robot Env: https://github.com/PhilipZRH/ferm
- SAC/ PixelEncoder/ PixelDecoder: https://github.com/denisyarats/pytorch_sac_ae
- Contrastive Model: https://github.com/MishaLaskin/curl
- ppo2: https://github.com/DLR-RM/stable-baselines3
- sac/ppo: https://github.com/openai/spinningup
- trpo: https://github.com/Nirnai/DeepRL
- DIM/ Impala: https://github.com/mila-iqia/atari-representation-learning