# PER_DQN
DQN with Prioritized Experience Replay 

# Requirements
gym == 0.17.3

pytorch-lightining == 1.6.0

pyglet == 1.5.27

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install pygame gym==0.17.3 pytorch-lightning==1.6.0 pyvirtualdisplay optuna

!pip install git+https://github.com/GrupoTuring/PyGame-Learning-Environment

!pip install git+https://github.com/lusob/gym-ple


# Description
Prioritized Experience Replay DQN is a continuation of double DQN in order for the Deep Q Network to store data with higher priority, this allows the network to have a higher chance of training with data that has higher expected return or that it can be relevant in improving the network weights. This method allows for the network to optimize based on relevant data and therefor allowing convergence to happen faster.

# Game
Flappy Bird

# Architecture
Double DQN with Priotirized Experience Replay

# optimizer
AdamW

# Loss
smooth L1 loss function

