# BeatAtari
Implementation of an agent to beat Atari based on deep Q-Network.

## Environment
- Python 3.6
- TensorFlow  
- Keras  
- Opencv  
- Gym
- PlaidML (If you want to train on AMD GPU)


Install prerequisites with:

    $ pip install gym[atari]
    $ pip install plaidml-keras
    $ pip install opencv-python
    $ pip install tensorflow

Setup PlaidML with:

    $ plaidml-setup
    

## Results

Result of training for about 20 hours using GTX 1080 ti.
Actually, you can break through 300 points by 10000 games training, which takes about 10 hours on GTX 1080 ti.

![best](BeatAtari.gif)

