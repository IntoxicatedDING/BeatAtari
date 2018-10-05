# DQN-Beat-Atari
Keras/TensorFlow implementation of an agent to beat Atari based on deep Q-Network.  
This implementation has referred to the [code](https://github.com/devsisters/DQN-tensorflow.git) by devsisters.
However, the tricks to make the model converge are too complicated and the agent can rarely achieve more than 100 points. You have to carefully adjust the hyper-parameters. 
To solve this problem, I apply **layer normalization** to each layer and finally the agent gets a much better performance

## Environment
- Python 3.6
- TensorFlow  
- Keras  
- Opencv  
- Gym
- [PlaidML](https://github.com/plaidml/plaidml) (If you want to train on AMD GPU)


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

