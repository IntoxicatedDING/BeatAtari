import os
from keras import backend as K
# ================================================
# use plaidml as backend
# install plaidml:
# pip install plaidml-keras
# plaidml-setup
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# ================================================
# use tensorflow as backend
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
# ================================================

import numpy as np
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.layers import Convolution2D, Lambda
from keras.engine.topology import Layer
from keras import optimizers
from keras import initializers
from keras import losses
import gym
import cv2
from collections import deque
import random

# ================================================
# set the boolean variable TRAIN to true to train the model
# set the boolean variable INITIAL to true to initialize the model
TRAIN = False
INITIAL = True & TRAIN
# ================================================

NUM_ACTIONS = 4
STATE_LENGTH = 4
EPSILON_INITIAL = 1.
EPSILON_FINAL = 0.05
EXPLORATION_STEPS = 500000
MEMORY_LOAD = 100000
SAVE_INTERVAL = 200
BATCH_SIZE = 32
TRAIN_INTERVAL = 4
UPDATE_TARGET_NETWORK_INTERVAL = 1
GAMMA = 0.99
TAU = 0.001
LEARNING_RATE_START = 1e-4
LEARNING_RATE_END = 5e-5
SHAPE = (None, 93, 80, STATE_LENGTH)


def dense_to_one_hot(data, depth=10):
    return (np.arange(depth) == np.array(data)[:, None]).astype(np.bool)


def rgb2gray(im):
    return (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)).astype(np.uint8)


def down_sample(gray):
    return gray[25::2, ::2]


class LayerNormalization(Layer):

    def __init__(self, eps=1e-5, activation=None, **kwargs):
        self.eps = eps
        self.channels = None
        self.activation = activation
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        shape = [1] * (len(input_shape) - 1)
        shape.append(self.channels)
        self.add_weight('gamma', shape, dtype='float32', initializer='ones')
        self.add_weight('beta', shape, dtype='float32', initializer='zeros')

        super(LayerNormalization, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        dim = len(K.int_shape(inputs)) - 1
        mean = K.mean(inputs, axis=dim, keepdims=True)
        var = K.mean(K.square(inputs - mean), axis=dim, keepdims=True)
        outputs = (inputs - mean) / K.sqrt(var + self.eps)
        outputs = outputs * self.trainable_weights[0] + self.trainable_weights[1]
        if self.activation is None:
            return outputs
        else:
            return self.activation(outputs)
        # inputs_reshaped = K.reshape(inputs, (-1, self.channels))
        # mean = K.mean(inputs_reshaped, axis=1, keepdims=True)
        # var = K.mean(K.square(inputs_reshaped - mean), axis=1, keepdims=True)
        # outputs = (inputs_reshaped - mean) / K.sqrt(var + self.eps)
        # outputs = outputs * self.trainable_weights[0] + self.trainable_weights[1]
        # if self.activation is None:
        #     return K.reshape(outputs, K.shape(inputs))
        # return self.activation(K.reshape(outputs, K.shape(inputs)))


class Agent:
    def __init__(self, root, init=INITIAL):
        self.root = root
        self.batch = deque()
        self.q_out, self.model_train = Agent.build_train_network()
        self.model_target = Agent.build_target_network()

        if init:
            self.episode = 1
            self.decay_step = 0
            self.learning_rate = LEARNING_RATE_START
            self.epsilon = EPSILON_INITIAL
        else:
            self.episode, self.decay_step, self.epsilon, self.learning_rate = self.restore()
        # self.opt = optimizers.rmsprop(lr=self.learning_rate, rho=0.95)
        self.opt = optimizers.adam(lr=self.learning_rate)
        # self.opt = optimizers.RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        self.model_train.compile(optimizer=self.opt, loss=[Agent.huber_loss])

    def feed_batch(self, data):
        if len(self.batch) >= MEMORY_LOAD:
            self.batch.popleft()
        self.batch.append(data)
        return self.batch

    def sample_batch(self):
        batch_q, batch_state, batch_mask, states_next, rewards, done =\
            map(lambda x: np.array(list(x)), zip(*random.sample(self.batch, BATCH_SIZE)))
        batch_state = np.transpose(batch_state, axes=[0, 2, 3, 1])
        states_next = np.transpose(states_next, axes=[0, 2, 3, 1])
        batch_mask = dense_to_one_hot(batch_mask, NUM_ACTIONS)
        q_next = self.model_target.predict(states_next)
        batch_q[batch_mask] = np.array(rewards) + GAMMA * np.array(done) * np.max(q_next, axis=1)
        return batch_q, batch_state, batch_mask

    @staticmethod
    def build_train_network():
        X = Input(shape=SHAPE[1:], dtype='float32')
        mask = Input(shape=(NUM_ACTIONS,), dtype='float32')
        q_out, model = Agent.infer(X)
        q_ = Lambda(lambda x: K.reshape(K.sum(x * mask, axis=1), (-1, 1)), output_shape=(1,))(q_out)
        return K.function([X], [q_out]), Model(inputs=[X, mask], outputs=q_)

    @staticmethod
    def huber_loss(x, y):
        error = K.abs(x - y)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part, axis=-1)
        return loss

    @staticmethod
    def build_target_network():
        X = Input(shape=SHAPE[1:], dtype='float32')
        Q, model = Agent.infer(X, trainable=False, init=initializers.zeros())
        return model

    @staticmethod
    def infer(X, trainable=True, init=initializers.truncated_normal(stddev=0.01)):
        init_w = init
        init_b = initializers.constant(0.)
        normed = Lambda(lambda x: x / 255., output_shape=K.int_shape(X)[1:])(X)
        h_conv1 = Convolution2D(32, (8, 8), strides=(4, 4),
                                kernel_initializer=init_w, use_bias=False, padding='same')(normed)
        h_ln1 = LayerNormalization(activation=K.relu)(h_conv1)
        h_conv2 = Convolution2D(64, (4, 4), strides=(2, 2),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln1)
        h_ln2 = LayerNormalization(activation=K.relu)(h_conv2)
        h_conv3 = Convolution2D(64, (3, 3), strides=(1, 1),
                                kernel_initializer=init_w, use_bias=False, padding='same')(h_ln2)
        h_ln3 = LayerNormalization(activation=K.relu)(h_conv3)
        h_flat = Flatten()(h_ln3)
        fc_advantage = Dense(512, use_bias=False, kernel_initializer=init_w)(h_flat)
        h_ln_fc_advantage = LayerNormalization(activation=K.relu)(fc_advantage)
        advantage = Dense(NUM_ACTIONS, kernel_initializer=init_w,
                          use_bias=False, bias_initializer=init_b)(h_ln_fc_advantage)
        fc_value = Dense(512, use_bias=False, kernel_initializer=init_w)(h_flat)
        h_ln_fc_value = LayerNormalization(activation=K.relu)(fc_value)
        value = Dense(1, kernel_initializer=init_w, use_bias=False, bias_initializer=init_b)(h_ln_fc_value)
        z = Lambda(lambda x: x[1] + x[0] - K.mean(advantage, axis=1, keepdims=True), output_shape=(NUM_ACTIONS,))([advantage, value])
        # z = LayerNormalization()(fc2)
        model = Model(inputs=X, outputs=z)
        model.trainable = trainable
        return z, model

    def train(self):
        batch_q, batch_state, batch_mask = self.sample_batch()
        self.model_train.fit([batch_state, batch_mask], np.sum(batch_mask * batch_q, axis=1), verbose=0)

    def update_epsilon(self):
        self.epsilon = np.maximum(EPSILON_FINAL,
                                  self.epsilon - (EPSILON_INITIAL - EPSILON_FINAL) / EXPLORATION_STEPS)

    def predict(self, state):
        q = self.q_out([state])
        q = np.array(q).flatten()
        # print(np.argmax(q))
        # print(q)
        return q, np.argmax(q)

    def update_learning_rate(self):
        self.learning_rate = self.learning_rate * (0.99 ** (self.decay_step / 1000))
        K.set_value(self.model_train.optimizer.lr, self.learning_rate)
        self.decay_step += 1

    def update_target_network(self):
        self.model_target.set_weights(self.model_train.get_weights())
        # self.sess.run(self.update_param)

    def save(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.model_train.save_weights(os.path.join(self.root, 'model_train.h5'))
        self.model_target.save_weights(os.path.join(self.root, 'model_target.h5'))
        # self.model_train.save(os.path.join(self.root, 'model_train.h5'))
        # self.model_target.save(os.path.join(self.root, 'model_target.h5'))
        np.save(os.path.join(self.root, 'params'), [self.episode, self.decay_step, self.epsilon, self.learning_rate])

    def restore(self):
        self.model_train.load_weights(os.path.join(self.root, 'model_train.h5'))
        self.model_target.load_weights(os.path.join(self.root, 'model_target.h5'))
        episode, decay_step, eps, learning_rate = np.load(os.path.join(self.root, 'params.npy'))
        return int(episode), int(decay_step), eps, learning_rate


if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')
    ROOT = 'Model_Duel'
    agent = Agent(ROOT)
    if TRAIN:
        is_done = False
        cnt_frames = 0
        while True:
            frame = env.reset()
            is_done = False
            frame = down_sample(rgb2gray(frame))
            frame_stack = [frame, frame, frame, frame]
            points = 0
            while not is_done:
                cnt_frames += 1
                state_current = frame_stack[-STATE_LENGTH:]
                q_current, action = agent.predict(np.expand_dims(np.transpose(state_current, [1, 2, 0]), axis=0))
                eps = agent.epsilon
                if np.random.random() < eps:
                    action = env.action_space.sample()
                frame, reward, is_done, _ = env.step(action)
                points += reward
                frame_next = down_sample(rgb2gray(frame))
                frame_stack.append(frame_next)
                state_next = frame_stack[-STATE_LENGTH:]
                training_data = (q_current, state_current, action, state_next, np.sign(reward), int(not is_done))
                agent.feed_batch(training_data)
                if len(agent.batch) >= BATCH_SIZE and cnt_frames % TRAIN_INTERVAL == 0:
                    agent.train()
                    agent.update_epsilon()
                frame_stack = frame_stack[-STATE_LENGTH:]

            eps = agent.epsilon
            lr = agent.learning_rate
            print('episode: %d, points: %d,'
                  ' epsilon: %f, learning_rate %.7f' % (agent.episode, points, eps, lr))
            if np.abs(eps - EPSILON_FINAL) < 1e-5 and lr > LEARNING_RATE_END:
                agent.update_learning_rate()
            agent.save()
            if agent.episode % UPDATE_TARGET_NETWORK_INTERVAL == 0:
                agent.update_target_network()
            agent.episode += 1
    else:
        is_done = False
        cnt_frames = 0
        while True:
            frame = env.reset()
            is_done = False
            frame = down_sample(rgb2gray(frame))
            frame_stack = [frame, frame, frame, frame]
            points = 0
            while not is_done:
                env.render()
                state_current = frame_stack[-STATE_LENGTH:]
                _, action = agent.predict(np.expand_dims(np.transpose(state_current, [1, 2, 0]), axis=0))
                eps = agent.epsilon
                if np.random.random() < eps:
                    action = env.action_space.sample()
                frame, reward, is_done, _ = env.step(action)
                points += reward
                frame_next = down_sample(rgb2gray(frame))
                frame_stack.append(frame_next)
                frame_stack = frame_stack[-STATE_LENGTH:]
            print('Scores: %d' % points)