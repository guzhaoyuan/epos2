#!/usr/bin/env python3

#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import scipy.signal
import argparse

import multiprocessing
import threading
import os
import time
from datetime import datetime

import gym
from gym import wrappers

from utils import Logger, Scaler
import signal

from epos2.srv import *
import rospy
import sys

N_WORKERS = 1
episode = 0
GLOBAL_NET_SCOPE = 'Global_Net'

def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        # print(policy.name)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})
    print(np.mean([t['rewards'].sum() for t in trajectories]))
    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


class ACNet(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, scope, globalAC=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = scope
        with tf.variable_scope(scope):
            with tf.variable_scope('critic'):
                self.replay_buffer_x = None
                self.replay_buffer_y = None
                self.epochs_c = 10
                self.lr_c = None  # learning rate set in _build_graph()
                self._build_graph_critic()
            with tf.variable_scope('actor'):
                self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
                self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
                self.kl_targ = kl_targ
                self.epochs_a = 20
                self.lr = None
                self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
                self._build_graph_actor()
            with tf.name_scope('local_grad'):
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor' + '/policy_nn')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic' + '/value_nn')
                self.a_grads = tf.gradients(self.loss, self.a_params)
                self.c_grads = tf.gradients(self.loss_c, self.c_params)
        if scope != GLOBAL_NET_SCOPE:# non global agent update global parameter
            self.globalAC = globalAC
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    # OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
                    # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
                    # self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    # self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    optimizer_c = tf.train.AdamOptimizer(self.lr_c)
                    optimizer = tf.train.AdamOptimizer(self.lr_ph)
                    self.train_op_c = optimizer_c.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    self.train_op = optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
        else:# global agent update itself
            optimizer_c = tf.train.AdamOptimizer(self.lr_c)
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
            self.train_op_c = optimizer_c.apply_gradients(zip(self.c_grads, self.c_params))
            self.train_op = optimizer.apply_gradients(zip(self.a_grads, self.a_params))

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def update_a(self, feed_dict):
        SESS.run(GLOBAL_AC.train_op, feed_dict)
        # SESS.run(self.pull_a_params_op)

    def update_global_a(self, feed_dict):
        SESS.run(self.train_op, feed_dict)

    def update_c(self, feed_dict):
        SESS.run(GLOBAL_AC.train_op_c, feed_dict = feed_dict)
        # SESS.run(self.pull_c_params_op)

    def update_global_c(self, feed_dict):
        SESS.run(self.train_op_c, feed_dict)

###################################################################################################
    def _build_graph_critic(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        # self.g = tf.Graph()
        # with self.g.as_default():
        with tf.variable_scope('placeholders_c'):
            self.obs_ph_c = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
        with tf.variable_scope('value_nn'):
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr_c = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr_c: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr_c))
            # 3 hidden layers with tanh activations
            out_c = tf.layers.dense(self.obs_ph_c, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out_c = tf.layers.dense(out_c, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out_c = tf.layers.dense(out_c, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out_c = tf.layers.dense(out_c, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out_c = tf.squeeze(out_c)
        with tf.name_scope('loss_train_op'):
            self.loss_c = tf.reduce_mean(tf.square(self.out_c - self.val_ph))  # squared loss
            # optimizer = tf.train.AdamOptimizer(self.lr_c)
            # self.train_op_c = optimizer.minimize(self.loss_c)

    def fit(self, x, y, logger):
        """ Critic Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = GLOBAL_AC.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if GLOBAL_AC.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, GLOBAL_AC.replay_buffer_x])
            y_train = np.concatenate([y, GLOBAL_AC.replay_buffer_y])
        GLOBAL_AC.replay_buffer_x = x
        GLOBAL_AC.replay_buffer_y = y
        for e in range(GLOBAL_AC.epochs_c):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {GLOBAL_AC.obs_ph_c: x_train[start:end, :],
                             GLOBAL_AC.val_ph: y_train[start:end]}
                # _ = SESS.run(self.train_op_c, feed_dict=feed_dict)
                GLOBAL_AC.update_c(feed_dict)
        y_hat = GLOBAL_AC.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Critic Predict method """
        feed_dict = {self.obs_ph_c: x}
        y_hat = SESS.run(self.out_c, feed_dict=feed_dict)

        return np.squeeze(y_hat)
####################################################################################################
    def _build_graph_actor(self):
        """ Build and initialize TensorFlow graph """
        # self.g = tf.Graph()
        # with self.g.as_default():
        with tf.variable_scope('placeholders_a'):
            """ Input placeholders"""
            # observations, actions and advantages:
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
            self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
            # strength of D_KL loss terms:
            self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
            self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
            # learning rate:
            self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
            # log_vars and means with pi_old (previous step's policy parameters):
            self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
            self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
        with tf.variable_scope('policy_nn'): 
            """ Neural net for policy approximation function
                Policy parameterized by Gaussian means and variances. NN outputs mean
                action based on observation. Trainable variables hold log-variances
                for each action dimension (i.e. variances not determined by NN).
            """
            # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
            hid1_size = self.obs_dim * 10  # 10 empirically determined
            hid3_size = self.act_dim * 10  # 10 empirically determined
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            self.means = tf.layers.dense(out, self.act_dim,
                                         kernel_initializer=tf.random_normal_initializer(
                                             stddev=np.sqrt(1 / hid3_size)), name="means")

            # logvar_speed is used to 'fool' gradient descent into making faster updates
            # to log-variances. heuristic sets logvar_speed based on network size.
            logvar_speed = (10 * hid3_size) // 48
            log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                       tf.constant_initializer(0.0))
            self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0
        with tf.name_scope('logprob'): 

            print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))
            """ Calculate log probabilities of a batch of observations & actions

            Calculates log probabilities using previous step's model parameters and
            new parameters being trained.
            """
            logp = -0.5 * tf.reduce_sum(self.log_vars)
            logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                         tf.exp(self.log_vars), axis=1)
            self.logp = logp

            logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
            logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                             tf.exp(self.old_log_vars_ph), axis=1)
            self.logp_old = logp_old

        with tf.name_scope('kl_entropy'): 
            """
            Add to Graph:
                1. KL divergence between old and new distributions
                2. Entropy of present policy given states and actions

            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
            """
            log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
            log_det_cov_new = tf.reduce_sum(self.log_vars)
            tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

            self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                           tf.reduce_sum(tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars), axis=1) -
                                           self.act_dim)
            self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))

        with tf.name_scope('loss_train_op'): 
            """
            Three loss terms:
                1) standard policy gradient
                2) D_KL(pi_old || pi_new)
                3) Hinge loss on [D_KL - kl_targ]^2

            See: https://arxiv.org/pdf/1707.02286.pdf
            """
            loss1 = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.loss = loss1 + loss2 + loss3
            # optimizer = tf.train.AdamOptimizer(self.lr_ph)
            # self.train_op = optimizer.minimize(self.loss)

        with tf.name_scope('sample'): 
            """ Sample from distribution, given observation """
            self.sampled_act = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)))
            
    def sample(self, obs):
        """Actor Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return SESS.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger):
        """ Actor Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        feed_dict = {GLOBAL_AC.obs_ph: observes,
                     GLOBAL_AC.act_ph: actions,
                     GLOBAL_AC.advantages_ph: advantages,
                     GLOBAL_AC.beta_ph: GLOBAL_AC.beta,
                     GLOBAL_AC.eta_ph: GLOBAL_AC.eta,
                     GLOBAL_AC.lr_ph: GLOBAL_AC.lr * GLOBAL_AC.lr_multiplier}
        old_means_np, old_log_vars_np = SESS.run([GLOBAL_AC.means, GLOBAL_AC.log_vars],
                                                      feed_dict)
        # print(old_means_np," ",old_log_vars_np)
        feed_dict[GLOBAL_AC.old_log_vars_ph] = old_log_vars_np
        feed_dict[GLOBAL_AC.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(GLOBAL_AC.epochs_a):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            # SESS.run(self.train_op, feed_dict)
            GLOBAL_AC.update_a(feed_dict)
            # print("run traino_p")
            loss, kl, entropy = SESS.run([GLOBAL_AC.loss, GLOBAL_AC.kl, GLOBAL_AC.entropy], feed_dict)
            # print(kl)
            if kl > GLOBAL_AC.kl_targ * 4:  # early stopping if D_KL diverges badly
                print("kl diverge")
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > GLOBAL_AC.kl_targ * 2:  # servo beta to reach D_KL target
            GLOBAL_AC.beta = np.minimum(35, 1.5 * GLOBAL_AC.beta)  # max clip beta
            if GLOBAL_AC.beta > 30 and GLOBAL_AC.lr_multiplier > 0.1:
                GLOBAL_AC.lr_multiplier /= 1.5
        elif kl < GLOBAL_AC.kl_targ / 2:
            GLOBAL_AC.beta = np.maximum(1 / 35, GLOBAL_AC.beta / 1.5)  # min clip beta
            if GLOBAL_AC.beta < (1 / 30) and GLOBAL_AC.lr_multiplier < 10:
                GLOBAL_AC.lr_multiplier *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.globalAC.beta,  
                    '_lr_multiplier': self.globalAC.lr_multiplier})

class Worker(object):
    def __init__(self, obs_dim, act_dim, kl_targ, name, globalAC):
        self.name = name
        self.globalAC = globalAC
        self.AC = ACNet(obs_dim, act_dim, kl_targ, name, globalAC)
        self.env = gym.make(env_name)#if use unwrap, the average exceeds 1000
        if self.name == 'W_0':
            self.env = wrappers.Monitor(self.env, aigym_path, force=True)
    def work(self):
        global episode
        
        while episode < num_episodes:
            lock.acquire()
            print("####################")
            self.AC.pull_global()
            trajectories = run_policy(self.env, self.AC, scaler, logger, episodes=batch_size)
            episode += len(trajectories)
            add_value(trajectories, self.AC)  # add estimated values to episodes
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # add various stats to training log:
            log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
            self.AC.update(observes, actions, advantages, logger)  # update policy
            self.AC.fit(observes, disc_sum_rew, logger)  # update value function
            print(self.name,", episode now:", episode)
            lock.release()
            time.sleep(0.05)
            # logger.write(display=True)  # write logger results to file and stdout





def request_torque(position, current, init=0):
    # print("wait for Service")
    #asset current in range(-2,2)
    rospy.wait_for_service('applyTorque')
    try:
        # print("now request service")
        applyTorque = rospy.ServiceProxy('applyTorque', Torque)
        # print("request service: ", current)
        res = applyTorque(position, current, init)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def request_init():
    # print("wait for Service")
    rospy.wait_for_service('applyTorque')
    try:
        # print("now request service")
        applyTorque = rospy.ServiceProxy('applyTorque', Torque)
        res = applyTorque(0, 0, 1);
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [position torque]"%sys.argv[0]

class Env(object):
    def __init__(self, env_name):
        self.name = env_name
        self.action_space = 1
        self.state_space = 3

    def random_action(self):
        return np.random.rand(self.action_space)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        torque = float(sys.argv[1])
        print(type(torque))
        res = request_torque(0, torque)
        print "position_new:", res.position_new, "\tvelocity:", res.velocity, "\t\t\treward:", res.reward#, "current:", res.current, 
    else:
        step = 0
        env = Env('Pendulum')
        request_init()
        while(True):
            # init the state by call env.reset(), getting the init state from the service
            # calculate the next move
            # call step service
            step += 1
            # print(env.random_action())
            # rospy.loginfo("request:%s",step)
            # if step % 2:
            #     res = request_torque(step, 10)
            # else:
            #     res = request_torque(step, 20)
            res = request_torque(step, env.random_action()[0]*2-1)
            print "position_new:", res.position_new, "\tvelocity:", res.velocity, "\treward:", res.reward#, "current:", res.current, 
            if res.done:
                print("done episode")
                res = request_torque(step+1, 0)
                break
            # rospy.loginfo("position_new:%s, velocity:%s, current:%s", res.position_new, res.velocity, res.current)
            # after getting the responce, calc the next move and call step service again

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-env_name', type=str, help='OpenAI Gym environment name',
                        default='InvertedPendulum-v1')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    args = parser.parse_args()
    
    env_name = args.env_name
    gamma = args.gamma
    lam = args.lam
    batch_size = args.batch_size
    kl_targ = args.kl_targ
    num_episodes = args.num_episodes

    lock = threading.Lock()

    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)
    SESS = tf.Session()
    
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """

    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    aigym_path = os.path.join('/tmp', env_name, now)
    # the Monitor is used to record video, no need for now
    # env = wrappers.Monitor(env, aigym_path, force=True)
    # run a few episodes of untrained policy to initialize scaler:
    scaler = Scaler(obs_dim)    
    
    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet(obs_dim, act_dim, kl_targ, GLOBAL_NET_SCOPE)
        
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(obs_dim, act_dim, kl_targ, i_name, GLOBAL_AC))
    
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    run_policy(env, GLOBAL_AC, scaler, logger, episodes=5)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
        time.sleep(0.5)
    COORD.join(worker_threads)

    logger.close()