#!/usr/bin/env python

'''
this script compares the two agent in normal environment,
we expect the double pro gives a better result than the single pro

this require us to load 2 checkpoints, one for single and one for double
'''

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

N_Adv_A = 1 #dimension of action space of adversary agent
ADV_BOUND = [i*0.05 for i in A_BOUND]# the external force for the adv is a little smaller


def showoff(env, global_agent):
    print ("now showoff result")
    AC = ACNet('showoff_agent', global_agent)
    AC.pull_global()

    for episodes in range(5):
       state = env.reset()
       reward_all = 0
       while(True):
           env.render()
           action = AC.choose_action(state)
           state_new, reward, done, _ = env.step(action) # take a random action
           reward_all = reward_all + reward
           if done:
               break
           state = state_new
       print ("episode:", episodes, ",reward: ", reward_all)

    reward_all_track = []
    for episodes in range(30):
        state = env.reset()
        reward_all = 0
        for i in range(1000):
            action = AC.choose_action(state)
            state_new, reward, done, _ = env.step(action) # take a random action
            reward_all = reward_all + reward
            if done:
                break
            state = state_new
        reward_all_track.append(reward_all)
    print(reward_all_track)
    print( "final reward", np.mean(reward_all_track[-100:]))
    return

def showoffReal(global_agent):
    print ("now showoff result")
    AC = ACNet('showoff_agent', global_agent)
    AC.pull_global()
    for i in range(5):
        buffer_s, buffer_a, buffer_r = [], [], []
        step = 0
        ep_r = 0
        s = request_init()
        while(True):
            step += 1

            a = AC.choose_action(s)
            res = request_torque(step, a)
            print "state:", s, ",action:", a[0], ",\treward:", res.reward
            # res = request_torque(step, env.random_action()[0]*4-2)
            # res = request_torque(step, 0)
            s_ = np.array(res.state_new)
            s = s_
            ep_r += res.reward

            if res.done:
                res = request_torque(step, 0)
                print("done episode, reward:", ep_r)
                break


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = tf.stop_gradient(normal_dist.entropy())  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]

class ACNetAdv(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope+'_adv'):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_adv' + '/actor_adv')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_adv' + '/critic_adv')
        else:   # local net, calculate losses
            with tf.variable_scope(scope+'_adv'):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_Adv_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.mu, self.sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss_adv'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.mu, self.sigma = self.mu * ADV_BOUND[1], self.sigma + 1e-4

                self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                with tf.name_scope('a_loss_adv'):
                    log_prob = self.normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = self.normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a_adv'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(self.normal_dist.sample(1), axis=0), ADV_BOUND[0], ADV_BOUND[1])
                with tf.name_scope('local_grad_adv'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_adv' + '/actor_adv')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '_adv' + '/critic_adv')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync_adv'):
                with tf.name_scope('pull_adv'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push_adv'):
                    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
                    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self ):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor_adv'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_Adv_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_Adv_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic_adv'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]

    def get_norm(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run([self.mu, self.sigma], {self.s: s})	



if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        GLOBAL_AC_ADV = ACNetAdv(GLOBAL_NET_SCOPE)

    COORD = tf.train.Coordinator()
    saver = tf.train.Saver()
    SESS.run(tf.global_variables_initializer())

    saver.restore(SESS, 'model_adv/single-1321')

    showoff(env, GLOBAL_AC)

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()