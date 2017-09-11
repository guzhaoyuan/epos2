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
from epos2.srv import *
import rospy

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 4#multiprocessing.cpu_count()
MAX_EP_STEP = 250
MAX_GLOBAL_EP = 300
MAX_R = -1600
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.6
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_MEAN_R = []
GLOBAL_EP = 0
X_amp = 0.2
env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

N_Adv_A = 1 #dimension of action space of adversary agent
ADV_BOUND = [i*X_amp for i in A_BOUND]# the external force for the adv is a little smaller
print(A_BOUND, ADV_BOUND)

service = 'applyTorque3'
'''
this can only showoff single and double pro in real model for now
'''
def showoffReal(global_agent, nonStop = 0):
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
            if nonStop:
                res = request_torque(1, a)
            else:
                res = request_torque(step, a)
            print "state:", s, ",action:", a[0], ",\treward:", res.reward

            s_ = np.array(res.state_new)
            s = s_
            ep_r += res.reward

            if res.done:
                res = request_torque(step, 0)
                print("done episode, reward:", ep_r)
                break

def showoffRealAdv(global_agent, global_agent_adv, nonStop = 0):
    print ("now showoff result")
    AC = ACNet('showoff_agent', global_agent)
    AC.pull_global()
    AC_adv = ACNetAdv('showoff_agent', global_agent)
    AC_adv.pull_global()
    for i in range(5):
        buffer_s, buffer_a, buffer_r = [], [], []
        step = 0
        ep_r = 0
        s = request_init_adv()
        while(True):
            step += 1

            a = AC.choose_action(s)
            a_adv = AC_adv.choose_action(s)

            if nonStop:
                res = request_torque_adv(1, a, a_adv)
            else:
                res = request_torque_adv(step, a, a_adv)
            print "state:", s, ",action:", a[0], ",\treward:", res.reward

            s_ = np.array(res.state_new)
            s = s_
            ep_r += res.reward

            if res.done:
                res = request_torque_adv(step, 0, 0)
                print("done episode, reward:", ep_r)
                break


def request_torque_adv(position, current, current2, init=0):
    # print("wait for Service")
    #asset current in range(-2,2)
    rospy.wait_for_service(service)
    try:
        # print("now request service")
        applyTorque = rospy.ServiceProxy(service, Torque2)
        # print("request service: ", current)
        res = applyTorque(position, current, current2, init)
        return res
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def request_init_adv():
    # print("wait for Service")
    rospy.wait_for_service(service)
    try:
        # print("now request service")
        applyTorque = rospy.ServiceProxy(service, Torque2)
        res = applyTorque(0, 0, 0, 1)
        return np.array(res.state_new)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

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
        # GLOBAL_AC_ADV = ACNetAdv(GLOBAL_NET_SCOPE)

    COORD = tf.train.Coordinator()
    saver = tf.train.Saver(max_to_keep=100)

#####################
    # have to reload double model first, this will init the double-adv agent
    GLOBAL_AC_ADV = ACNetAdv(GLOBAL_NET_SCOPE)

    AC = ACNet('showoff_agent', GLOBAL_AC)
    AC_Adv = ACNetAdv('showoff_agent', GLOBAL_AC_ADV)

    SESS.run(tf.global_variables_initializer())
    restore_file = 'model_adv_real/double-4367-113-42-214'
    # restore_file = 'model/ckpt-64'
    saver.restore(SESS, restore_file)
    # showoff(env, GLOBAL_AC,1)   
    # showoff_in_Adv(env, GLOBAL_AC, GLOBAL_AC_ADV, 1)
    # showoffReal(GLOBAL_AC, 1)

    # saver.restore(SESS, 'model_adv/single-1243')
    # # showoff(env, GLOBAL_AC,0)
    # # showoff_in_Adv(env, GLOBAL_AC, GLOBAL_AC_ADV, 0)
    # showoffReal(GLOBAL_AC, 1)

#####################

    # showoffReal(GLOBAL_AC,1)
    # showoffRealAdv(GLOBAL_AC, GLOBAL_AC_ADV)

    AC.pull_global()
    AC_Adv.pull_global()

    total_step = 1
    buffer_s, buffer_a, buffer_r, buffer_a_adv, buffer_r_adv = [], [], [], [], []
    while not rospy.is_shutdown() and GLOBAL_EP < MAX_GLOBAL_EP:
        s = request_init_adv()
        ep_r = 0
        for ep_t in range(MAX_EP_STEP):
            # if self.name == 'W_0':
            #     self.env.render()
            a = AC.choose_action(s)
            a_adv = AC_Adv.choose_action(s)
            if GLOBAL_EP < 100:
                res = request_torque_adv(ep_t, a, 0)
            else:
                res = request_torque_adv(ep_t, a, a_adv)
            print "state:", s, ",actions:", a[0],a_adv[0], ",\treward:", res.reward, "\tdone", res.done
            # print("s:",s,"a:",a,"adv:",a_adv,"r:",r)

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_a_adv.append(a_adv)
            buffer_r.append((res.reward+8)/8)    # normalize
            buffer_r_adv.append(-(res.reward+8)/8)    # normalize

            if total_step % UPDATE_GLOBAL_ITER == 0 or res.done:   # update global and assign to local net
                if res.done:
                    v_s_ = 0   # terminal
                    v_s_adv = 0   # terminal
                else:
                    v_s_ = SESS.run(AC.v, {AC.s: s_[np.newaxis, :]})[0, 0]
                    v_s_adv = SESS.run(AC_Adv.v, {AC_Adv.s: s_[np.newaxis, :]})[0, 0]

                buffer_v_target, buffer_v_target_adv = [], []

                for r in buffer_r[::-1]:    # reverse buffer r
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                for r in buffer_r_adv[::-1]:    # reverse buffer r
                    v_s_adv = r + GAMMA * v_s_adv
                    buffer_v_target_adv.append(v_s_adv)
                buffer_v_target_adv.reverse()
                
                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                buffer_a_adv, buffer_v_target_adv = np.vstack(buffer_a_adv), np.vstack(buffer_v_target_adv)

                feed_dict = {
                    AC.s: buffer_s,
                    AC.a_his: buffer_a,
                    AC.v_target: buffer_v_target,
                }

                feed_dict_adv = {
                    AC_Adv.s: buffer_s,
                    AC_Adv.a_his: buffer_a_adv,
                    AC_Adv.v_target: buffer_v_target_adv,
                }

                if GLOBAL_EP%40 < 20:
                    AC.update_global(feed_dict)
                    AC.pull_global()
                else:
                    AC_Adv.update_global(feed_dict_adv)
                    AC_Adv.pull_global()
                buffer_s, buffer_a, buffer_r, buffer_a_adv, buffer_r_adv = [], [], [], [], []

            s_ = np.array(res.state_new)
            s = s_
            ep_r += res.reward

            if res.done:
                print("now torque zero")
                res = request_torque_adv(201, 0, 0) # apply zero force to let stop
                GLOBAL_RUNNING_R.append(ep_r)
                GLOBAL_MEAN_R.append(np.mean(GLOBAL_RUNNING_R[-50:]))
                print(
                    "Ep:", GLOBAL_EP,
                    "| Ep_r: %i" % GLOBAL_MEAN_R[-1],
                      )
                GLOBAL_EP += 1
                if GLOBAL_RUNNING_R[-1] > -50 :
                    saver.save(SESS, restore_file ,global_step=GLOBAL_EP)
                    print("save episode:", GLOBAL_EP)
                    MAX_R = GLOBAL_RUNNING_R[-1]
                break