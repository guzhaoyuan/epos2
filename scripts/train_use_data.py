#!/usr/bin/env python

from epos2.srv import *
import rospy
import sys
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime

GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_NET_SCOPE = 'Global_Net'
MAX_GLOBAL_EP = 300
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []

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
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def request_init():
    # print("wait for Service")
    rospy.wait_for_service('applyTorque')
    try:
        # print("now request service")
        applyTorque = rospy.ServiceProxy('applyTorque', Torque)
        res = applyTorque(0, 0, 1)
        return np.array(res.state_new)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [position torque]"%sys.argv[0]

def myhook():
    request_torque(0, 0, init=0) # set step to zero, torque to zero
    print("shutdown time!")

class Env(object):
    def __init__(self, env_name):
        self.name = env_name
        self.action_space = 1
        self.observation_space = 3

    def random_action(self):
        return np.random.rand(self.action_space)

def shuffle():
    pass

def showoff(global_agent):
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

            a = LOCAL_AC.choose_action(s)
            res = request_torque(step, a)
            print "state:", s, ",action:", a[0], ",\treward:", res.reward
            # res = request_torque(step, env.random_action()[0]*4-2)
            # res = request_torque(step, 0)
            s_ = np.array(res.state_new)
            s = s_
            ep_r += res.reward

            if res.done:
                res = request_torque(step, 0)
                GLOBAL_RUNNING_R.append(ep_r)
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


if __name__ == "__main__":
    rospy.on_shutdown(myhook)

    SESS = tf.Session()

    env = Env('Pendulum-v0')
    N_S = env.observation_space
    N_A = env.action_space
    A_BOUND = [-2., 2.]

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        LOCAL_AC = ACNet('W_0', GLOBAL_AC)

    SESS.run(tf.global_variables_initializer())

    if len(sys.argv) == 2:
        torque = float(sys.argv[1])
        # print(type(torque))
        res = request_torque(0, torque)
        print("position_new:", res.position_new, "\tvelocity:", res.velocity, "\treward:", res.reward)#, "current:", res.current, 
    else:
        pickle_file = 'data/dataset-09-05-07:15.pkl'
        with open(pickle_file,'rb') as f:
            unpickled = []
            while True:
                try:
                    unpickled.append(pickle.load(f))
                except EOFError:
                    break

        num_data = len(unpickled)
        LOCAL_AC.pull_global()
        for i in range(num_data):
            buffer_s = unpickled[i]['s']
            buffer_a = unpickled[i]['a']
            buffer_v_target = unpickled[i]['v_target']
            feed_dict = {
                        LOCAL_AC.s: buffer_s,
                        LOCAL_AC.a_his: buffer_a,
                        LOCAL_AC.v_target: buffer_v_target,
                    }
            print(i)
            LOCAL_AC.update_global(feed_dict)
            LOCAL_AC.pull_global()

        # showoff(GLOBAL_AC)
    
        pickle_file = 'data/dataset-'+datetime.now().strftime('%m-%d-%H:%M')+'.pkl'
        list_dict = []
        with open(pickle_file, 'wb') as f:
            for episode in range(MAX_GLOBAL_EP):
                buffer_s, buffer_a, buffer_r = [], [], []
                step = 0
                ep_r = 0
                s = request_init()
                while(True):
                    # init the state by call env.reset(), getting the init state from the service
                    # calculate the next move
                    # call step service
                    step += 1

                    a = LOCAL_AC.choose_action(s)
                    res = request_torque(step, a)
                    print "state:", s, ",action:", a[0], ",\treward:", res.reward
                    # res = request_torque(step, env.random_action()[0]*4-2)
                    # res = request_torque(step, 0)
                    s_ = np.array(res.state_new)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append((res.reward+8)/8)    # normalize
                    # print "position_new:", res.position_new, "velocity:", res.velocity, "reward:", res.reward#, "current:", res.current, 
                    s = s_
                    ep_r += res.reward

                    if step % UPDATE_GLOBAL_ITER == 0 or res.done:   # update global and assign to local net
                        if res.done:
                            v_s_ = 0   # terminal
                        else:
                            v_s_ = SESS.run(LOCAL_AC.v, {LOCAL_AC.s: s_[np.newaxis, :]})[0, 0]
                        buffer_v_target = []
                        for r in buffer_r[::-1]:    # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                        dictionary = {'s': buffer_s, 'a': buffer_a, 'v_target': buffer_v_target, 'done':res.done}
                        if res.done:
                            dictionary['r'] = ep_r
                        # list_dict.append(dictionary)
                        
                        pickle.dump(dictionary, f)
                        
                        feed_dict = {
                            LOCAL_AC.s: buffer_s,
                            LOCAL_AC.a_his: buffer_a,
                            LOCAL_AC.v_target: buffer_v_target,
                        }
                        LOCAL_AC.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        LOCAL_AC.pull_global()

                    if res.done:
                        # make the pendulum stop faster
                        # if(s[2]<-15):
                        #     res = request_torque(step+1, 0.2)
                        # elif(s[2]>15):
                        #     res = request_torque(step+1,-0.2)
                        # else:
                        res = request_torque(step+1, 0)
                        GLOBAL_RUNNING_R.append(ep_r)
                        print("done episode:", episode, ",reward:", ep_r)
                        break