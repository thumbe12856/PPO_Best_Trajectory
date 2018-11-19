import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger

from collections import deque

import cv2

import matplotlib.pyplot as plt

import sonic_env
import architecture as policies

# Calculate cross entropy
from baselines.a2c.utils import cat_entropy
from utilities import make_path, find_trainable_variables, discount_with_dones

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner

import pandas as pd
import random
import math

SAVE_FILE_FLAG = False
#GAME = "sonic"
GAME = "mario"
#LEVEL = "GreenHillZone/Act1"
LEVEL = "1-2"
WORKER_NUM = 4
ALG = "bestTrajectory"
scoreByTimestep = []
scoreByTimestep_list = []
timesteps_list = []
distance_list = []

nowReward = [0]
lastReward = [0]

timesDead = 0
timesToGoalCounter = 0
goalDistance = 3250
nowDistance = 0
nowDistances = [0] * WORKER_NUM
nowMaxDistance = 0
realMaxDistance = 0
nowMaxDistanceCounter = 0
lastDistance = -1
lastDistances = [0] * WORKER_NUM
normalizationParameter = 1
frequentDeadDistance = {}
bestTrajectory = [[] for _ in range(40)]
#bestTrajectory = [[], [(41, 176), (43, 176), (48, 176), (54, 176), (62, 176), (72, 176), (84, 176), (96, 176), (108, 176), (120, 176), (132, 176), (144, 153), (156, 139), (168, 149), (180, 161), (192, 153), (204, 159), (216, 176), (228, 176), (240, 176), (252, 153), (264, 134), (276, 118), (288, 106), (300, 97), (312, 95), (324, 107), (336, 127), (344, 128), (351, 128), (354, 128), (354, 132), (354, 150), (354, 170), (354, 176), (354, 157), (354, 145), (354, 143), (354, 152), (354, 171), (354, 142), (354, 131), (354, 141), (354, 161), (354, 162), (354, 149), (354, 146), (354, 154), (354, 172), (354, 144), (354, 140), (354, 147), (354, 164), (354, 130), (354, 121), (355, 118), (354, 125), (354, 133), (354, 126), (354, 134), (354, 166), (356, 115), (359, 112), (363, 112), (369, 112), (376, 93), (383, 80), (390, 76), (397, 81), (404, 95), (411, 112), (417, 112), (424, 120), (431, 128), (437, 128), (443, 109), (450, 96), (457, 92), (464, 96), (471, 77), (478, 64), (485, 60), (492, 65), (499, 79), (506, 101), (513, 121), (520, 141), (527, 144), (537, 144), (548, 144), (560, 152), (572, 169), (584, 176), (596, 176), (608, 176), (620, 176), (632, 176), (644, 176), (656, 153), (668, 134), (680, 118), (691, 109), (690, 118), (690, 138), (691, 158), (693, 153), (696, 152), (699, 163), (703, 176), (708, 176), (715, 176), (723, 176), (733, 176), (745, 176), (757, 176), (769, 176), (781, 176), (793, 176), (805, 176), (817, 176), (829, 176), (841, 176), (853, 176), (865, 176), (877, 176), (887, 176), (897, 153), (906, 134), (915, 118), (924, 108), (934, 110), (942, 112), (947, 112), (950, 98), (955, 83), (961, 79), (968, 85), (975, 102), (978, 112), (978, 93), (978, 81), (978, 79), (978, 88), (978, 107), (977, 81), (975, 79), (976, 88), (977, 107), (978, 98), (978, 83), (978, 85), (978, 102), (978, 78), (978, 69), (978, 70), (978, 82), (978, 80), (978, 76), (977, 100), (977, 112), (974, 112), (973, 112), (974, 93), (974, 78), (973, 69), (974, 70), (974, 82), (976, 112), (975, 112), (972, 98), (970, 85), (968, 82), (965, 90), (962, 108), (958, 112), (956, 112), (956, 93), (956, 78), (957, 68), (959, 67), (960, 77), (962, 97), (964, 112), (963, 112), (963, 93), (964, 81), (967, 79), (971, 88), (976, 107), (977, 93), (977, 102), (978, 72), (977, 72), (976, 83), (976, 103), (978, 66), (977, 61), (970, 87), (968, 107), (966, 112), (967, 112), (969, 93), (971, 78), (973, 66), (975, 64), (978, 97), (976, 93), (976, 78), (975, 69), (973, 70), (972, 82), (969, 102), (965, 112), (959, 112), (952, 112), (943, 94), (936, 74), (930, 61), (925, 69), (920, 88), (917, 108), (914, 112), (910, 120), (906, 139), (903, 159), (900, 176), (899, 176), (901, 176), (905, 176), (910, 157), (916, 142), (919, 138), (919, 150), (919, 170), (921, 176), (924, 176), (928, 176), (929, 157), (930, 142), (931, 144), (931, 159), (932, 176), (931, 176), (932, 157), (934, 145), (937, 143), (940, 152), (945, 171), (951, 162), (957, 146), (963, 141), (969, 152), (975, 171), (980, 176), (986, 176), (993, 176), (999, 176), (1004, 176), (1010, 176), (1017, 176), (1025, 176), (1032, 176), (1039, 176), (1047, 176), (1056, 176), (1067, 176), (1079, 176), (1091, 153), (1103, 139), (1115, 149), (1127, 172), (1139, 176), (1150, 176), (1162, 176), (1174, 158), (1186, 140), (1197, 134), (1206, 141), (1214, 161), (1223, 176), (1230, 176), (1237, 176), (1247, 176), (1258, 176), (1270, 153), (1282, 134), (1294, 122), (1306, 122), (1318, 136), (1330, 156), (1342, 176), (1354, 176), (1366, 176), (1378, 158), (1390, 138), (1402, 126), (1414, 134), (1426, 154), (1438, 174), (1449, 176), (1456, 176), (1463, 176), (1470, 176), (1476, 176), (1484, 176), (1494, 176), (1506, 176), (1518, 176), (1530, 176), (1542, 153), (1554, 134), (1566, 118), (1578, 106), (1590, 101), (1602, 109), (1614, 127), (1626, 147), (1634, 167), (1634, 176), (1634, 157), (1634, 144), (1634, 140), (1634, 147), (1634, 164), (1634, 142), (1634, 130), (1634, 121), (1636, 115), (1638, 114), (1642, 122), (1646, 128), (1651, 109), (1657, 95), (1664, 90), (1671, 93), (1678, 106), (1685, 125), (1692, 145), (1699, 165), (1706, 176), (1713, 176), (1720, 176), (1727, 157), (1730, 142), (1730, 130), (1730, 120), (1730, 113), (1730, 110), (1731, 116), (1730, 151), (1730, 171), (1730, 176), (1730, 157), (1730, 121), (1731, 115), (1733, 112), (1736, 112), (1742, 112), (1749, 112), (1757, 89), (1767, 70), (1776, 54), (1785, 42), (1794, 33), (1805, 31), (1816, 43), (1828, 63), (1839, 83), (1851, 103), (1863, 123), (1875, 143), (1887, 163), (1898, 176), (1905, 176), (1912, 157), (1919, 142), (1926, 130), (1933, 120), (1940, 113), (1947, 111), (1954, 118), (1961, 128), (1970, 105), (1980, 86), (1991, 72), (2003, 70), (2015, 82), (2027, 102), (2039, 122), (2051, 142), (2063, 162), (2075, 176), (2087, 176), (2099, 176), (2111, 158), (2123, 138), (2135, 122), (2147, 108), (2159, 99), (2171, 95), (2183, 103), (2195, 112), (2207, 112), (2219, 117), (2231, 134), (2243, 145), (2253, 149), (2264, 153), (2276, 130), (2288, 114), (2300, 112), (2308, 122), (2309, 145), (2311, 165), (2314, 176), (2314, 157), (2314, 145), (2314, 143), (2317, 152), (2321, 171), (2326, 176), (2332, 157), (2339, 142), (2346, 130), (2353, 126), (2360, 136), (2367, 154), (2374, 165), (2381, 155), (2388, 153), (2395, 161), (2402, 176), (2410, 176), (2420, 153), (2431, 137), (2443, 135), (2455, 145), (2467, 168), (2479, 188), (2491, 208), (2503, 228), (2515, 248), (2527, 12), (2539, 32), (2551, 52), (2563, 72), (2575, 92), (2587, 112), (2599, 132), (2611, 152), (2623, 172), (2635, 192), (2646, 212), (2646, 232), (2646, 252), (2646, 16), (2646, 36), (2646, 56), (2646, 76), (2646, 96), (2646, 116), (2646, 136), (2646, 156), (2646, 176), (2646, 196), (2646, 216), (2646, 236), (2646, 0), (2646, 20), (2646, 40), (2646, 60), (2646, 80), (2646, 100), (2646, 120), (2646, 140), (2646, 160), (2646, 180), (2646, 200), (2646, 220), (2646, 240), (2646, 4), (2646, 24), (2646, 44), (2646, 64), (2646, 84), (2646, 104), (2646, 124), (2646, 144), (2646, 164), (2646, 184), (2646, 204), (2646, 224), (2646, 244), (2646, 3)], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
tempTrajectory = [[] for _ in range(WORKER_NUM)]
spawn_from = 0
spawn_from_switch = False

minClip = -0.1
maxClip = 1
nowY = 0

EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200000  # e-greedy threshold decay
EPS_threshold = 1
EPS_step = 0

# find the closest distance between the point and the best trojectory
def closestDistance(x, y, spawn_from):
    global bestTrajectory
    minDistance = 999999999
    minAxis = (0, 0)

    if(x > bestTrajectory[spawn_from][-1][0]):
        return 0

    for b in bestTrajectory[spawn_from]:
        c = (x - b[0]) ** 2 + (y - b[1]) ** 2
        if c < minDistance:
            minDistance = c
            minAxis = b

    if minDistance == 999999999:
        minDistance = 0

    #print("nowX, nowY = ({0} {1})".format(x, y))
    #print("min axis = {0}".format(minAxis))
    punishment = math.sqrt(minDistance) / 150.0
    if(punishment <= 0.1):
        punishment = 0
    return punishment

def bestTrajectoryAlg(infos, rewards, dones, values):
    global EPS_step, EPS_threshold, EPS_END, EPS_START, spawn_from, spawn_from_switch
    global nowDistance, lastDistance, nowMaxDistance, realMaxDistance, nowDistances, lastDistances
    global nowMaxDistanceCounter, normalizationParameter, timesToGoalCounter, timesDead
    global nowReward, lastReward, scoreByTimestep

    temp_mb_rewards = []
    lastDistances = nowDistances
    nowDistances = [infos[idx]['curr_page'] * 256 + infos[idx]['x'] for idx, _ in enumerate(infos)]
    lastReward = nowReward
    nowReward = rewards
    
    for infosIdx, _ in enumerate(infos):
        EPS_step = EPS_step + 1
        info = infos[infosIdx]
        nowDistance = nowDistances[infosIdx]
        lastDistance = lastDistances[infosIdx]
        nowY = info['y']
        
        # record coordinate
        if((nowDistance, nowY) not in tempTrajectory[infosIdx]):
            tempTrajectory[infosIdx].append((nowDistance, nowY))

        # Is is because if the agent jump too high will cause the overflow of y.
        # Like (100, 20) -> (100, 2) -> (100, 234)
        if(tempTrajectory[infosIdx] and nowY - tempTrajectory[infosIdx][-1][1] >= 180):
            tempTrajectory[infosIdx].pop()

        e = nowDistance - lastDistance
        #e = nowReward[infosIdx] - lastReward[infosIdx]
        if(e > normalizationParameter):
            normalizationParameter = e

        # normalization
        e_tilde = (2 / float(2 * normalizationParameter)) * (e + normalizationParameter) - 1
        #e_tilde = (2 / float(2 * normalizationParameter)) * (rewards[infosIdx] + normalizationParameter) - 1
        if(e_tilde < minClip):
            e_tilde = minClip
        elif(e_tilde > maxClip):
            e_tilde = maxClip

        reward = e_tilde
        
        # terminal
        if(dones[infosIdx]):
            spawn_from_switch = True
            print("value_[0]: " + str(values))
            frequentDeadDistance.setdefault(nowDistance / 100 * 100, 0)
            frequentDeadDistance[nowDistance / 100 * 100] = frequentDeadDistance[nowDistance / 100 * 100] + 1
            
            global distance_list, GAME, LEVEL, ALG, SAVE_FILE_FLAG
            dis_dir = "./model/" + GAME + "/" + LEVEL + "/scratch/action_repeat_4/30/" + ALG +"/" + str(infosIdx) + ".csv"
            distance_list.append(nowDistance)
            df = pd.DataFrame([], columns=["distance"])
            df["distance"] = distance_list
            if(SAVE_FILE_FLAG):
                df.to_csv(dis_dir, index=False)
            print("distance mean:")
            print('++++++++++++++++++++++++++')
            print(df["distance"].mean())
            print(frequentDeadDistance[nowDistance / 100 * 100])
            print('++++++++++++++++++++++++++')
            print("normalizationParameter: {0}".format(normalizationParameter))
            print("nowDistance:{0}".format(nowDistance))
            print("nowMaxDistance: {0}".format(nowMaxDistance))
            print("realMaxDistance: {0}".format(realMaxDistance))
            print("timesToGoalCounter: {0}".format(timesToGoalCounter))
            print("timesDead: {0}".format(timesDead))
            print("EPS_threshold: {0}".format(EPS_threshold))
            print("EPS_step: {0}".format(EPS_step))
            print("Best trojectory length: {0}".format(len(bestTrajectory[spawn_from])))
            print("")

            if(lastDistance >= goalDistance and lastDistance <= goalDistance + 100):
                timesToGoalCounter = timesToGoalCounter + 1

            timesDead = timesDead + 1
            if(EPS_threshold <= 0.2 and nowMaxDistance > 0):
            #if(EPS_step >= EPS_DECAY and nowMaxDistance > 0):
                if(nowMaxDistanceCounter < 2):
                    if(nowDistance / 100 <= nowMaxDistance / 100 - 2):
                        nowMaxDistance = nowMaxDistance - 100
                        nowMaxDistanceCounter = nowMaxDistanceCounter + 1
            
            tempTrajectory[infosIdx] = []
            nowDistances[infosIdx] = 0
            lastDistances[infosIdx] = 0
        else:

            if(nowDistance / 100 > nowMaxDistance / 100):
                nowMaxDistance = nowDistance
                
                if(EPS_step <= EPS_DECAY * 10 or timesToGoalCounter < 10):
                    bestTrajectory[spawn_from] = tempTrajectory[infosIdx]

                if(EPS_step >= EPS_DECAY * 3):
                    reward = reward + 1
                    nowMaxDistanceCounter = 0
                    #rollout.bonuses = [b + 1 for b in rollout.bonuses]
                    #mb_rewards = [mr + 1 for mr in mb_rewards]
            else:
                #if(EPS_step >= EPS_DECAY * 5 and timesToGoalCounter > 0):
                if(EPS_step >= EPS_DECAY * 5):
                    # punishment
                    punishment = closestDistance(nowDistance, nowY, spawn_from)
                    reward = reward - punishment
                    if(reward < minClip):
                        reward = minClip
                        #print(closestDistance(nowDistance, nowY, spawn_from) / normalizationParameter)
                        
        # if agent breaks the record of realMaxDistance
        if(nowDistance > realMaxDistance):
            realMaxDistance = nowDistance

        temp_mb_rewards.append(reward)
    #mb_rewards.append(temp_mb_rewards)
    return temp_mb_rewards

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self,
                 policy,
                ob_space,
                action_space,
                nenvs,
                nsteps,
                ent_coef,
                vf_coef,
                max_grad_norm):

        sess = tf.get_default_session()


        # CREATE THE PLACEHOLDERS
        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.float32, name="learning_rate_")
        # Keep track of old actor
        oldneglopac_ = tf.placeholder(tf.float32, [None], name="oldneglopac_")
        # Keep track of old critic 
        oldvpred_ = tf.placeholder(tf.float32, [None], name="oldvpred_")
        # Cliprange
        cliprange_ = tf.placeholder(tf.float32, [])


        # CREATE OUR TWO MODELS
        # Step_model that is used for sampling
        step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)
        
        # Test model for testing our agent
        #test_model = policy(sess, ob_space, action_space, 1, 1, reuse=False)

        # Train model for training
        train_model = policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)



        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
       
        # Clip the value
        # Get the value predicted
        value_prediction = train_model.vf

        # Clip the value = Oldvalue + clip(value - oldvalue, min = - cliprange, max = cliprange)
        value_prediction_clipped = oldvpred_ + tf.clip_by_value(train_model.vf - oldvpred_,  - cliprange_, cliprange_)

        # Unclipped value
        value_loss_unclipped = tf.square(value_prediction - rewards_)

        # Clipped value
        value_loss_clipped = tf.square(value_prediction_clipped - rewards_)

        # Value loss 0.5 * SUM [max(unclipped, clipped)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped,value_loss_clipped ))


        # Clip the policy
        # Output -log(pi) (new -log(pi))
        #print(train_model.pi)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)
        
        # Remember we want ratio (pi current policy / pi old policy)
        # But neglopac returns us -log(policy)
        # So we want to transform it into ratio
        # e^(-log old - (-log new)) == e^(log new - log old) == e^(log(new / old)) 
        # = new/old (since exponential function cancels log)
        # Wish we can use latex in comments
        ratio = tf.exp(oldneglopac_ - neglogpac) # ratio = pi new / pi old

        # Remember also that we're doing gradient ascent, aka we want to MAXIMIZE the objective function which is equivalent to say
        # Loss = - J
        # To make objective function negative we can put a negation on the multiplication (pi new / pi old) * - Advantages
        pg_loss_unclipped = -advantages_ * ratio 

        # value, min [1 - e] , max [1 + e]
        pg_loss_clipped = -advantages_ * tf.clip_by_value(ratio, 1.0 - cliprange_, 1.0 + cliprange_)

        # Final PG loss
        # Why maximum, because pg_loss_unclipped and pg_loss_clipped are negative, getting the min of positive elements = getting
        # the max of negative elements
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        #entropy = tf.reduce_mean(train_model.pd.entropy())
        entropy = tf.reduce_mean(train_model.entropy())

        # Total loss (Remember that L = - J because it's the same thing than max J
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef


        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = find_trainable_variables("model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Build our trainer
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, epsilon=1e-5)

        # 4. Backpropagation
        _train = trainer.apply_gradients(grads)


        # Train function
        def train(states_in, actions, returns, values, neglogpacs, last_features, lr, cliprange):
            
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # Returns = R + yV(s')
            advantages = returns - values

            # Normalize the advantages (taken from aborghi implementation)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # We create the feed dictionary
            td_map = {train_model.inputs_: states_in,
                     train_model.c_in: last_features[0],
                     train_model.h_in: last_features[1],
                     actions_: actions,
                     advantages_: advantages, # Use to calculate our policy loss
                     rewards_: returns, # Use as a bootstrap for real value
                     lr_: lr,
                     cliprange_: cliprange,
                     oldneglopac_: neglogpacs,
                     oldvpred_: values,
                     }

            policy_loss, value_loss, policy_entropy, _= sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            
            return policy_loss, value_loss, policy_entropy


        def save(save_path):
            """
            Save the model
            """
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            """
            Load the model
            """
            saver = tf.train.Saver()
            print('Loading ' + load_path)
            saver.restore(sess, load_path)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.get_initial_features
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env = env, model = model, nsteps = nsteps)

        # Discount rate
        self.gamma = gamma

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam

        # Total timesteps taken
        self.total_timesteps = total_timesteps

        # Reset lstm memory
        self.last_features = self.model.initial_state()

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_rewards, mb_values, mb_neglopacs, mb_dones, mb_last_features = [],[],[],[],[],[],[]

        global EPS_step, EPS_threshold, EPS_END, EPS_START, spawn_from
        global nowDistance, lastDistance, nowMaxDistance, realMaxDistance, nowDistances, lastDistances
        global nowMaxDistanceCounter, normalizationParameter, timesToGoalCounter, timesDead
        global nowReward, lastReward, scoreByTimestep


        # For n in range number of steps
        for n in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because AbstractEnvRunner run self.obs[:] = env.reset()

            #if(n % 2 == 0):
            #--actions, values, neglopacs = self.model.step(self.obs, self.dones) 
            fetched = self.model.step(self.obs, self.last_features[0], self.last_features[1])
            actions, values, neglopacs, features = fetched[0], fetched[1], fetched[2], fetched[3:]

            # Append the observations into the mb
            mb_obs.append(np.copy(self.obs)) #obs len nenvs (1 step per env)

            # Append the actions taken into the mb
            mb_actions.append(actions)

            # Append the values calculated into the mb
            mb_values.append(values)

            # Append the negative log probability into the mb
            mb_neglopacs.append(neglopacs)

            # Append the dones situations into the mb
            mb_dones.append(self.dones)

            # Append the last_features into the mb
            mb_last_features.append(self.last_features)
            self.last_features = features

            """
            if(value_[0] <= 0.8):
                EPS_threshold = 0.2
            else:
                #EPS_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * EPS_step / EPS_DECAY)
                EPS_threshold = 0
            """
            #EPS_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * EPS_step / EPS_DECAY)
            EPS_threshold = 0
            sample = random.random()

            #for idx, _ in enumerate(values):
                #if(values[idx] < 0.2):
                    #actions[idx] = random.randint(0, 12)

            if(sample < EPS_threshold):
                #randomAction = np.zeros(14)
                #randomAction[random.randint(0, 13)] = 1
                #actions = [random.randint(0, 6)]
                actions = [random.randint(0, 6) for _ in range(len(actions))]

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            # {'level_end_bonus': 0, 'rings': 0, 'score': 0, 'zone': 1, 'act': 0, 'screen_x_end': 6591, 'screen_y': 12, 'lives': 3, 'x': 96, 'y': 108, 'screen_x': 0}
            
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            self.env.render()
            spawn_from = infos[0]['levelHi'] * 4 + infos[0]['levelLo']
            
            rewards = bestTrajectoryAlg(infos, rewards, self.dones, values)
            mb_rewards.append(rewards)
            
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglopacs = np.asarray(mb_neglopacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_last_features = np.asarray(mb_last_features, dtype=np.float32)
        #--last_values = self.model.value(self.obs)
        #value(self, ob, c, h):
        last_values = self.model.value(self.obs, self.last_features[0], self.last_features[1])
          

        ### GENERALIZED ADVANTAGE ESTIMATION
        # discount/bootstrap off value fn
        # We create mb_returns and mb_advantages
        # mb_returns will contain Advantage + value
        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)

        lastgaelam = 0

        # From last step to first step
        for t in reversed(range(self.nsteps)):
            # If t == before last step
            if t == self.nsteps - 1:
                # If a state is done, nextnonterminal = 0
                # In fact nextnonterminal allows us to do that logic

                #if done (so nextnonterminal = 0):
                #    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0) 
                # else (not done)
                    #delta = R + gamma * V(st+1)
                nextnonterminal = 1.0 - self.dones
                
                # V(t+1)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                
                nextvalues = mb_values[t+1]

            # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
            mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        # Returns
        mb_returns = mb_advantages + mb_values

        #return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_neglopacs))
        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_neglopacs, mb_last_features))


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(policy,
            env,
            nsteps,
            total_timesteps,
            gamma,
            lam,
            vf_coef,
            ent_coef,
            lr,
            cliprange,
            max_grad_norm,
            log_interval):

    noptepochs = 4
    nminibatches = 8

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    batch_size = nenvs * nsteps # For instance if we take 5 steps and we have 5 environments batch_size = 25

    batch_train_size = batch_size // nminibatches

    assert batch_size % nminibatches == 0

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy,
                ob_space=ob_space,
                action_space=ac_space,
                nenvs=nenvs,
                nsteps=nsteps,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm)

    # Load the model
    # If you want to continue training
    # load_path = "./models/40/model.ckpt"
    # model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//batch_size+1

    for update in range(1, nupdates+1):
        # Start timer
        tstart = time.time()

        frac = 1.0 - (update - 1.0) / nupdates

        # Calculate the learning rate
        lrnow = lr(frac)

        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        global EPS_step
        print("Update: {0}, log interval:{1}, EPS_step:{2}".format(update, log_interval, EPS_step))
        # Get minibatch
        obs, actions, returns, values, neglogpacs, last_features = runner.run()

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mb_losses = []
        total_batches_train = 0

        # Index of each element of batch_size
        # Create the indices array
        indices = np.arange(batch_size)

        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(indices)

            # 0 to batch_size with batch_train_size step
            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values, neglogpacs, last_features))
                mb_losses.append(model.train(*slices, lrnow, cliprangenow))
            
            
        # Feedforward --> get losses --> update
        lossvalues = np.mean(mb_losses, axis=0)

        # End timer
        tnow = time.time()

        # Calculate the fps (frame per second)
        fps = int(batch_size / (tnow - tstart))

        if update % log_interval == 0:
            """
            Computes fraction of variance that ypred explains about y.
            Returns 1 - Var[y-ypred] / Var[y]
            interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
            """
            timesteps = update * batch_size
            ev = explained_variance(values, returns)
            logger.record_tabular("serial_timesteps", update*nsteps)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))
            
            #savepath = "./models/" + str(update) + "/model.ckpt"
            global GAME, LEVEL, ALG, SAVE_FILE_FLAG
            savepath = "./model/" + GAME + "/" + LEVEL + "/scratch/action_repeat_4/30/" + ALG + "/"
            if(SAVE_FILE_FLAG):
                model.save(savepath + str(timesteps) + "/model.ckpt")
                print('Saving to', savepath)

            # Test our agent with 3 trials and mean the score
            # This will be useful to see if our agent is improving
            test_score = testing(model)

            logger.record_tabular("Mean score test level", test_score)
            logger.dump_tabular()


            if(SAVE_FILE_FLAG):
                global bestTrajectory
                with open(savepath + "best_trajectory.txt", "w") as file:
                    file.write(str(bestTrajectory))

            global scoreByTimestep_list, timesteps_list
            scoreByTimestep_list.append(test_score)
            timesteps_list.append(timesteps)
            df = pd.DataFrame([], columns=["score", "timesteps"])
            df["score"] = scoreByTimestep_list
            df["timesteps"] = timesteps_list

            if(SAVE_FILE_FLAG):
                df.to_csv(savepath + "score.csv", index=False)
            
            
    env.close()

# Avoid error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def testing(model):
    """
    We'll use this function to calculate the score on test levels for each saved model,
    to generate the video version
    to generate the map version
    """

    test_env = DummyVecEnv([sonic_env.make_test])

    # Get state_space and action_space
    ob_space = test_env.observation_space
    ac_space = test_env.action_space
 
    # Play
    total_score = 0
    trial = 0

    # We make 3 trials
    for trial in range(3):
        obs = test_env.reset()
        done = False
        score = 0

        while done == False:
            # Get the action
            action, value, _ = model.step(obs)
            
            # Take action in env and look the results
            obs, reward, done, info = test_env.step(action)

            score += reward[0]
        total_score += score
        trial += 1
    test_env.close()

    # Divide the score by the number of trials
    total_test_score = total_score / trial
    return total_test_score
    

def generate_output(policy, test_env):
    """
    We'll use this function to calculate the score on test levels for each saved model,
    to generate the video version
    to generate the map version
    """

    # Get state_space and action_space
    ob_space = test_env.observation_space
    ac_space = test_env.action_space

    test_score = []

    # Instantiate the model object (that creates step_model and train_model)
    #models_indexes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    models_indexes = [624000]

    # Instantiate the model object (that creates step_model and train_model)
    validation_model = Model(policy=policy,
                ob_space=ob_space,
                action_space=ac_space,
                nenvs=1,
                nsteps=1,
                ent_coef=0,
                vf_coef=0,
                max_grad_norm=0)

    for model_index in models_indexes:
        # Load the model
        load_path = "./model/mario/1-2/scratch/action_repeat_4/30/bestTrajectory/"+ str(model_index) + "/model.ckpt"
        #load_path = "./models/"+ str(model_index) + "/model.ckpt"
        validation_model.load(load_path)

        # Play
        score = 0
        timesteps = 0

        # Play during 5000 timesteps
        obs = test_env.reset()
        total_trial = 1
        trials = 0
        #while timesteps < 50000:
        while trials < total_trial:
            timesteps +=1
            
            # Get the actions
            #actions, values, neglopacs = self.model.step(self.obs, self.dones)

            actions, values, _ = validation_model.step(obs)
            
            # Take actions in envs and look the results
            obs, rewards, dones, infos = test_env.step(actions)
            test_env.render()
            score += rewards

            """
            global spawn_from
            spawn_from = infos[0]['levelHi'] * 4 + infos[0]['levelLo']

            global nowDistance, nowY, lastDistance
            lastDistance = nowDistance
            nowDistance = infos[0]['curr_page'] * 256 + infos[0]['x']
            nowY = infos[0]['y']
            rewards[0] = (nowDistance - lastDistance) / 40
            print("ori rewards: {0}".format(rewards))

            punishment = closestDistance(nowDistance, nowY, spawn_from)
            rewards[0] -= punishment

            
            print("punishment: {0}".format(punishment))
            print("rewards: {0}".format(rewards))
            print("score: {0}".format(score))
            print()
            input()
            if(dones):
                if(infos[0]['x'] > 7000):
                    input()
                score = 0
            """
            if(dones):
                trials += 1

       
        # Divide the score by the number of testing environment
        #total_score = score / test_env.num_envs
        total_score = score * 100.0 / total_trial
        test_score.append(total_score)

        print("model {0}:",format(model_index))
        print(score)
        print("")
    
    test_env.close()

    return test_score


def play(policy, env, update):

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy,
                ob_space=ob_space,
                action_space=ac_space,
                nenvs=1,
                nsteps=1,
                ent_coef=0,
                vf_coef=0,
                max_grad_norm=0)
    
    # Load the model
    load_path = "./models/"+ str(update) + "/model.ckpt"
    print(load_path)

    obs = env.reset()

    # Play
    score = 0
    done = False

    while True:
        # Get the action
        actions, values, _ = model.step(obs)
        
        # Take actions in env and look the results
        obs, rewards, done, info = env.step(actions)
        
        score += rewards
    
        env.render()
        
    print("Score ", score)
    env.close()
    



