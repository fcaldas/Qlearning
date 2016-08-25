# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:25:22 2016

@author: fcaldas
"""
import numpy as np
from numpy import pi, min, max
import pickle

class Qlearner:

    nActions = 20
    actions = np.linspace(0,1,nActions)*2-1
    Q = np.random.rand(30, 40, 220, 40, nActions)/10
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.5
    epsilon_decay_rate = 0.999 # for each simulation we multiply epsilon by this constant
    lastS = []
    episode = 0
    
    def __init__(self):
        pass

    def setInitialState(self, state):
        self.lastS = self.getStateIndex(state)
        idx = self.lastS
        self.epsilon *= self.epsilon_decay_rate
        if(np.random.rand() < self.epsilon):
             # epsilon policy
            iargmax = np.random.randint(0, high=self.nActions);
        else:
             # use greedy policy instead
            iargmax = np.argmax(self.Q[idx[0], idx[1], idx[2], idx[3], :]);
        self.lastS = [idx[0], idx[1], idx[2], idx[3], iargmax]
        return self.actions[iargmax]

    def getStateIndex(self, X):
        Xl = np.zeros(4);        
        suplim = np.array([0.17, 2, pi, pi/2]);
        inflim = np.array([-0.17, -2, -pi, -pi/2])
        nInterval = np.array([30, 40, 220, 40])
        dInterval = (suplim - inflim) / nInterval
        Xl[2] = X[2] % 2*pi;
        if(Xl[2] > pi):
            Xl[2] = 2*pi - Xl[2];
        elif(Xl[2] < -pi):
            Xl[2] = 2*pi + Xl[2];
        # clip values out of our mapping function
        Xl[0] = max([inflim[0], min([suplim[0], X[0]])]);
        Xl[1] = max([inflim[1], min([suplim[1], X[1]])]);
        Xl[3] = max([inflim[3], min([suplim[3], X[3]])]);
        # find indexes we will be using for this timestep
        idx = np.floor((Xl - inflim) / dInterval);
        idx[0] = min([idx[0], nInterval[0] - 1])
        idx[1] = min([idx[1], nInterval[1] - 1])
        idx[2] = min([idx[2], nInterval[2] - 1])
        idx[3] = min([idx[3], nInterval[3] - 1])
        return np.array(idx, dtype=np.int);

    def getAction(self, S, reward):
        lastS = self.lastS
        S = self.getStateIndex(S) 
        # decide on next action using \epsilon-greedy
        if(np.random.rand() < self.epsilon):
             # epsilon policy
            iargmax = np.random.randint(0, high=self.nActions);
        else:
             # use greedy policy instead
            iargmax = np.argmax(self.Q[S[0], S[1], S[2], S[3], :]);
  
        # in case this is not the first move do backtracking.
        self.Q[lastS[0], lastS[1], lastS[2], lastS[3], lastS[4]] = \
                    (1 - self.alpha) * self.Q[lastS[0], lastS[1], lastS[2], lastS[3], lastS[4]] +\
                    self.alpha * (reward + self.gamma * (max(self.Q[S[0], S[1], S[2], S[3], :])));

        self.lastS = [S[0], S[1], S[2], S[3], iargmax]
        return self.actions[iargmax];        
    
    def getNIterations():
        print self.episode
    
    def dump(self):
        fd = open("learner_" + str(self.episode).zfill(6) + ".obj", "wb")
        pickle.dump(self, fd)
        fd.flush()
        fd.close()