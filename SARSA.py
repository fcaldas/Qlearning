# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:33:57 2016

@author: fcaldas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:25:22 2016

@author: fcaldas
"""
import numpy as np
from numpy import pi, min, max

class SARSAlearner:

    nActions = 20
    actions = np.linspace(0,1,20)*2.-1
    Q = np.zeros([20, 80, 180, 80, nActions])
    bin0 = []
    bin1 = []
    bin2 = []
    bin3 = []
    alpha = 0.2 # learning rate
    gamma = 0.9 # discount rate
    epsilon = 0.05
    epsilon_decay_rate = 0.99 # for each simulation we multiply epsilon by this constant
    lastS = []
    episode = 0
    
    lastS = None
    lastSp = None
    
    def __init__(self):
        
        self.suplim = np.array([0.17, 3, pi/2, pi*3]);
        self.inflim = np.array([-0.17, -3, -pi/2, -pi*3])
        self.bin0 = np.linspace(self.inflim[0], self.suplim[0], num=self.Q.shape[0])
        self.bin1 = np.linspace(self.inflim[1], self.suplim[1], num=self.Q.shape[1])
        self.bin2 = np.linspace(self.inflim[2], self.suplim[2], num=self.Q.shape[2])
        self.bin3 = np.linspace(self.inflim[3], self.suplim[3], num=self.Q.shape[3])
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
        suplim = self.suplim
        inflim = self.inflim
        Xl = np.zeros(4);        
        nInterval = np.array([self.Q.shape[0], self.Q.shape[1], self.Q.shape[2], self.Q.shape[3]])
        dInterval = (suplim - inflim) / nInterval
        Xl[2] = X[2] % 2*pi;
        if(Xl[2] > pi):
            Xl[2] = 2*pi - Xl[2];
        elif(Xl[2] < -pi):
            Xl[2] = 2*pi + Xl[2];
        # clip values out of our mapping function
        Xl[0] = min([suplim[0], X[0]]);
        Xl[1] = min([suplim[1], X[1]]);
        Xl[2] = min([suplim[2], X[2]]);
        Xl[3] = min([suplim[3], X[3]]);
        # find indexes we will be using for this timestep
        idx = [0,0,0,0]
        idx[0] = np.argmax(Xl[0] <= self.bin0 )
        idx[1] = np.argmax(Xl[1] <= self.bin1 )
        idx[2] = np.argmax(Xl[2] <= self.bin2 )
        idx[3] = np.argmax(Xl[3] <= self.bin3 )
        return np.array(idx, dtype=np.int);

    def getAction(self, S, reward):
        lastS = self.lastS
        S = self.getStateIndex(S) 
        # decide on next action using \epsilon-greedy
        if(np.random.rand() < self.epsilon):
             # epsilon policy
            iargmax = np.random.randint(0, high=self.nActions);
        else:
             # use greedy policy instead, solve ties randomly
            iargmax = np.random.choice(np.where(self.Q[S[0], S[1], S[2], S[3], :] == np.max(self.Q[S[0], S[1], S[2], S[3], :]))[0])
  
        # in case this is not the first move do backtracking.
        if(self.lastS is not None):
            lastS = self.lastS
            self.Q[lastS[0], lastS[1], lastS[2], lastS[3], lastS[4]] = \
                        (1 - self.alpha) * self.Q[lastS[0], lastS[1], lastS[2], lastS[3], lastS[4]] +\
                        self.alpha * (reward + 
                                      self.gamma * self.Q[S[0], S[1], S[2], S[3], iargmax]  
                                  #    self.gamma**2 * max(self.Q[S[0], S[1], S[2], S[3], :])
                        );
        
        self.lastS = [S[0], S[1], S[2], S[3], iargmax]
        return self.actions[iargmax];        
    
    def getNIterations():
        print self.episode
    
    def dump(self):
        fd = open("learner_" + str(self.episode).zfill(6) + ".obj", "wb")
        pickle.dump(self, fd)
        fd.flush()
        fd.close()