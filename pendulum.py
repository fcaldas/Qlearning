# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:25:30 2016

@author: fcaldas
"""
import numpy as np
from numpy import sin, cos, pi

class Simulator():
    # pendulum constants    
    M = 0.6242;
    m = 0.1722;
    b = 1.;
    J = 6.0306e-04;
    g = 9.8;
    l = 0.41/2;
    
    # motor parameters
    n = 5.9;
    r = 0.0283/2;
    Rm = 1.4;
    Ke = 12./(2 * pi * n * 1000/(60));
    Kt = 1.5/(n * 15.5);

    X = 0.; # car position
    X_d = 0.; # car speed
    P = 0.; # pendulum angle (in rads 0 = up)
    P_d = 0.; # angular speed
    
    dt = 0.02
    
    def __init__(self, dt = 0.01):
        self.dt = dt

    def restart(self, initial_state):
        self.X = initial_state[0]
        self.X_d = initial_state[1]
        self.P = initial_state[2]
        self.P_d = initial_state[3]
    
    
    def getState(self):    
        return np.array([self.X, self.X_d, self.P, self.P_d])
 
    def step(self, u):
        PWM=u/12;
        # calculate second derivatives of the system
        X_dd = -(self.Rm*sin(self.P)*self.l**3*self.m**2*self.P_d**2*self.r**2 - self.Rm*self.g*cos(self.P)*\
                sin(self.P)*self.l**2*self.m**2*self.r**2 + self.Rm*self.b*self.X_d*self.l**2*self.m\
                *self.r**2-12*self.Kt*PWM*self.n*self.l**2*self.m*self.r + self.Ke*self.Kt*self.X_d*self.l**2\
                *self.m + self.J*self.Rm*sin(self.P)*self.l*self.m*self.P_d**2*self.r**2 + self.J*\
                self.Rm*self.b*self.X_d*self.r**2-12*self.J*self.Kt*PWM*self.n*self.r+\
                self.J*self.Ke*self.Kt*self.X_d)/(self.Rm*self.r**2*(self.J*self.m + self.J*self.M\
                + self.l**2*self.m**2 - self.l**2*self.m**2*cos(self.P)**2 + self.M*self.l**2*self.m));
        
        P_dd = -(self.l*self.m*(self.Ke*self.Kt*self.X_d*cos(self.P) - self.M*self.Rm*self.g*self.r**2\
                *sin(self.P) + self.Rm*self.b*self.r**2*self.X_d*cos(self.P) - self.Rm*self.g*self.m*self.r**2*\
                sin(self.P)-12*self.Kt*PWM*self.n*self.r*cos(self.P)+self.Rm*self.l*self.m*self.P_d**2*self.r**2*\
                cos(self.P)*sin(self.P)))/(self.Rm*self.r**2*(self.J*self.m + self.J*self.M +\
                self.l**2*self.m**2 - self.l**2*self.m**2*cos(self.P)**2 + self.M*self.l**2*self.m));

        self.X_d = self.X_d + X_dd * self.dt
        self.X = self.X + self.X_d * self.dt
        self.P_d = self.P_d + P_dd * self.dt        
        self.P = self.P + self.P_d * self.dt
        
        
    def getReward(self):
        if(np.abs(self.X) <= 0.15):
            if(np.abs((self.P % (2*pi))) <= 12*2*pi/(360)):
                if(np.abs(self.X) > 0.10):
                    return 1
            return 0
        else:
            return -1
    
    
    def stopSim(self):
        if(np.abs(self.X) > 0.17 or np.abs(self.P) > pi/4):
            return True
        return False