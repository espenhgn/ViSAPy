#!/usr/bin/env python

import numpy as np
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cyextensions

class NonStationaryPoisson(object):
    '''
    non-stationary Poisson class implementation
    '''
    def __init__(self, N=100,
                tvec=np.arange(1001.),
                lambda_t=np.sin(np.arange(1001.) * np.pi * 2 / 200),
                rate=10.):
        '''
        Class initialization
        
        Keyword arguments:
        ::
            
            N : int, Number of trials
            tvec : np.ndarray, time vector in units of ms
            lambda_t : np.ndarray, rate expectation function,
                normalized and scaled
            rate : float, expectation mean rate of each trial
        '''
        self.N = N
        self.tvec = tvec
        self.lambda_t = lambda_t
        self.rate = rate
        self.dt = np.diff(self.tvec)[0]
        
        #norm lambda_t to [0, 1], multiply by rate scaled by lambda_t mean
        self.lambda_t -= self.lambda_t.min()
        self.lambda_t /= self.lambda_t.max()
        self.lambda_t *= self.rate
        self.lambda_t *= 2
        
        #get list of N times with apprxmt rate rate
        self.poisson = self.n_nonstationary_poisson()
        
        poissontimes = np.array([])
        for x in self.poisson:
            poissontimes = np.r_[poissontimes, x]
        
        print 'non-stationary poisson process average rate: %.3f' % \
                (poissontimes.size / float(N) / self.tvec[-1] * 1E3)
    

    def nonstationary_poisson(self,
                        tvec=np.arange(1001.),
                        dt = 1.,
                        lambda_t=np.sin(np.arange(1001.) * np.pi * 2 / 200),
                        rate=10.):
        '''forward call to cython extension'''
        
        return cyextensions.nonstationary_poisson(tvec=tvec,
                                                dt=dt,
                                                lambda_t=lambda_t.astype(float),
                                                rate=rate)
        
    
    def n_nonstationary_poisson(self):
        '''
        Generate list of n entries where each element is an array of
        spike times on [0, T]
        
        the noise signal is normed to [0, rate*2], and rate is rate*2
        '''
        poisson = []
        for i in xrange(self.N):
            poisson.append(self.nonstationary_poisson(self.tvec, self.dt,
                                        self.lambda_t, self.rate))
        
        return poisson

    def plot_stuff(self):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
        ax1.plot(self.tvec, self.lambda_t)
        ax1.set_xlim(self.tvec[0], self.tvec[-1])
        ax1.set_ylabel('$\lambda(t)$')
        
        i = 0
        for x in self.poisson:
            ax2.plot(x, np.zeros(x.size) + i, 'o',
                     markersize=1,
                     markerfacecolor='k',
                     markeredgecolor='k',
                     alpha=0.25)
            i += 1
        
        ax2.set_xlim(self.tvec[0], self.tvec[-1])
        ax2.set_ylabel('$S(t)$')
        ax2.set_ylim(0, self.N)
        
        
        poissontimes = np.array([])
        for x in self.poisson:
            poissontimes = np.r_[poissontimes, x]
        
        bins = self.tvec
        ax3.hist(poissontimes, bins=bins)
        ax3.set_xlim(self.tvec[0], self.tvec[-1])
        ax3.set_xlabel('time (ms)')
        ax3.set_ylabel('hist($S(t)$)')
        

if __name__ == '__main__':
    
    np.random.seed(12345)
    
    tvec = np.arange(10001) / 10.
    rate = 10.
    N = 1000
    
    #noise = np.random.randn(tvec.size).cumsum() 
    lambda_t = np.sin(2*np.pi*10*tvec * 1E-3)
    
    poisson = NonStationaryPoisson(N=N, tvec=tvec, lambda_t=lambda_t, rate=rate)
        
    poisson.plot_stuff()
    plt.show()
