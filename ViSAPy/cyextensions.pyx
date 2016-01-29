#!/usr/bin/env python

cimport numpy as np
import numpy as np

cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEC = np.complex128
ctypedef np.complex128_t DTYPE_c

@cython.boundscheck(False)
cpdef nonstationary_poisson(
        np.ndarray[DTYPE_t, ndim=1, negative_indices=True] tvec,
        double dt,
        np.ndarray[DTYPE_t, ndim=1, negative_indices=False] lambda_t,
        double rate):
    '''Generate random times on [0, T] with approximate rate rate,
    as a non-stationary poisson process, where lambda_t is a vector
    containing noise at tsteps defined in tvec.
    
    lambda_t <= rate at any time, i.e on [0, rate]
    
    tvec : time vector (ms)
    dt : time resolution (ms)
    noise : rate as function of tvec
    rate : approximate average rate of output (Hz)
    
    The algorithm is implemented as follows:
    
    (0) T = tvec[-1]
    (1) t = 0, I = 0.
    (2) Generate a random number U on U(0, 1).
    (3) t = t - 1/lambda lnU. If t > T then stop.
    (4) Generate a random number U on U(0, 1).
    (5) If U <= lambda(t)/lambda, set I = I + 1, S(I) = t.
    (6) Go to step 2.
    '''    
    #set some props
    cdef double T
    cdef int I
    cdef double t, rate_inv0, rate_inv1
    cdef list S
    
    T = tvec[-1]
    S = []
    rate_inv0 = 5E2 / rate #speed up
    rate_inv1 = 0.5 / rate #speed up
    t = np.random.exponential() * rate_inv0
    while t <= T:
        I = int(t / dt)
        if np.random.rand() <= lambda_t[I] * rate_inv1:
            S.append(I * dt)
 
        t = t + np.random.exponential() * rate_inv0
    
    return np.array(S)



@cython.boundscheck(False)
def ouProcess(
    double T=1.,
    double dt=5E-5,
    double X0=0.,
    double m=0.,
    double sigma=1.,
    int nX=1,
    **kwargs):
    '''
    Mean-reverting Ornstein-Uhlenbeck process:
    SDE:      dX = (m-X) dt + sigma db
    Mean of solution:  X0 exp(-t) + m(1 - exp(-t)).
    
    John Kerl
    kerl at math dot arizona dot edu
    2008-05-12
    '''
    
    cdef double t = 0.
    cdef int ntsteps = round(T / dt + 1)
    cdef np.ndarray[DTYPE_t,
                    ndim=1,
                    negative_indices=False] X = np.random.normal(X0, sigma, nX)
    cdef double sqrtdt = np.sqrt(dt)
    cdef np.ndarray[DTYPE_t,
                    ndim=2,
                    negative_indices=False] Xarray = np.empty((ntsteps, nX))
    
    cdef int i
    cdef int k
    cdef np.ndarray[DTYPE_t,
                    ndim=2,
                    negative_indices=False] dB = np.random.normal(0, sqrtdt,
                                                                  size=(ntsteps,
                                                                        nX))
    for i in range(ntsteps):
        for k in range(nX):
            X[k] += (m - X[k]) * dt + sigma * dB[i, k]
        
        Xarray[i, ] = X

    return Xarray.T


@cython.boundscheck(False)
def ouProcessCorrelated(
    double T=1.,
    double dt=5E-5,
    double X0=0.,
    double m=0.,
    double sigma=1.,
    np.ndarray[DTYPE_t, ndim=2, negative_indices=False] corrcoef = np.ones(1),
    **kwargs
    ):
    '''
    Mean-reverting Ornstein-Uhlenbeck process:
    SDE:      dX = (m-X) dt + sigma db
    Mean of solution:  X0 exp(-t) + m(1 - exp(-t)).
    
    John Kerl
    kerl at math dot arizona dot edu
    2008-05-12
    '''
    if corrcoef.shape[0] != corrcoef.shape[1]:
        raise ValueError, 'corrcoef must be a square matrix!'
    cdef int nX = corrcoef.shape[0]
    cdef int nY = corrcoef.shape[1]
    cdef np.ndarray[DTYPE_t,
                    ndim=2,
                    negative_indices=False] chol = np.linalg.cholesky(corrcoef)
    cdef double t = 0.
    cdef int ntsteps = round(T / dt + 1)
    cdef np.ndarray[DTYPE_t,
                    ndim=1,
                    negative_indices=False] X = np.random.normal(X0, sigma, nX)
    cdef double sqrtdt = np.sqrt(dt)
    cdef np.ndarray[DTYPE_t,
                    ndim=2,
                    negative_indices=False] Xarray = np.empty((ntsteps, nX))
    
    cdef int i
    cdef int k
    cdef np.ndarray[DTYPE_t,
                    ndim=2,
                    negative_indices=False] dB = np.random.normal(0, sqrtdt,
                                                                  size=(ntsteps,
                                                                        nX))
    for i in range(ntsteps):
        for k in range(nX):
            X[k] += (m - X[k]) * dt + sigma * dB[i, k]
            
        Xarray[i, ] = X

    #seems more logical to impose correlations after signals are generated
    Xarray = np.dot(Xarray, chol.T)
    return Xarray.T
    

@cython.boundscheck(False)
def anypowerNoise(double T=1.,
                double dt=5E-5,
                double alpha=-1.,
                double sigma=1.,
                int nX = 1,
                **kwargs
            ):
    '''
    Generate nX rows of noise with any (negative) power alpha
    frequency dependency.
    
    The method works by filtering white gaussian noise in Fourier space.
    
    The returned argument have standard deviation sigma.
    '''    
    cdef int ntsteps = round(T / dt + 1)
    cdef int outputntsteps = ntsteps
    cdef double srate = 1./dt

    #setting ntsteps to nearest power of two upwards, i.e 16001 -> 16384, faster
    cdef int i
    i = 0
    cdef np.ndarray[long, ndim=1, negative_indices=False] power2s = 2**np.arange(33)
    
    ntsteps = power2s[np.where(ntsteps <= power2s)[0][0]]
        
    cdef np.ndarray[DTYPE_t,
                    ndim=1,
                    negative_indices=False] freqz = np.fft.fftfreq(int(ntsteps),
                                                                   1./srate)
    cdef np.ndarray[DTYPE_c, ndim=2, negative_indices=False] X

    cdef np.ndarray[DTYPE_t, ndim=2,
                    negative_indices=False] x = np.random.normal(0, sigma,
                                                                 size=(ntsteps,
                                                                       nX)).T

    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] r
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] theta
    
    X = np.fft.fft(x)
    
    r = (X.real**2 + X.imag**2)**0.5
    theta = np.arctan2(X.imag, X.real)
    
    r[:, 1:] = r[:, 1:] * (abs(freqz[1:])**alpha)**0.5
    
    X.real = r * np.cos(theta)
    X.imag = r * np.sin(theta)
    
    x = np.fft.ifft(X).real[:, :outputntsteps]
    
    return x / x.std() * sigma



@cython.boundscheck(False)
def anypowerCorrelatedNoise(
                        double T=1.,
                        double dt=5E-5,
                        double alpha=-1.,
                        double sigma=1.,
                        np.ndarray[DTYPE_t, ndim=2,
                                   negative_indices=False] corrcoef = np.array([[1.]]),
                        **kwargs
                       ):
    '''Generate correlated noise with any (negative) power alpha
    frequency dependency.
    
    The method works by filtering white gaussian noise in Fourier space.
    
    The returned argument have standard deviation sigma.
    '''    
    if corrcoef.shape[0] != corrcoef.shape[1]:
        raise ValueError, 'corrcoef must be a square matrix!'
    cdef int nX = corrcoef.shape[0]
    cdef int nY = corrcoef.shape[1]
    cdef int ntsteps = round(T / dt + 1)
    cdef int outputntsteps = ntsteps
    cdef double srate = 1./dt

    #setting ntsteps to nearest power of two upwards, i.e 16001 -> 16384, faster
    cdef int i
    i = 0
    cdef np.ndarray[long, ndim=1,
                    negative_indices=False] power2s = 2**np.arange(33)
    
    ntsteps = power2s[np.where(ntsteps <= power2s)[0][0]]
        
    cdef np.ndarray[DTYPE_t, ndim=1,
                    negative_indices=False] freqz = np.fft.fftfreq(int(ntsteps),
                                                                   1./srate)
    cdef np.ndarray[DTYPE_c, ndim=2,
                    negative_indices=False] X

    
    cdef np.ndarray[DTYPE_t, ndim=2,
                    negative_indices=False] chol = np.linalg.cholesky(corrcoef)
    cdef np.ndarray[DTYPE_t, ndim=2,
                    negative_indices=False] x = np.random.normal(0, sigma,
                                                                 size=(ntsteps,
                                                                       nX)).T

    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] r
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] theta
    
    x[:, :outputntsteps] = np.dot(x[:, :outputntsteps].T, chol.T).T
    
    X = np.fft.fft(x)
    
    r = (X.real**2 + X.imag**2)**0.5
    theta = np.arctan2(X.imag, X.real)
    
    r[:, 1:] = r[:, 1:] * (abs(freqz[1:])**alpha)**0.5
    
    X.real = r * np.cos(theta)
    X.imag = r * np.sin(theta)
    
    x = np.fft.ifft(X).real[:, :outputntsteps]
    
    return x / x.std() * sigma
    
@cython.boundscheck(False)
cpdef np.ndarray[int, ndim=1, negative_indices=False] createarray(nestlist):
    '''transferring to arrays'''
    cdef np.ndarray[int, ndim=1, negative_indices=False] nestarray, x
    nestarray = np.array([], dtype='int32')
    for x in nestlist:
        nestarray = np.r_[nestarray, x]
    return nestarray



