#!/usr/bin/env python
'''
Using log-gabor filter bank to mimic noise correlation structure of data.

[TODO] We don't reproduce highest frequency band
'''
import numpy as np
from scipy.signal import lfilter, resample, filtfilt, butter
from scipy.fftpack import ifft, fft, fftshift, fftfreq
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import h5py
import os
import glob
from .spike_a_cut import SpikeACut
from scipy.interpolate import interp1d
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


################# Function definitions #########################################

def scale_matrix(m, new, order=1, output=None):
    '''
    Rescale a matrix m to be of shape new with interpolation order = order.

    Keyword arguments:
    ::
        
        m      : input matrix
        new    : tuple indicating shape of new matrix
        order  : spline interpolation order
        output : dtype of returned matrix


    Returns:
    ::
        
        ndarray with shape new
    '''
    if output is None: output = m.dtype
    old = m.shape
    slices = tuple([slice(0,i-1,k*1j) for i,k in zip(old,new)])
    coords = np.mgrid[slices]
    
    return map_coordinates(m, coords, order=order, output=output)


# nonlinearity for stretching x axis (and its inverse)
def nlin(x):
    return np.log(x + 1e-20)
def invnl(x):
    return np.exp(x) - 1e-20 # inverse nonlinearity

def ff(x, c, dc):
    """raised cosine basis vector"""
    y = (x-c) * np.pi / dc /2
    y[y>np.pi] = np.pi
    y[y<-np.pi] = -np.pi
    return (np.cos(y)+1)/2

def orth(A):
    U, S, V = np.linalg.svd(A)
    m, n = A.shape
    return U

def logbumps(n=5, t=32, alpha=.01, spikes=False, identity=False, normalize=True,
             debug=False, figno=25):
    """
    Use Jon Pillow's cosine bump basis exactly so it is trivial to cite.
    An alternative basis would be the decomposition in the steerable pyramid.
    
    Keyword arguments:
    ::

        n         : number of bumps
        t         : time resolution
        alpha     : controls non-linearity of bumps (closer to zero is more nonlinear)
        spikes    : discard the basis with peak at 0
        identity  : prepend an identity filter at beginning
        
    Returns:
    ::
        lb        : np.ndarray
        
    """
    # create n cosine bumps (with first and last being half bumps)
    n += 1
    if spikes: n += 1
    if identity: n -= 1
    x = np.linspace(0,1,t)
    c = np.linspace(0,x[-1],n)
    dc = c[1] - c[0]
    y = (x[None,:] - c[:,None]) * np.pi / dc
    y[y>np.pi] = np.pi
    y[y<-np.pi] = -np.pi
    b = (np.cos(y)+1)/2
    
    
    # change to log time scale
    lb = np.zeros_like(b)
    logx = np.log(x+alpha)
    
    for i in range(n):
        f = interp1d(logx, b[i], kind='linear')
        nx = np.linspace(logx[0], logx[-1], t)
        lb[i] = f(nx)

    # flip temporal order for convolution filter
    lb = lb[:,::-1]

    # flip time order
    lb = lb[::-1]
    
    ## discard last bf
    lb = lb[:-1]
    
    # if spike basis, remove first bf
    if spikes:
        lb = lb[1:]

    if identity:
        # prepend identity filter        
        id = np.zeros(t)
        id[0] = 1.
        lb = np.concatenate(([id], lb))

    if normalize:
        # filters should sum to 1
        lb /= lb.sum(axis=1)[:,None]
        
    if debug:
        plt.figure(figno)
        plt.clf()        
        plt.subplot(4,1,1)
        plt.plot(x, b.T)
        plt.title('bumps')
        plt.subplot(4,1,2)
        plt.plot(lb.sum(axis=0))
        plt.title('bumps should add to one')
        plt.subplot(4,1,3)
        plt.loglog(b.T)
        plt.title('bases')
        plt.subplot(4,1,4)
        plt.plot(lb.T)
        plt.title('log-bases')
        plt.subplots_adjust(hspace=.6)

    
    return lb


################# Class definitions ############################################

class LogBumpFilterBank(object):
    '''class LogBumpFilterBank'''
    def __init__(self, nyquist=16000., n=10, taps=101, alpha=0.08,
                 **kwargs):
        '''
        Initialization of class LogBumpFilterBank
        
        This class provides methods to construct log-bump filter coefficients
        
        Keyword arguments:
        ::
            
            nyquist  : int, Nyquist frequency
            n        : int, number of log-gabor bumps to use
            taps     : int, number of taps of filter
            alpha    : float, squeeze parameter
            **kwargs : discarded arguments
            
        '''
        #set some attributes
        self.nyquist = int(nyquist)
        self.n = n
        self.taps = taps
        self.alpha = alpha
        if not self.taps % 2:
            raise ValueError('taps should be odd')

        #sets attributes logbases, filt
        self.create_filter_bank()


    def create_filter_bank(self):
        '''
        Create time-domain acausal log-gabor filter bank
        
        Non-public method
        '''
        # fourier domain logarithmic basis
        self.logbases = logbumps(n=self.n, t=self.nyquist,
                                 spikes=False, identity=False,
                                 normalize=False, alpha=self.alpha)        
        
        # symmetrize so ifft's are real
        symm = np.empty((self.n, 2*self.nyquist-1))
        symm[:, :self.nyquist] = self.logbases
        symm[:, self.nyquist:] = self.logbases[:, 1:][:, ::-1]
        
        
        # frequency to time domain, shift to make causal
        self.filt = ifft(symm, axis=1)
        self.filt = np.real(self.filt)
        self.filt = fftshift(self.filt, axes=1)

        # truncate filters to specifed number of taps
        self.filt = self.filt[:, self.nyquist - (self.taps//2+1):self.nyquist + self.taps//2]


    def filter(self, data):
        '''
        Apply each log-bump filter in filter bank to data
        
        Keyword arguments:
        ::
            
            data : np.ndarray, noise data of shape (T, nchannels)
            
        Returns:
        ::
            
            np.ndarray, of shape (n, T, nchannels)
        
        '''
        fdata = np.zeros((self.n,) + data.shape)

        # filter data with filter bank, handling delay
        zi = np.zeros((self.taps-1, data.shape[1]))
        pad = (self.taps-1)//2
        for i in range(self.n):
            print('.'),
            zi.fill(0.)
            f, trans = lfilter(self.filt[i], 1, data, axis=0, zi=zi)
            fdata[i,:-pad] = f[pad:]
            fdata[i,-pad:] = trans[:pad]
    
        return fdata


class NoiseFeatures(LogBumpFilterBank):
    '''class NoiseFeatures, inherits class LogBumpFilterBank'''
    def __init__(self,
                 fname,
                 outputfile=os.path.join('savedata', 'ViSAPy_noise.h5'),
                 T=1000.,
                 srate_in=32000,
                 srate_out=32000,
                 remove_spikes=True,
                 remove_spikes_args = {
                    'TEMPLATELEN' : 32,
                    'TEMPLATEOFFS' : 0.5,
                    'threshold' : 5, #standard deviations
                    'data_filter' : {
                        'filter_design' : butter,
                        'filter_design_args' : {
                            'N' : 2,
                            'Wn' : np.array([300., 5000.]) / 16000.,
                            'btype' : 'pass',
                            },
                        'filter' : filtfilt
                        },
                    },
                 psdmethod='mlab',
                 NFFT=2**16,
                 **kwargs
                 ):
        '''
        Initialization of class NoiseFeatures, inherits class LogBumpFilterBank
        
        Keyword arguments:
        ::

            fname         : str, HDF5 file with h5['data'].shape = (time, channels)
            outputfile    : str, path to file output
            T             : float, sample time of input data in (ms)
            srate_in      : float, sampling rate of input data
            srate_out     : float, sampling rate of output data
            remove_spikes : remove segments of data with putative spikes
            remove_spikes_args : see class ViSAPy.SpikeACut
            psdmethod     : str, 'mlab' or 'scipy.fft', method used for PSD est.
            NFFT          : int, FFT block length used for PSD estimate
            **kwargs      : see parent class LogBumpFilterBank
            
        '''
        #initialize parent class
        LogBumpFilterBank.__init__(self, **kwargs)
        
        self.fname = fname
        self.outputfile = outputfile
        self.T = T
        self.srate_in = srate_in
        self.srate_out = srate_out
        self.tsteps_in = int(self.T * self.srate_in / 1000.)
        self.remove_spikes = remove_spikes
        self.remove_spikes_args = remove_spikes_args
        self.NFFT=NFFT
        self.psdmethod = psdmethod

        #assess if inputdata should be resampled
        if self.srate_in != self.srate_out:
            self.resample = True
        else:
            self.resample = False

        #skip loading raw data and doing analysis, if outputfile exist
        if os.path.isfile(self.outputfile):
            self.load()
        else:
            self._load_data()
            if self.remove_spikes:
                self._remove_spikes(**self.remove_spikes_args)        
            self.psd, self.freqs = self.spectra(self.input_data)
            
            self.fdata = self.filter(self.input_data)
            self.C = self.covariance(self.fdata)
        
            #dump class attributes derived from experimental traces
            self.save()


    def save(self, ):
        '''
        save class attributes to HDF5 file so that class instance 
        can be reconstructed later without computing output once more
        
        This method takes no keywords
        
        '''
        f = h5py.File(self.outputfile)
        for attribute in ['input_data', 'psd', 'freqs', 'fdata', 'C', 'NFFT', 'alpha', 'n', 'taps']:
            f[attribute] = getattr(self, attribute)
        f.close()

    
    def load(self, ):
        '''
        reload class attributes from HDF5 file output, reconstructing a
        previously created class instance.
        
        If the method cannot load all attributes, it will resort to create
        all attributes.

        This method takes no keywords.

        '''
        f = h5py.File(self.outputfile)
        for attribute in ['input_data', 'psd', 'freqs', 'fdata', 'C']:
            try:
                setattr(self, attribute, f[attribute][()])
            except:
                f.close()
                self._load_data()
                if self.remove_spikes:
                    self._remove_spikes(**self.remove_spikes_args)        
                self.psd, self.freqs = self.spectra(self.input_data)
                self.fdata = self.filter(self.input_data)
                self.C = self.covariance(self.fdata)
            
                #dump class attributes derived from experimental traces
                self.save()
                break


    def _load_data(self):
        '''
        Load at most self.tsteps time points, if provided
        
        Non-public method
        '''
        if self.fname.endswith('h5'):
            h5 = h5py.File(self.fname, 'r')
            self.input_data = h5['data'][()]
            h5.close()
        elif self.fname.endswith('npy'):
            self.input_data = np.load(self.fname)
        else:
            raise Exception('end of %s must be .npy or .h5' % self.fname)
        
        
        #ensure cols are time, rows channels:
        if self.input_data.shape[1] > self.input_data.shape[0]:
            self.input_data = self.input_data.T
        
        if self.tsteps_in != None:
            self.input_data = self.input_data[:self.tsteps_in]
            
        if self.resample:
            self.input_data = resample(self.input_data,
                                       int(self.T * self.srate_out / 1000))
        
        
    def _remove_spikes(self, **kwargs):
        '''
        concatenate parts of lfp-signals without spikes using class SpikeACut
        
        Non-public method
        '''
        cut = SpikeACut(data=self.input_data.T, **kwargs)
        
        self.input_data = cut.data_processed.T

        
    def covariance(self, data):
        '''
        Compute signal covariance across channels for each frequency band
        
        Keyword arguments:
        ::
        
            data: np.ndarray, shape (n, T, channels), frequency resolved data
        
        Returns:
        ::
            
            np.ndarray, shape (n, nchannels, nchannels)
        
        '''
        C = np.empty((self.n,) + self.input_data.shape[1:]*2)
        for i in range(self.n):
            C[i] = np.cov(data[i], rowvar=0)

        return C


    def spectra(self, data):
        '''
        Compute per channel 2-sided psd's
        
        Keyword arguments:
        ::
            
            data : np.ndarray, shape (T, channels), frequency resolved data
        
        Returns:
        ::
            
            psd : np.ndarray, shape(NFFT, nchannels) mean power spectral density
                (PSD) across channels
            freqs : np.ndarray, length NFFT PSD frequency vector
        '''
        if self.psdmethod == 'scipy.fft':
            psd = np.abs(fft(self.input_data, axis=0, n=self.NFFT))
            freqs = fftfreq(n=self.NFFT)
        elif self.psdmethod == 'mlab':
            psd0 = np.zeros((self.NFFT, self.input_data.shape[1]))
            for i in range(self.input_data.shape[1]):
                #compute the per channel PSD
                XX, freqs0 = plt.mlab.psd(self.input_data[:, i],
                                          Fs=1,
                                          NFFT=self.NFFT,
                                          sides='twosided',
                                          noverlap=int(self.NFFT*3//4))
                #redo normalization
                psd0[:, i] = np.sqrt(XX*self.NFFT).T
            
            #for compatibility with 'scipy.fft' output, reorder elements
            freqs = np.r_[freqs0[self.NFFT//2:], freqs0[:self.NFFT//2]]
            psd = np.r_[psd0[self.NFFT//2:, ], psd0[:self.NFFT//2, ]]
        else:
            errmsg = "psdmethod = {} not in ['mlab', 'scipy.fft']".format(
                self.psdmethod)
            raise Exception(errmsg)

        return psd, freqs



class CorrelatedNoise(LogBumpFilterBank):
    '''class CorrelatedNoise, inherits class NoiseFeatures'''
    def __init__(self, psd, C, amplitude_scaling=1., savefolder='savefolder', SEED=12345678, **kwargs):
        '''
        Initialization of class CorrelatedNoise, inherits class NoiseFeatures.
        Provide methods for generating 
        
        Keyword arguments:
        ::
            
            psd : np.ndarray, shape (NFFT, nchannels), PSD in each channel
            C : np.ndarray, shape (nbumps, nchannels, bchannels), covariance
                between channels in each nbumps frequency band
            amplitude_scaling : float, final output scale factor
            SEED : int
            **kwargs : see class LogBumpFilterBank
        
        '''
        #initialize parent class
        LogBumpFilterBank.__init__(self, **kwargs)
        
        #set some attributes
        self.psd = psd
        self.C = C
        self.amplitude_scaling = amplitude_scaling
        self.savefolder = savefolder
        self.SEED = SEED
        

    def filter(self, data):
        '''
        Apply each log-bump filter in filter bank to data
        
        Keyword arguments:
        ::
            
            data : np.ndarray, noise data of shape (T, nchannels)
            
        Returns:
        ::
            
            np.ndarray, of shape (n, T, nchannels)
        
        '''
        f_ = h5py.File(os.path.join(self.savefolder,
                                    'tmpnoise_rank{}.h5'.format(RANK)), 'w')

        # filter data with filter bank, handling delay
        zi = np.zeros((self.taps-1, data.shape[1]))
        pad = (self.taps-1)//2
        for i in range(self.n):
            if i % SIZE == RANK:
                print('.'),
                f, trans = lfilter(self.filt[i], 1, data, axis=0, zi=zi)
                f_[str(i)] = np.r_[f[pad:], trans[:pad]].astype('float32')
        
        f_.close()
        
        # allow all file writes to finish
        COMM.Barrier()
        
        if RANK == 0:
            f = h5py.File(os.path.join(self.savefolder, 'tmpnoise.h5'), 'w')
            f['data'] = np.zeros((self.n,) + data.shape, dtype='float32')
            for j in range(SIZE):
                f_ = h5py.File(os.path.join(self.savefolder,
                                            'tmpnoise_rank{}.h5'.format(j)),
                               'r')
                for i in range(self.n):
                    if i % SIZE == j:
                        f['data'][i, ] += f_[str(i)] #.astype('float32')
                f_.close()
            f.close()
            
            print('finished writing {}'.format(os.path.join(self.savefolder,
                                                            'tmpnoise.h5')))
            
            
            # remove temporary pink noise files
            for j in range(SIZE):
                fname = os.path.join(self.savefolder,
                                     'tmpnoise_rank{}.h5'.format(j))
                print('deleting {}'.format(fname))
                try:
                    os.remove(fname)
                except OSError as e:  ## if failed, report it back to the user ##
                    print("Error: {} - {}.".format(e.filename, e.strerror))
                                             
        COMM.Barrier()
    

    def correlated_noise(self, T):
        '''
        Generate correlated noise with per channel PSD and noise covariances
        similar to recorded data.

        Keyword arguments:
        ::
        
            T : float, sample length in ms

        Returns:
        ::
            
            np.ndarray, shape (nchannels, T*srate_out*1E3+1), correlated noise
        '''
        
        # rf = self.filter(self.pink_noise(T))
        self.filter(self.pink_noise(T))

        if RANK == 0:
            f = h5py.File(os.path.join(self.savefolder, 'tmpnoise.h5'), 'r')
            DATA = np.zeros(f['data'][0, ].shape, dtype='float32')
            for i in range(self.n):
                sqrtC = np.linalg.cholesky(self.C[i])
                data = f['data'][i, ]
                data -= data.mean(axis=0)
                data /= data.std(axis=0)
                data = np.dot(sqrtC, data.T).T
                DATA += data.astype('float32')
            
            f.close()
            # remove temporary file
            print('removing temporary file {}'.format(os.path.join(self.savefolder, 'tmpnoise.h5')))
            try:
                os.remove(os.path.join(self.savefolder, 'tmpnoise.h5'))
            except OSError as e:
                print("Error: {} - {}.".format(e.filename, e.strerror))
            DATA = DATA.T
            DATA *= self.amplitude_scaling
            DATA = DATA.astype('float32')
        else:
            DATA = None
        return COMM.bcast(DATA)
  

    def pink_noise(self, T):
        '''
        Create noise with correct mean spectrum
        
        Same second axis as in the number of channels of file fname
        
        Keyword arguments:
        ::
        
            T : float, sample length in ms
        
        Returns:
        ::
            
            np.ndarray, shape (T*srate_out*1E3+1, nchannels),
                uncorrelated pinkened noise
        '''
        #create pink noise one channel at the time, concatenate last axis
        rows = int(T * self.nyquist*2. / 1000. + 1)
        cols = self.C.shape[-1]
        
        #temporary work with dimensions in base 2**12, much faster ifft
        rowsfft = 2**12 * (divmod(rows, 2**12)[0]+1)
        
        data = np.zeros((rows, cols), dtype='float32')
        
        #get state of random number generation, set unique seed per RANK
        state = np.random.get_state()
        np.random.seed(self.SEED+RANK)
        
        
        #deal with each channel independently
        for i in range(cols):
            if i % SIZE == RANK:
                #scale the PSD to fit new dimensions
                psd = scale_matrix(self.psd[:, i], (rowsfft, )).astype('complex')
                
                #random phase angles distributed in frequency domain
                phases = np.random.uniform(low=0, high=2*np.pi, size=rowsfft)
                
                #consruct ifft vector
                psd *= np.exp(1j*phases)
                del phases
                
                #create pinkened nosie
                pink0 = ifft(psd).real
                del psd
                
                #normalize noise
                pink0 -= pink0.mean()
                pink0 /= pink0.std()
                
                data[:, i] = pink0[:rows].astype('float32')
                
            print('.'),
        
        print('pinkened noise done')

        #reset state
        np.random.set_state(state)
        
        #sum arrays
        DATA = np.zeros(data.shape, dtype='float32')
        COMM.Allreduce(data, DATA)        
        return DATA


if __name__ == '__main__':
    
    log_bump_params = dict(n=16, taps=401, alpha=0.01, nyquist=16000,)
    
    parameterset = dict(log_bump_params)
    parameterset.update(dict(outputfile=os.path.join('savedata', 'ViSAPy_noise.h5'),
        T=5000.,
        srate_in=48000.,
        srate_out=32000.,
        remove_spikes=False,
        amplitude_scaling=1.,
        psdmethod='mlab',
        NFFT=2**16,
    ))
    
    parametersets = [parameterset, dict(parameterset)]
    parametersets[1].update({
        'remove_spikes' : True,
        'remove_spikes_args' : {
            'TEMPLATELEN' : 32, #1 ms
            'TEMPLATEOFFS' : 0.5,
            'threshold' : 5, #standard deviations
            'data_filter' : {
                'filter_design' : butter,
                'filter_design_args' : {
                    'N' : 2,
                    'Wn' : np.array([300., 5000.]) / 16000.,
                    'btype' : 'pass',
                    },
                'filter' : filtfilt   
            },
        },
    })

    #iterate with and without spike removal
    for parameterset in parametersets:

        #clear out savedata folder
        if os.path.isdir('savedata'):
            for f in glob.glob('savedata/*'):
                os.remove(f)
        else:
            os.mkdir('savedata')
        
        
        #set up NoiseFeatures object
        fname = '../../ExperimentalData2012.10.19_ChristinaPolytrode/08_2012101910.bin_tetrode_raw_cleaned.h5'
        noisefeatures = NoiseFeatures(fname, **parameterset)
    
 
        #set up CorrelatedNoise object and create some noise
        c = CorrelatedNoise(psd=noisefeatures.psd,
                            C=noisefeatures.C,
                            amplitude_scaling=1.,
                            **log_bump_params)
        noise = c.correlated_noise(T=noisefeatures.T)
        
        
        #compare input and output
        fig, axes = plt.subplots(2,1)
        
        axes[0].plot(noisefeatures.input_data)
        axes[0].axis(axes[0].axis('tight'))
        axes[0].set_title('input data, resampled')
        
        axes[1].plot(noise.T)
        axes[1].axis(axes[1].axis('tight'))
        axes[1].set_xlabel('time step (-)', labelpad=0.1)
        axes[0].set_title('output data')
    
    
    plt.show()
    
    
