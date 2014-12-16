#!/usr/bin/env python
'''class SpikeACut for removing spike events in raw extracellular recordings'''
import numpy as np
import scipy.signal as ss

class SpikeACut(object):
    def __init__(self, data,
                 TEMPLATELEN=32,
                 TEMPLATEOFFS=0.5,
                 threshold=3.,
                 data_filter = {
                    'filter_design' : ss.butter,
                    'filter_design_args' : {
                        'N' : 2,
                        'Wn' : np.array([300., 5000.]) / 16000.,
                        'btype' : 'pass',
                    },
                    'filter' : ss.filtfilt
                },
            ):
        '''
        Initialization of class SpikeACut
        
        The intended usage of this class is removing putative spike events in
        raw extracellular recordings, by bandpass filtering the input data,
        detecting threshold crossings, mask out time bins surrounding threshold
        crossings with NaNs, and concatenate time bins without NaNs. 
        
        Keyword arguments:
        ::
            
            data : np.ndarray, nchannels x ntsteps
            TEMPLATELEN : int, tsteps removed per threshold crossing 
            TEMPLATEOFFS : float on [0, 1), relative offset
            threshold : float, threshold in unit of standard deviations
            data_filter : dict, containing
                filter_design, function reference, e.g., scipy.signal.butter
                filer_design_args, filter design settings for filter_design fun
                filter : function reference, e.g., scipy.signal.filtfilt
        
        '''
        self.data = data
        self.TEMPLATELEN = TEMPLATELEN
        self.TEMPLATEOFFS = TEMPLATEOFFS
        self.threshold = threshold
        self.data_filter = data_filter
        
        #create filter coefficients and apply filter to data
        b, a = data_filter['filter_design'](**data_filter['filter_design_args'])
        data_filtered = self.data_filter['filter'](b, a, data)
        
        #find where filtered is above threshold
        bools = abs(data_filtered) >= self.threshold * data_filtered.std()
        
        #indices where above threshold
        spi = np.where(bools.sum(axis=0) >= 1)[0]
        
        #main attribute 
        self.data_processed = self.calcLFPnospikes(data, spi)

    
    def calcLFPnospikes(self, data, spi):
        '''
        return data without spikes at times spi
        
        Keyword arguments:
        ::
            
            data : np.ndarray of shape (nchannels, ntsteps) w. extracellular pot.
            spi : np.nparray, vector containing tsteps which cross threshold
        
        Returns:
        ::
            
            np.ndarray, shape (nchannels, arbitrary), extracellular potential
                with high amplitude events removed
        '''
        #data traces with NaNs in place of putative spikes
        data_nan = self._calcResidualNoise(data, spi)

        #delete NaN entries by concatenating numerical entries
        data_nospikes = np.delete(data_nan, np.where(np.isnan(data_nan))[1], 1)
        
        return data_nospikes
        

    def _calcResidualNoise(self, data, spi):
        '''
        data at self.TEMPLATELEN indicies determined by spiketimes in spi
        are set to nan, offset by off.

        Keyword arguments:
        ::
            
            data : np.ndarray of shape (nchannels, ntsteps) w. extracellular pot.
            spi : np.nparray, vector containing tsteps which cross threshold
        
        Returns:
        ::
            
            np.ndarray, shape (nchannels, ntsteps), extracelluilar potential w.
                high amplitude events masked with NaNs
        '''
        for i in spi:
            #for each timestep in spi we mask out neighbouring indicies 
            #of a length TEMPLATELEN and with a relative offset of TEMPLATEOFFS
            inds = np.linspace(i - int(self.TEMPLATEOFFS*self.TEMPLATELEN)+1,
                               i + int((1-self.TEMPLATEOFFS)*self.TEMPLATELEN),
                               self.TEMPLATELEN).astype(int)

            #fill in NaN values, taking into account the signal duration
            data[:, inds[(inds >= 0) & (inds < data.shape[1])]] = np.nan
    
        return data



if __name__ == '__main__':
    pass
