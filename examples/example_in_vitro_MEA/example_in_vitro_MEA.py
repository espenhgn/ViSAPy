#!/usr/bin/env python
'''do sims of cells, LFP, and output data'''
#import modules
import uuid
import numpy as np
import h5py
import os
from glob import glob
#workaround for plots on cluster
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, lfilter
from time import time, asctime
import ViSAPy
import MoI
import neuron
from mpi4py import MPI


######## set random number generator seed ######################################
SEED = 1234567
POPULATIONSEED = 1234567
np.random.seed(SEED)

################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


######## create unique output folder and copy simulation script ################
if RANK == 0:
    #savefolder = glob('savedata_in_vitro_MEA*')[-1]
    string = asctime().split()
    savefolder = os.path.join(os.path.split(__file__)[0], 'savedata_in_vitro_MEA_')
    for s in string:
        for ss in s.split(':'):
            savefolder += ss + '_'
    savefolder += uuid.uuid4().hex
    os.mkdir(savefolder)
    os.system("cp %s  '%s'" % (__file__, savefolder + '/.'))
else:
    savefolder = None
savefolder = COMM.bcast(savefolder, root=0)



##### load NMODL mechanisms ####################################################
#neuron.h.load_file('stdlib.hoc')
#neuron.h.load_file('nrngui.hoc')
neuron.load_mechanisms("modfiles")


################################################################################
# PARAMETERS
################################################################################

tstart = 0
tend = 60000
dt = 0.05

#set up base parameter file for the LFPy.Cell or LFPy.TemplateCell class,
#without specifying cell model.
cellParameters = {
    'v_init' : -65,
    'passive' : False,
    'timeres_NEURON' : dt,
    'timeres_python' : dt,
    'tstartms' : tstart,
    'tstopms' : tend,
    'verbose' : False,
    'pt3d' : False,
}


# set the default rotation of the cells
defaultrotation = {}


#LFPy can simulate directly to file, but for performance reasons, this
#feature should be avoided
simulationParameters = {
    #'to_file' : True, #file_name set in cellsim()
}

#list up all model folders, associate model neurons by morphology name
morphologies = glob('neuron_models/Large/*/*Morph.hoc') + \
        glob('neuron_models/Medium*/*/*Morph.hoc') + \
        glob('neuron_models/Small*/*/*Morph.hoc')

#one custom code file per morphology
model_paths = glob('neuron_models/Large/*') + \
        glob('neuron_models/Medium*/*') + \
        glob('neuron_models/Small*/*')

#custom codes for cell simulations
custom_codes = []
for model_path in model_paths:
    cell_name = os.path.split(model_path)[-1].lower()
    custom_codes += [os.path.join(model_path, cell_name + '.hoc')]


def getParamsMoIMapping(
    slice_thickness = 200.,
    n_rows = 6,
    n_cols = 17,
    elec_sep = 18.,
    elec_radius = 3.5):
    
    '''Set up MEA with MoI'''

    n_elecs = n_rows * n_cols
    
    # FIXING EARLIER WRONG ELECTRODE POSITIONS
    elec_x_int = np.load('z_integer.npy')
    elec_y_int = np.load('y_integer.npy')
    # For some reason they seem to need individual scaling factors. From pythagoras
    ky = 9 / np.max(np.diff(sorted(elec_y_int)))  
    ###kx = 9 * np.sqrt(3) / np.max(np.diff(sorted(elec_x_int)))
    
    elec_x = elec_x_int * ky
    elec_y = elec_y_int * ky
    elec_x -= np.min(elec_x)
    elec_y -= np.min(elec_y)
        
    paramsMapping = {
                    'use_line_source': True,
                    'include_elec': True,
                    'elec_z' : -slice_thickness/2., #SCALAR
                    'elec_y' : elec_y, # ARRAY
                    'elec_x' : elec_x, # ARRAY
                    'elec_radius': elec_radius,
                    'n_avrg_points' : 10, #Number of electrode averaging points
                    }
    
    paramsMoI = {
        'sigma_G': 0.0, # Below electrode
        'sigma_S': 1.5, # Saline conductivity
        'sigma_T': 0.1, # Tissue conductivity
        'slice_thickness': slice_thickness,
        'steps' : 10,}


    return paramsMapping, paramsMoI
    
paramsMapping, paramsMoI = getParamsMoIMapping()  

#dummy electrodeParameters
electrodeParameters = dict(
    x = paramsMapping['elec_x'],
    y = paramsMapping['elec_y'],
    z = np.array([paramsMapping['elec_z'] for x in paramsMapping['elec_x']]),
)

def getPopParams(#NCOLS=1, NROWS=1,
                 NCOLS=4, NROWS=14,
                 PITCH=np.sqrt(2/(np.sqrt(3)*1400))*1E3, #~1400 mm-2, hex tiling
                 PITCH_STD=5.,
                 HEIGHT = 15.,
                 HEIGHT_STD = 1.,
                 XOFFSET=0., YOFFSET=0., ZOFFSET=-100):
    #set up hexagonal grid of cells
    POPULATION_SIZE = NCOLS * NROWS
    
    x = []
    y = []
    for i in xrange(NROWS):
        if i % 2 == 0:
            x = np.r_[x, np.arange(NCOLS)*PITCH]
        else:
            x = np.r_[x, np.arange(NCOLS)*PITCH + np.cos(np.pi/3)*PITCH]
    
    
        y = np.r_[y, i * np.ones(NCOLS) * np.sin(np.pi/3) * PITCH]
    
    #apply spatial jitter and center population on MEA grid
    x += np.random.normal(scale=PITCH_STD, size=x.size, )
    x -= x.mean()
    x += XOFFSET
    y += np.random.normal(scale=PITCH_STD, size=y.size, )
    y -= y.mean()
    y += YOFFSET
    z = np.random.normal(ZOFFSET+HEIGHT, HEIGHT_STD, x.size)
    
    
    return dict(
        POPULATION_SIZE = NCOLS * NROWS,
        X = x,
        Y = y,
        Z = z
    )

populationParameters = getPopParams(XOFFSET = paramsMapping['elec_x'].mean(),
                                YOFFSET = paramsMapping['elec_y'].mean(),
                                ZOFFSET = paramsMapping['elec_z'])



#set up stimulus by graded synapse input modeled as OU process conductance
gsynParams = dict(
    OUParams = dict(
        T = (tend - tstart)*1E-3,
        dt = dt*1E-3,
        X0 = 0,
        m = 0,
        sigma = 1.,
        nX = populationParameters['POPULATION_SIZE']),
    lambda_d = np.sqrt(2/(np.sqrt(3)*1400))*1E3, #mean cell pitch
    gsyn_mean = 1. / 50000,
    gsyn_std = 1. / 75000,   
)


#some signal processing parameters
nyquist = 1000. / cellParameters['timeres_python'] / 2 

filters = [] 
#presample filter to avoid aliasing
b, a = butter(1, np.array([0.5, 8000]) / nyquist, btype='pass')
filters.append({
    'b' : b,
    'a' : a,
    'filterFun' : lfilter
})

#filter parameters, filterFun must be either scipy.signal.lfilter or filtfilt
b, a = butter(4, np.array([300, 5000]) / nyquist, btype='pass')
filters.append({
    'b' : b,
    'a' : a,
    'filterFun' : filtfilt
})


#Parameters for class ViSAPy.LogBumpFilterBank that sets up
#series of cosine log-bump filters:
logBumpParameters = dict(
    n = 16,
    taps = 401,
    alpha = 0.01,
    nyquist=nyquist,
)


#download experimental data for use in generation of noise
fname = os.path.join('data', 'signal_converted.npy')
if RANK == 0:
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isfile(fname):
        u = urllib2.urlopen('https://www.dropbox.com/s/u6auynymlcbbp36/' +
                            'signal_converted.npy?dl=1')
        f = open(fname, 'w')
        f.write(u.read())
        f.close()    
COMM.Barrier()


#Noise parameters including noise covariance matrix
noiseParameters = None
#extract noise covariances extracted from experimental tetrode recording
noiseFeaturesParameters = dict(logBumpParameters)
noiseFeaturesParameters.update({
    'fname' : fname,
    'outputfile' : os.path.join(savefolder, 'ViSAPy_noise.h5'),
    'T' : 15000,
    'srate_in' : 20000,
    'srate_out' : 2 * nyquist,
    'NFFT' : 2**16,
    'psdmethod': 'mlab',
    'remove_spikes' : True,
    #parameters passed to class SpikeACut, only used if remove_spikes == True
    'remove_spikes_args' : {
        'TEMPLATELEN' : 32,
        'TEMPLATEOFFS' : 0.5,
        'threshold' : 5, #standard deviations
        'data_filter' : {
            'filter_design' : butter,
            'filter_design_args' : {
                'N' : 2,
                'Wn' : np.array([300., 5000.]) / nyquist,
                'btype' : 'pass',
                },
            'filter' : filtfilt,
        },
    },
    'amplitude_scaling' : 1E-3,
})

#container file for noise output etc.
noise_output_file = os.path.join(savefolder, 'ViSAPy_noise.h5')



################################################################################
## MAIN
################################################################################

################################################################################
## Step 1:  Estimate PSD and covariance between channels, here using
##          an experimental dataset.
##          
##          In the present ViSAPy, we should use only a single RANK for this
##          and subsequent steps, we also skip regenerating noise and spike
##          events, because it can take some time for long simulation durations
##          
if RANK == 0:
    if not os.path.isfile(noise_output_file):
        noise_features = ViSAPy.NoiseFeatures(**noiseFeaturesParameters)


################################################################################
## Step 2:  Generate synthetic noise with PSD and covariance channels extracted
##          using class NoiseFeatures, preserving the overall amplitude.
##          We choose to save directly to file, as it will be used in
##          later steps
##          
        noise_generator = ViSAPy.CorrelatedNoise(psd=noise_features.psd,
                                                 C=noise_features.C,
                                                 **noiseFeaturesParameters)
        #file object containing extracellular noise and related data
        f = h5py.File(noise_output_file)
        f['data'] = noise_generator.correlated_noise(T = cellParameters['tstopms'])
        f.close()

#sync
COMM.Barrier()

################################################################################
## Step 3:  Fix seed and set up Testdata object, generating a model cell
##          population, find and distribute synapse inputs with spiketrains from
##          network, run simulations for extracellular potentials,
##          collect data and generate final benchmark data
##

np.random.seed(POPULATIONSEED)

benchmark_data = MoI.BenchmarkDataMoI(
    cellParameters = cellParameters,
    morphologies = morphologies,
    defaultrotation = defaultrotation,
    simulationParameters = simulationParameters,
    populationParameters = populationParameters,
    electrodeParameters = electrodeParameters,
    noiseFile = noise_output_file,
    filters = filters,
    savefolder = savefolder,
    default_h5_file = 'lfp_cell_%.3i.h5',
    nPCA = 2,
    TEMPLATELEN = 80,
    TEMPLATEOFFS = 0.3,
    spikethreshold = 3.,
    custom_codes = custom_codes,
    paramsMapping = paramsMapping,
    paramsMoI = paramsMoI,
    gsynParams = gsynParams,
)
print 'setup ok!'

benchmark_data.run()
print 'run ok'

benchmark_data.collect_data()
print 'collect ok'



#plot single cell output
myplot = ViSAPy.plotBenchmarkData(benchmark_data)


for i in range(populationParameters['POPULATION_SIZE']):
    if i % SIZE == RANK:
        fig = myplot.plot_figure_13(cellindices=np.array([i]),
                                    bins=10**np.linspace(np.log10(10), np.log10(1E3), 67))
        fig.savefig(os.path.join(savefolder, 'cell_%.2i.pdf' % i))
        plt.close(fig)

COMM.Barrier()

