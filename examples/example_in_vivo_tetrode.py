#!/usr/bin/env python
'''
ViSAPy example script for generating benchmark corresponding to an in vivo
tetrode recording
'''
#import modules
import uuid
import urllib.request, urllib.error, urllib.parse
import zipfile
import numpy as np
import h5py
import os
import glob
#workaround for plots on cluster
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
from scipy.signal import filtfilt, butter, lfilter
from time import asctime, time
# import main ViSAPy classes used throughout this script. We also import
# NEST even if it is not used in order to prevent spurious segmentation
# errors that may otherwise occur when NEST is loaded with MPI and NEURON
# at the same time. 
import nest
from ViSAPy import NoiseFeatures, CorrelatedNoise, ExternalNoiseRingNetwork, RingNetwork, BenchmarkDataRing, plotBenchmarkData
import neuron
from mpi4py import MPI

#tic - toc
tic = time()

######## set random number generator seed ######################################
SEED = 123456
POPULATIONSEED = SEED
np.random.seed(SEED)


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


######## create unique output folder and copy simulation script ################
if RANK == 0:
    # savefolder = glob.glob('savedata_tetrode*')[-1]
    string = asctime().split()
    savefolder = os.path.join(os.path.split(__file__)[0], 'savedata_tetrode_')
    for s in string:
        for ss in s.split(':'):
            savefolder += ss + '_'
    savefolder += uuid.uuid4().hex
    os.mkdir(savefolder)
    os.system("cp %s  '%s'" % (__file__, savefolder + '/.'))
else:
    savefolder = None
savefolder = COMM.bcast(savefolder, root=0)


######### Fetch Hay et al. 2011 model files, unzip locally #####################
if not os.path.isfile('L5bPCmodelsEH/morphologies/cell1.asc'):
    if RANK == 0:
        #get the model files:
        u = urllib.request.urlopen('http://senselab.med.yale.edu/ModelDB/' +
                            'eavBinDown.asp?o=139653&a=23&mime=application/zip')
        localFile = open('L5bPCmodelsEH.zip', 'w')
        localFile.write(u.read())
        localFile.close()
        #unzip:
        myzip = zipfile.ZipFile('L5bPCmodelsEH.zip', 'r')
        myzip.extractall('.')
        myzip.close()
    
        #compile NMODL language files
        os.system('''
                  cd L5bPCmodelsEH/mod/
                  nrnivmodl
                  cd -
                  ''')
    COMM.Barrier()


##### load NEURON mechanisms from Hay et al. 2011 ##############################
neuron.load_mechanisms("L5bPCmodelsEH/mod")


################################################################################
# PARAMETERS
################################################################################

#set up base parameter file for the LFPy.Cell or LFPy.TemplateCell class,
#without specifying cell model. 
cellParameters = {
    'v_init' : -80,
    'passive' : False,
    'nsegs_method' : None,
    'dt' : 2**-5,
    'tstart' : 0.,
    'tstop' : 120500, #2 minutes without 500 ms startup transient
    'verbose' : False,
    'pt3d' : False,
}


#in this particular set up, each cell will be a random permutation of each
#morphology and templatefile specification of Hay et al 2011.
morphologies = [
    'L5bPCmodelsEH/morphologies/cell1.asc',
    'L5bPCmodelsEH/morphologies/cell2.asc',
]
templatefiles = [
    ['L5bPCmodelsEH/models/L5PCbiophys2.hoc',
     'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
    ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
     'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
    ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
     'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
]
#pointer to template specification name, cf. Linden et al. 2014
cellParameters.update(dict(templatename = 'L5PCtemplate'))


# set the default rotation of the cells
defaultrotation = {}


#LFPy can simulate directly to file, but for performance reasons, this
#feature should be avoided
simulationParameters = {
    #'to_file' : True, #file_name set in cellsim()
}


#parameters for the signal-generating model population. 
populationParameters = {
    'POPULATION_SIZE' : 6,
    'radius' : 50,
    'killzone' : 25,
    'z_min' : 0,
    'z_max' : 150,
    'X' : np.array([ [0, 0,  0,  -40, -40, 0, 0],
                   [0, 0,  0,   40,  40, 0, 0]]),
    'Y' : np.array([ [0, 0, -50, -50, -50, 0, 0],
                   [0, 0,  0,    0,   0, 0, 0]]),
    'Z' : np.array([-np.inf, -50.01, -50, 0, 2000, 2000.01, np.inf]),
    'min_cell_interdist' : 25.,
}


#Recording electrode emulating NeuroNexus laminar tetrode array
#first contact superficial
x, y, z = -np.mgrid[0:1, 0:1, -3:1] * 50
N = np.empty((x.size, 3))
for i in range(N.shape[0]): N[i,] = [1, 0, 0]

#dictionary passed to class LFPy.RecExtElectrode
electrodeParameters = {
    'x' : x.flatten(),
    'y' : y.flatten(),
    'z' : z.flatten(),
    'sigma' : 0.3,      #extracellular conductivity
    'N' : N,            #electrode contact surface normal
    'r' : 7.5,          #contact radius
    'n' : 100,          #number of points averaged over on contact
    'method' : 'soma_as_point'   #soma segment assumed spherical, dendrites lines
}

##one dimensional drift, here; shifting electrode.z +5 mum every 1000 ms
#driftParameters = {
#    'driftInterval' : 1000,
#    'driftShift' : 5,
#    'driftOnset' : 0,
#}
##but we're not using that feature here
driftParameters = None

#synaptic parameters: AMPA - excitatory, GABA_A - inhibitory
synparams_AMPA = {         #Excitatory synapse parameters
    'e' : 0,           #reversal potential
    'syntype' : 'Exp2Syn',   #conductance based two-exponential synapse
    'tau1' : 1.,         #Time constant, rise
    'tau2' : 3.,         #Time constant, decay
    'weight' : 0.0125,   #Synaptic weight
    'section' : ['apic', 'dend'],
    'nPerArea' : [45E-3, 45E-4], #mean +- std
}
synparams_GABA_A = {         #Inhibitory synapse parameters
    'e' : -80,
    'syntype' : 'Exp2Syn',
    'tau1' : 1.,
    'tau2' : 12.,
    'weight' : 0.025,
    'section' : ['soma', 'apic', 'dend'],
    'nPerArea' : [20E-3, 20E-4],
}


#parameters for ViSAPy.*Network instance
networkParameters = {
    #class Network
    'simtime' :     cellParameters['tstop']-cellParameters['tstart'],
    'dt' :          cellParameters['dt'],
    'total_num_virtual_procs' : SIZE,
    'savefolder' :  savefolder,
    'label' :       'spikes',
    'to_file' :     True,
    'to_memory' :   False,
    'print_time' :  False,
    #class RingNetwork
    'N' :           12500,
    'theta' :       20.,
    'tauMem' :      20.,
    'delay' :       2.,
    'J_ex' :        0.05,
    'g' :           5.0,
    'eta' :         0.9,
}
#class ExternalNoiseRingNetwork we're using below need a few extra arguments
ExternalNoiseRingNetworkParameters = {
    'tstop' :         cellParameters['tstop'],
    'invertnoise_ex' :  True,
    'invertnoise_in' :  False,
    'rate' :            40,
    'projection':       ['exc', 'inh'],
    'weight' :          0.5,
}



#nyquist frequency of simulation output
nyquist = 1000. / cellParameters['dt'] / 2 


#Parameters for class ViSAPy.LogBumpFilterBank that sets up
#series of cosine log-bump filters:
logBumpParameters = dict(
    n = 16,
    taps = 401,
    alpha = 0.01,
    nyquist=nyquist,
)


#set up filtering steps of extracellular potentials
filters = [] 
#presample filter to avoid aliasing
b, a = butter(1, np.array([0.5, 8000.]) / nyquist, btype='pass')
filters.append({
    'b' : b,
    'a' : a,
    'filterFun' : lfilter
})
b, a = butter(4, np.array([300., 5000.]) / nyquist, btype='pass')
filters.append({
    'b' : b,
    'a' : a,
    'filterFun' : filtfilt
})
#note, filterFun should be either scipy.signal.lfilter or filtfilt


#download experimental data for use in generation of noise
fname = os.path.join('data', '08_2012101910.bin_tetrode_raw_cleaned.h5')
if RANK == 0:
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isfile(fname):
        u = urllib.request.urlopen('https://www.dropbox.com/s/28hmlig0h1d745e/' +
                            '08_2012101910.bin_tetrode_raw_cleaned.h5?dl=1')
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
    'T' : 15000.,
    'srate_in' : 48000.,
    'srate_out' : 2 * nyquist,
    'NFFT' : 2**16,
    'psdmethod': 'mlab',
    'remove_spikes' : True,
    #parameters passed to class SpikeACut, only used if remove_spikes == True
    'remove_spikes_args' : {
        'TEMPLATELEN' : 32, #1 ms
        'TEMPLATEOFFS' : 0.5,
        'threshold' : 5, #standard deviations
        'data_filter' : {
            'filter_design' : butter,
            'filter_design_args' : {
                'N' : 2,
                'Wn' : np.array([300., 5000.]) / nyquist,
                'btype' : 'pass',
                },
            'filter' : filtfilt
            },
        },
})


#container file for noise output etc.
noise_output_file = os.path.join(savefolder, 'ViSAPy_noise.h5')

setup_time = time()-tic


################################################################################
## MAIN SIMULATION PROCEDURE
################################################################################

tic = time()

################################################################################
## Step 1:  Estimate PSD and covariance between channels, here using
##          an experimental dataset.
##          
if not os.path.isfile(noise_output_file):
    if RANK == 0:
        noise_features = NoiseFeatures(**noiseFeaturesParameters)
        psd = noise_features.psd
        C = noise_features.C    
    else:
        psd = None
        C = None
    psd = COMM.bcast(psd, root=0)
    C = COMM.bcast(C, root=0)      


################################################################################
## Step 2:  Generate synthetic noise with PSD and covariance channels extracted
##          using class NoiseFeatures, preserving the overall amplitude.
##          We choose to save directly to file, as it will be used in
##          later steps
##          
    noise_generator = CorrelatedNoise(psd=psd,
                                             C=C,
                                             amplitude_scaling=1.,
                                             savefolder=savefolder,
                                             **logBumpParameters)
    noise = noise_generator.correlated_noise(T = cellParameters['tstop'])
    #file object containing extracellular noise and related data
    if RANK == 0:
        f = h5py.File(noise_output_file)
        f['data'] = noise


################################################################################
## Step 3:  Create a rate expectation envelope lambda_t for generating
##          non-stationary Poisson spike trains
##          
    #band-pass filter mean noise before non-stat Poisson generation
    b, a = butter(N=2, Wn=np.array([1., 25.]) / nyquist, btype='pass')
    #compute lambda function, use signal averaged over space. It will be
    #normalized by the ViSAPy.NonStationaryPoisson instance later.
    lambda_t = filtfilt(b, a, noise.mean(axis=0))
    if RANK == 0:
        f['lambda_t'] = lambda_t
        f.close()
else:
    #file exists, so we make some attempts at loading the non-stationarity
    f = h5py.File(noise_output_file)
    lambda_t = f['lambda_t'][()]
    f.close()

noise_time = time() - tic
COMM.Barrier()

################################################################################
## Step 4:  Run network simulation, generating a pool of synaptic activation
##          times used by the ViSAPy.Testdata instance
##          
tic = time()

#if database files exist, skip regenerating spike events
if not os.path.isfile(os.path.join(savefolder, 'SpTimesEx.db')) \
    and not os.path.isfile(os.path.join(savefolder, 'SpTimesIn.db')):
    #create an instance of our network
    networkInstance = ExternalNoiseRingNetwork(lambda_t=lambda_t,
                                **{k : v for k, v in list(networkParameters.items()) +
                                   list(ExternalNoiseRingNetworkParameters.items())})
    networkInstance.run()
    networkInstance.get_results()
    networkInstance.process_gdf_files()
else:
    #create instance of parent RingNetwork 
    networkInstance = RingNetwork(**networkParameters)

network_time = time() - tic

################################################################################
## Step 5:  Fix seed and set up Testdata object, generating a model cell
##          population, find and distribute synapse inputs with spiketrains from
##          network, run simulations for extracellular potentials,
##          collect data and generate final benchmark data
##

tic = time()

#set some seeds AFTER network sim, may want noise and spiking to be different,
#but populations to be equal. We are not explicitly setting the seed for NEST. 
np.random.seed(POPULATIONSEED)

#Create BenchmarkData object
benchmark_data = BenchmarkDataRing(
    cellParameters = cellParameters,
    morphologies = morphologies,
    templatefiles = templatefiles,
    defaultrotation = defaultrotation,
    simulationParameters = simulationParameters,
    populationParameters = populationParameters,
    electrodeParameters = electrodeParameters,
    noiseFile = noise_output_file,
    filters = filters,
    savefolder = savefolder,
    default_h5_file = 'lfp_cell_%.3i.h5',
    nPCA = 2,
    TEMPLATELEN = 100,
    TEMPLATEOFFS = 0.3,
    spikethreshold = 3.,
    networkInstance = networkInstance,
    synapseParametersEx = synparams_AMPA,
    synapseParametersIn = synparams_GABA_A,
    driftParameters = driftParameters)
print('setup ok!')
#run simulations and gather results
benchmark_data.run()
benchmark_data.collect_data()

bench_time = time() - tic

################################################################################
## Step 6:  Plot simulation output to the default simulation output folder

tic = time()

#utilize plot methods provided by class plotBenchmarkData. We are
#removing a startup transient of 500 ms. 
myplot = plotBenchmarkData(benchmark_data, TRANSIENT=500.)
myplot.run()

plot_time = time() - tic


################################################################################
## print out some stats.

if RANK == 0:
    print('\nsimulation times:\n')
    print('setup: \t\t%.1f seconds' % setup_time)
    print('noise: \t\t%.1f seconds' % noise_time)
    print('network: \t%.1f seconds' % network_time)
    print('benchmark: \t%.1f seconds' % bench_time)
    print('plots: \t\t%.1f seconds\n' % plot_time)
