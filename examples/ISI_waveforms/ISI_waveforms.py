#!/usr/bin/env python
'''Test how interspike interval affect spike waveforms in terms of
spike amplitude and spike width

The script is set up as the other population scripts, but location
of the single cell is in origo, and we test only along x-axis'''

#import modules
import uuid
import urllib2
import zipfile
import numpy as np
import h5py
import os
import glob
#workaround for plots on cluster
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import filtfilt, butter, lfilter
import scipy.interpolate as si
import scipy.optimize as so
from time import time, asctime
import spike_sort
import ViSAPy
import neuron
from mpi4py import MPI

plt.rcdefaults()
plt.rcParams.update({
    'xtick.labelsize' : 11,
    'xtick.major.size': 5,
    'ytick.labelsize' : 11,
    'ytick.major.size': 5,
    'font.size' : 15,
    'axes.labelsize' : 15,
    'axes.titlesize' : 15,
    'legend.fontsize' : 14,
    'figure.subplot.wspace' : 0.4,
    'figure.subplot.hspace' : 0.4,
    'figure.subplot.left': 0.1,
})
smallfontsize=11
alphabet = 'abcdefghijklmnopqrstuvwxyz'


######## set random number generator seed ######################################
SEED = 123456
POPULATIONSEED = 123456
np.random.seed(SEED)


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


######## create unique output folder and copy simulation script ################
if RANK == 0:
    #savefolder = glob.glob('savedata_ISI_waveforms_*')[-1]
    string = asctime().split()
    savefolder = os.path.join(os.path.split(__file__)[0], 'savedata_ISI_waveforms_')
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
        u = urllib2.urlopen('http://senselab.med.yale.edu/ModelDB/' +
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
    'timeres_NEURON' : 2**-5,
    'timeres_python' : 2**-5,
    'tstartms' : 0.,
    'tstopms' : 4250.,
    'verbose' : False,
    'pt3d' : True,
}


#in this particular set up, each cell will use the same
#morphology and templatefile specification of Hay et al 2011.
morphologies = [
    'L5bPCmodelsEH/morphologies/cell1.asc',
]
templatefiles = [
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

populationParameters = {
    'POPULATION_SIZE' : 32,
    'radius' : 50,
    'killzone' : 25,
    'z_min' : -25,
    'z_max' : 175,
    'X' : np.array([ [0, 0,  0,  -40, -40, 0, 0],
                   [0, 0,  0,   40,  40, 0, 0]]),
    'Y' : np.array([ [0, 0, -50, -50, -50, 0, 0],
                   [0, 0,  0,    0,   0, 0, 0]]),
    'Z' : np.array([-np.inf, -50.01, -50, 0, 2000, 2000.01, np.inf]),
    'min_cell_interdist' : 1.,
}

#Recording electrode geometry, seven contacts
N = np.empty((7, 3))
for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts

electrodeParameters = {
    'x' : np.array([10, 50, 100, 25, 25, 25, 25]),
    'y' : np.array([0, 0, 0, 0, 0, 0, 0]),
    'z' : np.array([0, 0, 0, 100, 50, 0, -50]),
    'sigma' : 0.3,
    'r' : 7.5,
    'n' : 10,
    'N' : N, 
    'method' : 'som_as_point',
}
driftParameters = None


#synaptic parameters: AMPA - excitatory, GABA_A - inhibitory
synparams_AMPA = {         #Excitatory synapse parameters
    'e' : 0,           #reversal potential
    'syntype' : 'Exp2Syn',   #conductance based exponential synapse
    'tau1' : 1.,         #Time constant, rise
    'tau2' : 3.,         #Time constant, decay
    'weight' : 0.0125,   #Synaptic weight
    'section' : ['apic', 'dend'],
    'nPerArea' :  [475E-4, 475E-5], #mean +- std
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
    'simtime' :     cellParameters['tstopms']-cellParameters['tstartms'],
    'dt' :          cellParameters['timeres_NEURON'],
    'total_num_virtual_procs' : SIZE,
    'savefolder' :  savefolder,
    'label' :       'statPoisson',
    'to_file' :     True,
    'to_memory' :   False,
    'print_time' :  False,
    #class StationaryPoissonNetwork
    'NE' : 60000,
    'NI' : 20000,
    'frateE' : 10.0,
    'frateI' : 10.0
}


#nyquist frequency of simulation output
nyquist = 1000. / cellParameters['timeres_python'] / 2


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


#Noise parameters including noise covariance matrix
noiseParameters = None
noiseFeaturesParameters = None


# set the rotations
rotations = []
defaultrotation = {}


#container file for noise output etc.
noise_output_file = None


################################################################################
## MAIN
################################################################################

TIME = time()

#if database files exist, skip regenerating spike events
if not os.path.isfile(os.path.join(savefolder, 'SpTimesEx.db')) \
    and not os.path.isfile(os.path.join(savefolder, 'SpTimesIn.db')):
    networkInstance = ViSAPy.StationaryPoissonNetwork(**networkParameters)
    networkInstance.run()
    networkInstance.get_results()
    networkInstance.process_gdf_files()
else:
    networkInstance = ViSAPy.StationaryPoissonNetwork(**networkParameters)


#set some seeds AFTER network sim, want noise and spikes to be different,
#but populations to be equal!!!!!!
np.random.seed(POPULATIONSEED)

benchmark_data = ViSAPy.BenchmarkDataRing(
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
    driftParameters = driftParameters,
    #pick Poisson trains with flat probability
    randdist = np.random.rand,
    sigma_EX = populationParameters['POPULATION_SIZE']+1,
    sigma_IN = populationParameters['POPULATION_SIZE']+1,
)

#override the random locations and rotations
for i in xrange(benchmark_data.POPULATION_SIZE):
    benchmark_data.pop_soma_pos[i] = {
        'xpos' : 0,
        'ypos' : 0,
        'zpos' : 0,
    }
    benchmark_data.rotations[i] = {
        'z' : 0,
    }

#run the cell simulations, skip collect method
benchmark_data.run()


################################################################################
# Function declarations
################################################################################
    
def calc_spike_widths(LFP, tvec, threshold=0.5):
    '''
    calculate spike widths of all spike traces, defined at treshold which is a
    fraction of the min-max amplitude of each trace
    '''
    def calc_spike_width(trace, tvec, threshold):
        '''calculate the spike width of the negative phase at
        threshold of extracellular spike trace'''
        
        #find global minima    
        minind = np.where(trace == trace.min())[0]
        
        #offset trace with local minima prior to spike before calc. spikewidth
        trace -= trace.max()
        
        #assess threshold crossings which may occur several times
        inds = trace <= trace.min() * threshold
        
        #go backwards in time and check for consistensy
        for i in xrange(minind, 0, -1):
            #on first occurance of False, stop loop and set remaining as False 
            if inds[i] == False:
                inds[:i] = False
                break
        #go forward in time
        for i in xrange(minind, len(trace)):
            #on first occurance of False, stop loop and set remaining as False
            if inds[i] == False:
                inds[i:] = False
                break
        inds = np.where(inds)[0]
        
        #linear interpolation to find crossing of threshold
        x0 = np.array([tvec[inds[0]-1], tvec[inds[0]]])
        y0 = np.array([trace[inds[0]], trace[inds[0]-1]])
        
        f = si.interp1d(y0, x0)
        t0 = f(trace.min() * threshold)
        
        x1 = np.array([tvec[inds[-1]], tvec[inds[-1]+1]])
        y1 = np.array([trace[inds[-1]], trace[inds[-1]+1]])
        
        f = si.interp1d(y1, x1)
        t1 = f(trace.min() * threshold)
        
        spw = t1 - t0
        if spw <= 0:
            return np.nan
        else:
            return spw

    spike_widths = []
    for trace in LFP:
        try:
            spike_widths.append(calc_spike_width(trace.copy(), tvec, threshold))
        except:
            spike_widths.append(np.nan)

    return np.array(spike_widths)


def nonlinfun(x, xdata):
    return x[0]*np.log(xdata) + x[1]


def costfun(x, xdata, ydata):
    '''cost function to be minimized, return sum of abs difference'''
    #eval x for xdata-values
    out = nonlinfun(x, xdata)
    return abs(ydata-out).sum()

def optimize(xdata, ydata):
    methods = ['Nelder-Mead'] #['Nelder-Mead'] # , 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    
    for method in methods:
        xf = so.minimize(costfun, x0=np.array([.1, 1.]),
                     args=(xdata, ydata),
                     method=method,
                     options={'maxiter' : 1000, 'xtol': 1e-8, 'disp': True})
    return xf.x


def normalfun(x, xdata):
    mu = x[0]
    sigma = x[1]
    return 1 / np.sqrt(2*np.pi*sigma) * np.exp(-(xdata-mu)**2/(2*sigma**2))     
    

def plot_figure_06(features, sp_waves, cmap=plt.cm.coolwarm, TRANSIENT=500):
    '''plot the features from python spikesort'''
    
    from matplotlib.colors import LogNorm
    
    fig = plt.figure(figsize=(10,10))
    
    #feature plots
    data = features['data']
    rows = cols = features['data'].shape[1]
    
    #create a 4X4 grid for subplots on [0.1, 0.5], [0.1, 0.5]
    width = 0.925 / rows * 0.99
    height= 0.925 / rows * 0.99
    x = np.linspace(0.05, 0.975-width, rows)
    y = np.linspace(0.05, 0.975-height, rows)[::-1]

    #sorting array
    argsort = np.argsort(ISI)

    
    for i, j in np.ndindex((rows, rows)):
        if i < j:
            continue
        if i == j:
            ax = fig.add_axes([x[i], y[j], width, height])
            bins = np.linspace(data[:, i].min(), data[:, i].max(), 50)
            ax.hist(data[:, i], bins=bins,
                    histtype='stepfilled', alpha=1,
                    edgecolor='none', facecolor='gray')

            #draw normal function from mean and std
            [count, bins] = np.histogram(data[:, i], bins=bins)
            normal = normalfun([data[:, i].mean(),
                                data[:, i].std()], bins)
            #scale to histogram:
            normal /= normal.sum()
            normal *= count.sum()
            
            #superimpose normal function
            ax.plot(bins, normal, 'k', lw=1)
    
            ax.set_ylabel(features['names'][j])
        else:
            ax = fig.add_axes([x[i], y[j], width, height])
            sc = ax.scatter(data[argsort, i], data[argsort, j],
                    marker='o',
                    c=ISI[argsort],
                    norm=LogNorm(),
                    cmap = plt.cm.get_cmap(cmap, 51),
                    edgecolors='none',
                    s=5, alpha=1, rasterized=True)
            
    
        if j == 0:
            ax.set_title(features['names'][i])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.axis(ax.axis('tight'))
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
                
        if j == 0 and i == 0:        
            ax.text(-0.3, 1.0, 'b',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax.transAxes)




    #plot extracted and aligned waveforms,
    tvec = sp_waves['time']

    vlim = abs(sp_waves['data']).max()
    scale = 2.**np.round(np.log2(vlim))
        
    yticks = []
    yticklabels = []
    for i in xrange(4):
        yticks.append(-i*scale)
        yticklabels.append('ch. %i' % (i+1))

    ax1 = fig.add_axes([0.05, 0.05, 0.2, 0.5])
    for i in xrange(4):
        #create a line-collection
        zips = []
        for j, x in enumerate(sp_waves['data'][:, argsort, i].T):
            zips.append(zip(tvec, x - i*scale))
        linecollection = LineCollection(zips,
                                        linewidths=(1),
                                        cmap = plt.cm.get_cmap(cmap, 51),
                                        rasterized=True,
                                        norm=LogNorm(),
                                        alpha=1,
                                        clip_on=False)
        linecollection.set_array(ISI[argsort])
        ax1.add_collection(linecollection)
    
    axis = ax1.axis('tight')
    ax1.axis(axis)
    ax1.set_ylim(axis[2]-0.05*scale, axis[3])
    for loc, spine in ax1.spines.iteritems():
        if loc in ['right', 'top',]:
            spine.set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_title('waveforms')
    ax1.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax1.plot([tvec[-1], tvec[-1]], [axis[2],axis[2]+scale],
        lw=4, color='k', clip_on=False)
    ax1.text(tvec[-1]*1.03, axis[2], '%.2f mV' % scale,
             fontsize=smallfontsize,
             ha='left', va='bottom', rotation='vertical'
             )
    
    ax1.text(-0.1, 1.0, 'a',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax1.transAxes)


    #colorbar for scatters and lines
    cax = fig.add_axes([0.27, 0.05, 0.015, 0.3])
    cax.set_rasterization_zorder(1)
    ticks = [5, 10, 20, 50, 100, 200, 500, 1000]
    axcb = fig.colorbar(linecollection, cax=cax, ticks=ticks)
    axcb.ax.set_visible(True)
    axcb.ax.set_yticklabels(ticks)
    axcb.set_label('ISI (ms)')


    return fig




def plot_figure_05(cells, benchmark_data, cmap=plt.cm.coolwarm, TRANSIENT=500.):
    '''Plot some traces'n'stuff'''
    
    from matplotlib.colors import LogNorm
 
    TEMPLATELEN = benchmark_data.TEMPLATELEN

    f = h5py.File(os.path.join(benchmark_data.savefolder, 'testISIshapes.h5'))
    amplitudes_raw = f['amplitudes_raw'].value
    amplitudes_flt = f['amplitudes_flt'].value
    templatesRaw = f['templatesRaw'].value
    templatesFlt = f['templatesFlt'].value
    ISI = f['ISI'].value
    concAPtemplates = f['APtemplates'].value
    AP_amplitudes = f['AP_amplitudes'].value
    AP_widths = f['AP_widths'].value
    f.close()
    

    #sorting array
    argsort = np.argsort(ISI)

    #plot some LFP-traces for a single cell
    fig = plt.figure(figsize=(10, 13))
    fig.subplots_adjust(wspace=0.4, hspace=0.3, bottom=0.05, top=0.95,
                        left=0.075, right=0.90)
    
    ax = fig.add_subplot(5, 3, 1)

    #find spikecount in total
    numspikes = []
    for cell in cells.values():
        numspikes = np.r_[numspikes, cell.AP_train.sum()]
    #pick an index with "average" rate
    cellkey = np.abs(numspikes - numspikes.mean()).argmin()

    ##plot some PSDs from somav and LFP    
    #choose one cell
    cell = cells[cellkey]
    cell.tvec = np.arange(cell.somav.size) * cell.timeres_python
    inds = np.where((cell.tvec >= TRANSIENT) & (cell.tvec <= TRANSIENT+500))[0]
    somav = cell.somav[inds]
    somav -= somav.min()
    somav /= somav.max()
    traces = somav
    xmins = []
    for j in xrange(3):
        x = cell.LFP[j, inds]
        xmin = x.min()
        xmins.append(xmin)
        x /= -xmin
        x -= 1.5*j + 0.5
        traces = np.c_[traces, x]
    
    #print traces.shape
    traces = traces.T
    
    ax.set_xlim(TRANSIENT, cell.tvec[inds][-1])
    ax.set_ylim(traces.min(), traces.max())
    
    line_segments = LineCollection([zip(cell.tvec[inds], x) \
                                    for x in traces],
        linewidths=(1),
        colors=('k'),
        linestyles='solid',
        rasterized=True,
        clip_on=False)

    ax.add_collection(line_segments)


    #scalebars
    ax.plot([cell.tvec[inds[-1]], cell.tvec[inds[-1]]],
            [1, 0], 'k', lw=4, clip_on=False)
    ax.text(cell.tvec[inds[-1]]*1.03, 0.,
                    r'%.0f' % (cell.somav[inds].max()-cell.somav[inds].min()) + '\n' + 'mV',
                    color='k', fontsize=smallfontsize, va='bottom', ha='left')
    for j in xrange(3):
        ax.plot([cell.tvec[inds[-1]], cell.tvec[inds[-1]]],
                [-j*1.5-0.5, -j*1.5-1.5], 'k', lw=4, clip_on=False)
        ax.text(cell.tvec[inds[-1]]*1.03, -j*1.5-1.5,
                r'%.0f' % (abs(xmins[j]*1E3)) + '\n' + '$\mu$V', color='k',
                fontsize=smallfontsize,
                va='bottom',
                ha='left'
                )

    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top', 'left']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
    ax.set_yticks([0.0, -0.5, -2, -3.5])
    ax.set_yticklabels([r'$V_\mathrm{soma}$',
                        r'$\Phi_{x=10}$',
                        r'$\Phi_{x=50}$',
                        r'$\Phi_{x=100}$'])
    ax.axis(ax.axis('tight'))
    
    ax.text(-0.2, 1.0, 'a',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)



    #raise Exception

    PSDs = np.array([])
    #psd of somav
    psd, freqs = plt.mlab.psd(cell.somav[cell.tvec > TRANSIENT]-cell.somav[cell.tvec > TRANSIENT].mean(),
                              NFFT=2**15+2**14, noverlap=int((2**15+2**14)*3./4), #5096,
                              Fs=1E3/np.diff(cell.tvec)[-1])
    PSDs = np.r_[PSDs, psd[1:]]
    #psds of LFPs
    for j in xrange(3):
        psd, freqs = plt.mlab.psd(cell.LFP[j, cell.tvec > TRANSIENT]-cell.LFP[j, cell.tvec > TRANSIENT].mean(),
                                  NFFT=2**15+2**14, noverlap=int((2**15+2**14)*3./4), #NFFT=5096,
                              Fs=1E3/np.diff(cell.tvec)[-1])
        PSDs = np.c_[PSDs, psd[1:]]
    
    PSDs = PSDs.T
    
    #create axes object
    ax = fig.add_subplot(5, 3, 2)
    ax.set_xlim(freqs[1], freqs[-1])
    ax.set_ylim(PSDs[1:].min(),
                PSDs[1:].max())

    #create line collection
    line_segments = LineCollection([zip(freqs[1:], x) \
                                    for x in PSDs],
        linewidths=(1),
        colors=('k'),
        linestyles='solid',
        rasterized=True)

    ax.add_collection(line_segments)
    plt.sci(line_segments) # This allows interactive changing of the colormap.
    
    ax.loglog()
    
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'$f$ (Hz)', labelpad=0.1)
    ax.set_title(r'PSD (mV$^2$/Hz)')
    ax.axis(ax.axis('tight'))
    ax.grid('on')

    ax.text(-0.2, 1.0, 'b',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)



    #plot histogram over ISI
    ax = fig.add_subplot(5, 3, 3)
    bins = 10**np.linspace(np.log10(1), np.log10(1E3), 100)
    ax.hist(ISI, bins=bins,
            color='gray',
            histtype='stepfilled',
            linewidth=0)
    ax.semilogx()
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.axis(ax.axis('tight'))
    ax.set_xlim([bins.min(), bins.max()])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('count (-)', labelpad=0)
    ax.set_xlabel('ISI (ms)', labelpad=0.1)
    ax.set_title('ISI distr. %i APs' % ISI.size)

    ax.text(-0.2, 1.0, 'c',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)




    #plot nonfiltered spike waveforms    
    ax = fig.add_subplot(5,3,4)
    
    line_segments = LineCollection([zip(np.arange(TEMPLATELEN), x) \
                    for x in concAPtemplates[argsort, TEMPLATELEN*0:TEMPLATELEN*1]],
        linewidths=(1),
        linestyles='solid',
        norm=LogNorm(),
        cmap = plt.cm.get_cmap(cmap, 51),
        rasterized=True)
    line_segments.set_array(ISI[argsort])
    ax.add_collection(line_segments)
    
    ax.axis(ax.axis('tight'))
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(r'$V_\mathrm{soma}$ (mV)', labelpad=0)
    ax.set_xlabel('samples (-)', labelpad=0.1)

    ax.text(-0.2, 1.0, 'd',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)
    
    
    
    
    #plot AP amplitudes vs widths
    ax = fig.add_subplot(5,3,5)
    
    #mask out invalid widths
    mask = True - np.isnan(AP_widths)
    
    sc = ax.scatter(AP_widths[mask[argsort]], AP_amplitudes[mask[argsort]], marker='o',
                    edgecolors='none', s=5,
                c=ISI[mask[argsort]], norm=LogNorm(),
                cmap=plt.cm.get_cmap(cmap, 51), #bins.size)
                alpha=1, clip_on=False, rasterized=True)
    ax.set_ylabel('AP ampl. (mV)', labelpad=0)
    ax.set_xlabel('AP width (ms)', labelpad=0.1)
    
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim([AP_widths[mask].min(), AP_widths[mask].max()])
    ax.set_ylim([AP_amplitudes[mask].min(), AP_amplitudes[mask].max()])
    
    ax.text(-0.2, 1.0, 'e',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)
    


    
    
    
    ax = fig.add_subplot(5,3,6)
    
    #set lims
    ax.set_xlim(0, TEMPLATELEN)
    ax.set_ylim(templatesRaw[np.isfinite(templatesRaw[:, 0]), :][:, TEMPLATELEN*0:TEMPLATELEN*1].min(),
                templatesRaw[np.isfinite(templatesRaw[:, 0]), :][:, TEMPLATELEN*0:TEMPLATELEN*1].max())
    
    #create linecollections
    line_segments = LineCollection([zip(np.arange(TEMPLATELEN), x) \
                    for x in templatesRaw[argsort, TEMPLATELEN*0:TEMPLATELEN*1]],
        linewidths=(1),
        linestyles='solid',
        norm=LogNorm(),
        cmap = plt.cm.get_cmap(cmap, 51),
        rasterized=True)
    
    line_segments.set_array(ISI[argsort])
    ax.add_collection(line_segments)
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel(r'$\Phi_{x=10}$ (mV)', labelpad=0)
    ax.set_xlabel('samples (-)', labelpad=0.1)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.015
    cax = fig.add_axes(rect)
    cax.set_rasterization_zorder(1)
    axcb = fig.colorbar(line_segments, cax=cax)
    axcb.ax.set_visible(True)
    ticks = [5, 10, 20, 50, 100, 200, 500, 1000]
    axcb.set_ticks(ticks)
    axcb.set_ticklabels(ticks)
    axcb.set_label('ISI (ms)', va='center', ha='center', labelpad=0)


    ax.text(-0.2, 1.0, 'f',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)

    
    
    #plot FILTERED spike waveforms
    ax = fig.add_subplot(5,3,7)
    
    #set lims
    ax.set_xlim(0, TEMPLATELEN)
    ax.set_ylim(templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*0:TEMPLATELEN*1].min(),
                templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*0:TEMPLATELEN*1].max())
    
    #create linecollections
    line_segments = LineCollection([zip(np.arange(TEMPLATELEN), x) \
                    for x in templatesFlt[argsort, TEMPLATELEN*0:TEMPLATELEN*1]],
        linewidths=(1),
        linestyles='solid',
        norm=LogNorm(),
        cmap = plt.cm.get_cmap(cmap, 51),
        rasterized=True)
    line_segments.set_array(ISI[argsort])
    ax.add_collection(line_segments)
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('samples (-)', labelpad=0.1)
    ax.set_ylabel(r'$\Phi_{x=10}$ (mV)', labelpad=0)    


    ax.text(-0.2, 1.0, 'g',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)


    ax = fig.add_subplot(5,3,8)
    
    #set lims
    ax.set_xlim(0, TEMPLATELEN)
    ax.set_ylim(templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*1:TEMPLATELEN*2].min(),
                templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*1:TEMPLATELEN*2].max())
    
    #create linecollections
    line_segments = LineCollection([zip(np.arange(TEMPLATELEN), x) \
                    for x in templatesFlt[argsort, TEMPLATELEN*1:TEMPLATELEN*2]],
        linewidths=(1),
        linestyles='solid',
        norm=LogNorm(),
        cmap = plt.cm.get_cmap(cmap, 51),
        rasterized=True)
    line_segments.set_array(ISI[argsort])
    ax.add_collection(line_segments)
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('samples (-)', labelpad=0.1)
    ax.set_ylabel(r'$\Phi_{x=50}$ (mV)', labelpad=0)    


    ax.text(-0.2, 1.0, 'h',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)
    

    ax = fig.add_subplot(5,3,9)

    #set lims
    ax.set_xlim(0, TEMPLATELEN)
    ax.set_ylim(templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*2:TEMPLATELEN*3].min(),
                templatesFlt[np.isfinite(templatesFlt[:, 0]), :][:, TEMPLATELEN*2:TEMPLATELEN*3].max())

    #create linecollections
    line_segments = LineCollection([zip(np.arange(TEMPLATELEN), x) \
                    for x in templatesFlt[argsort, TEMPLATELEN*2:TEMPLATELEN*3]],
        linewidths=(1),
        linestyles='solid',
        norm=LogNorm(),
        cmap = plt.cm.get_cmap(cmap, 51),
        rasterized=True)
    line_segments.set_array(ISI[argsort])
    ax.add_collection(line_segments)
    plt.sci(line_segments) # This allows interactive changing of the colormap.
    for loc, spine in ax.spines.iteritems():
        if loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('samples (-)', labelpad=0.1)
    ax.set_ylabel(r'$\Phi_{x=100}$ (mV)', labelpad=0)    

    ax.text(-0.2, 1.0, 'i',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=18, fontweight='demibold',
        transform=ax.transAxes)    
    
    
    for i in xrange(3):
        ax = fig.add_subplot(5, 3, i+10)

        sc = ax.scatter(spikewidths_flt[i, argsort], amplitudes_flt[i, argsort], marker='o',
                    edgecolors='none', s=5,
                    c=ISI[argsort],
                    norm=LogNorm(),
                    cmap = plt.cm.get_cmap(cmap, 51),
                    label='filtered', alpha=1, clip_on=False,
                    rasterized=True)
        
        if i == 0: ax.set_ylabel('amplitude (mV)', labelpad=0)
        ax.set_xlabel('width (ms)', labelpad=0.1)
        
        ax.set_xlim([spikewidths_flt[i, :].min(), spikewidths_flt[i, :].max()])
        ax.set_ylim([amplitudes_flt[i, :].min(), amplitudes_flt[i, :].max()])
        
        for loc, spine in ax.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        
        ax.text(-0.2, 1.0, alphabet[i+9],
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)
    

    for i in xrange(3):
        ax = fig.add_subplot(5 , 3, i+13)
        
        sc = ax.scatter(ISI, amplitudes_flt[i, :], marker='o',
                    edgecolors='none', s=5,
                    facecolors='k',
                    label='filtered', alpha=1, clip_on=False,
                    rasterized=True)
        
        if i == 0: ax.set_ylabel('amplitude (mV)', labelpad=0)
        ax.set_xlabel('ISI (ms)', labelpad=0.1)
        ax.set_xlim([ISI.min(), ISI.max()])
        ax.set_ylim([amplitudes_flt[i, :].min(), amplitudes_flt[i, :].max()])
        ax.semilogx()
                
        
        for loc, spine in ax.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        
        ax.text(-0.2, 1.0, alphabet[i+12],
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)



    return fig



################################################################################
# Perform some preprocessing for plots
################################################################################

if RANK == 0:
    cellindices = np.arange(benchmark_data.POPULATION_SIZE)
    cells = benchmark_data.read_lfp_cell_files(cellindices)
    print 'cells ok'

    #recalculate AP_trains using the slope of somav
    AP_threshold = -30
    for cellkey, cell in cells.iteritems():
        setattr(cell, 'AP_train',
                benchmark_data.return_spiketrains(v=cell.somav, v_t=AP_threshold, TRANSIENT=500.))
    
    #do some filtering of the LFP traces of each cell
    for i, cell in cells.iteritems():
        LFP_flt = cell.LFP.value.copy()
        for fltr in benchmark_data.filters:
            LFP_flt = fltr['filterFun'](fltr['b'], fltr['a'], LFP_flt)
        setattr(cell, 'LFP_flt', LFP_flt)

    
    #Will reformat template waveform to correspond with ISI, so
    #for each cell in population we extract the SECOND etc spike waveform.
    AllTemplatesRaw = {}
    AllTemplatesFlt = {}
    APtemplates = {}
    spWavesTetrode = {}

    #Contain ISI from AP_trains
    ISI = []
    for cellkey, cell in cells.iteritems():
        #use spike_sort module to extract, upsample, and align spike waveforms

        #sample window in ms:
        sp_win = ((np.array([0, benchmark_data.TEMPLATELEN]) -
                    benchmark_data.TEMPLATELEN*benchmark_data.TEMPLATEOFFS)
                * cellParameters['timeres_python']).tolist()

        #raw LFP per cell
        spRaw = {
            'data' : cell.LFP,
            'FS' : 1E3 / cellParameters['timeres_python'],
            'n_contacts' : 3
        }
        
        #filtered LFP per cell
        spFlt = {
            'data' : cell.LFP_flt,
            'FS' : 1E3 / cellParameters['timeres_python'],
            'n_contacts' : 3
        }
        
        spAPs = {
            'data' : cell.somav.reshape((1, -1)),
            'FS' : 1E3 / cellParameters['timeres_python'],
            'n_contacts' : 1
        }
                
        #find spike events, and prune spikes near bounds
        OFF = int(benchmark_data.TEMPLATELEN*benchmark_data.TEMPLATEOFFS)
        APs = np.where(cell.AP_train == 1)[0] #*cellParameters['timeres_python']
        
        #collect ISI        
        APs = APs[(APs > OFF) & (APs < cell.AP_train.size - (benchmark_data.TEMPLATELEN-OFF))].astype(float)
        APs *= cellParameters['timeres_python']    
        ISI = np.r_[ISI, np.diff(APs)]
                
        
        sptAPs = {
            'data' : APs[1:], #discard first waveform, ISI not known for these
            'contact' : 0,
            'thresh' : 0,
        }        
        
        #allow independent alignment of waveforms
        sptRaw = sptAPs.copy()
        sptFlt = sptAPs.copy()
        
    
        #aligned spike times to min of channel 0 for raw and filtered traces
        sptRaw = spike_sort.extract.align_spikes(spRaw, sptRaw, sp_win,
                                            contact=0, remove=False, type="min")
        sptFlt = spike_sort.extract.align_spikes(spFlt, sptFlt, sp_win,
                                            contact=0, remove=False, type="min")
        sptAPs = spike_sort.extract.align_spikes(spAPs, sptAPs, sp_win,
                                            contact=0, remove=False, type="max")
        
        #extract spike waveforms:
        spWavesRaw = spike_sort.extract.extract_spikes(spRaw, sptRaw, sp_win,
                                                       contacts=[0, 1, 2])
        spWavesFlt = spike_sort.extract.extract_spikes(spFlt, sptFlt, sp_win,
                                                       contacts=[0, 1, 2])
        spWavesAPs = spike_sort.extract.extract_spikes(spAPs, sptAPs, sp_win,
                                                       contacts=[0])
        
        #spikes from "tetrode"
        spWavesTetrode.update({cellkey : spike_sort.extract.extract_spikes(spFlt,
                                        sptFlt, sp_win, contacts=[3, 4, 5, 6])})
        
        
        #convert to 2D arrays, each row is 3 channels concatenated
        temp = []
        for i in xrange(spWavesRaw['data'].shape[1]):
            temp.append(spWavesRaw['data'][:, i, :].T.flatten())
        spWavesRaw = np.array(temp)
        temp = []
        for i in xrange(spWavesFlt['data'].shape[1]):
            temp.append(spWavesFlt['data'][:, i, :].T.flatten())
        spWavesFlt = np.array(temp)
        temp = []
        for i in xrange(spWavesAPs['data'].shape[1]):
            temp.append(spWavesAPs['data'][:, i, :].T.flatten())
        spWavesAPs = np.array(temp)
        
        #fill in values
        AllTemplatesRaw.update({cellkey : spWavesRaw})
        AllTemplatesFlt.update({cellkey : spWavesFlt})
        APtemplates.update({cellkey : spWavesAPs})
        
    #delete some variables
    del temp, spWavesRaw, spWavesFlt, sptRaw, sptFlt, spRaw, spFlt, sp_win


    #employ spike_sort to calculate PCs, first reformat structure
    sp_waves = {
        'time' : spWavesTetrode[0]['time'],
        #'data' : [],
        'FS' : spWavesTetrode[0]['FS']
    }
    for key, value in spWavesTetrode.items():
        if key == 0:
            sp_waves['data'] = value['data']
        else:
            sp_waves['data'] = np.r_['1', sp_waves['data'], value['data']]
    
    #compute features
    features = ViSAPy.plottestdata.fetPCA(sp_waves, ncomps=benchmark_data.nPCA)
    
    
    

               
    #concatenate the templates and basis functions for each cell object,
    #except first spike
    concTemplatesRaw = None
    concTemplatesFlt = None
    concAPtemplates = None
    channels = ['ch. 0', 'ch. 1', 'ch. 2']
    for i, cell in cells.iteritems():
        if i == 0:
            if AllTemplatesRaw[i].size > 0:
                concTemplatesRaw = AllTemplatesRaw[i]
                concTemplatesFlt = AllTemplatesFlt[i]
                concAPtemplates = APtemplates[i]
        else:
            if AllTemplatesRaw[i].size > 0:
                concTemplatesRaw = np.r_[concTemplatesRaw,
                                         AllTemplatesRaw[i]]
                concTemplatesFlt = np.r_[concTemplatesFlt,
                                         AllTemplatesFlt[i]]
                concAPtemplates = np.r_[concAPtemplates, APtemplates[i]]


    del AllTemplatesRaw, AllTemplatesFlt, APtemplates
    print 'concTemplatesRaw, concTemplatesFlt, concAPtemplates ok'
    
    
    
    
    #extract the amplitudes between min and max of the templates in
    #the three contacts
    amplitudes_raw = np.empty((len(channels), concTemplatesRaw.shape[0]))
    amplitudes_flt = np.empty((len(channels), concTemplatesFlt.shape[0]))
    
    TEMPLATELEN = benchmark_data.TEMPLATELEN
    for j in xrange(3):
        i = 0
        for x in concTemplatesRaw[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]:
            amplitudes_raw[j, i] = x.max() - x.min()
            i += 1
    
    for j in xrange(3):
        i = 0
        for x in concTemplatesFlt[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]:
            amplitudes_flt[j, i] = x.max() - x.min()
            i += 1
    print 'amplitudes ok'
    
    
    #calc spikewidths for each contact and raw, filtered shapes
    spikewidths_raw = []
    spikewidths_flt = []
    cell.tvec = np.arange(cell.somav.size)*cell.timeres_python
    tvec = cell.tvec[:TEMPLATELEN]
    for j in xrange(3):
        LFP = concTemplatesRaw[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]
        spikewidths_raw.append(calc_spike_widths(LFP, tvec, threshold=0.5))
    
        LFP = concTemplatesFlt[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]
        spikewidths_flt.append(calc_spike_widths(LFP, tvec, threshold=0.5))
    
    spikewidths_raw = np.array(spikewidths_raw)
    spikewidths_flt = np.array(spikewidths_flt)
    
    #spike width and amplitude of APs
    AP_widths = calc_spike_widths(-concAPtemplates, tvec, threshold=0.5)
    AP_amplitudes = concAPtemplates.max(axis=1) - concAPtemplates.min(axis=1)
    print 'spikewidths_*, AP_widths, AP_amplitudes ok'
    
    
    #calculate projections according to Fee et al. 1996 (eq. 8)
    projectionFeeRaw = None
    projectionFeeFlt = None
    data = concTemplatesRaw
    for j in xrange(3):
        templates = data[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]
        notnans = np.isfinite(templates[:, 0])
        V_Smean = templates[notnans, ][ISI[notnans, ] <= 10, ].mean(axis=0)
        V_Lmean = templates[notnans, ][ISI[notnans, ] >= 100, ].mean(axis=0)
        dV_LS = V_Lmean - V_Smean
        if j == 0:
            projectionFeeRaw = np.dot((templates - V_Lmean), dV_LS) / \
                             np.dot(dV_LS, dV_LS)
        else:
            projectionFeeRaw = np.r_[projectionFeeRaw,
                                     np.dot((templates - V_Lmean), dV_LS) / \
                                     np.dot(dV_LS, dV_LS)]
    
    data = concTemplatesFlt
    for j in xrange(3):
        templates = data[:, TEMPLATELEN*j:TEMPLATELEN*(j+1)]
        notnans = np.isfinite(templates[:, 0])
        V_Smean = templates[notnans, ][ISI[notnans, ] <= 10, ].mean(axis=0)
        V_Lmean = templates[notnans, ][ISI[notnans, ] >= 100, ].mean(axis=0)
        dV_LS = V_Lmean - V_Smean

        if j == 0:
            projectionFeeFlt = np.dot((templates - V_Lmean), dV_LS) / \
                             np.dot(dV_LS, dV_LS)
        else:
            projectionFeeFlt = np.r_[projectionFeeFlt,
                                     np.dot((templates - V_Lmean), dV_LS) / \
                                     np.dot(dV_LS, dV_LS)]
    
    projectionFeeRaw = projectionFeeRaw.reshape(3, -1)
    projectionFeeFlt = projectionFeeFlt.reshape(3, -1)
    
    #save additional sim results
    f = h5py.File(os.path.join(benchmark_data.savefolder, 'testISIshapes.h5'))
    try: f['ISI'] = ISI
    except:
        del f['ISI']
        f['ISI'] = ISI
    
    try: f['templatesRaw'] = concTemplatesRaw
    except:
        del f['templatesRaw']
        f['templatesRaw'] = concTemplatesRaw
        
    try: f['templatesFlt'] = concTemplatesFlt
    except:
        del f['templatesFlt']
        f['templatesFlt'] = concTemplatesFlt
    
    try: f['amplitudes_raw'] = amplitudes_raw
    except:
        del f['amplitudes_raw']
        f['amplitudes_raw'] = amplitudes_raw
    
    try: f['amplitudes_flt'] = amplitudes_flt
    except:
        del f['amplitudes_flt']
        f['amplitudes_flt'] = amplitudes_flt
    
    try: f['spikewidths_raw'] = spikewidths_raw
    except:
        del f['spikewidths_raw']
        f['spikewidths_raw'] = spikewidths_raw
    
    try: f['spikewidths_flt'] = spikewidths_flt
    except:
        del f['spikewidths_flt']
        f['spikewidths_flt'] = spikewidths_flt
    
    try: f['projectionFeeRaw'] = projectionFeeRaw
    except:
        del f['projectionFeeRaw']
        f['projectionFeeRaw'] = projectionFeeRaw
    
    try: f['projectionFeeFlt'] = projectionFeeFlt
    except:
        del f['projectionFeeFlt']
        f['projectionFeeFlt'] = projectionFeeFlt
    
    try: f['APtemplates'] = concAPtemplates
    except:
        del f['APtemplates']
        f['APtemplates'] = concAPtemplates
    
    try: f['AP_widths'] = AP_widths
    except:
        del f['AP_widths']
        f['AP_widths'] = AP_widths

    try: f['AP_amplitudes'] = AP_amplitudes
    except:
        del f['AP_amplitudes']
        f['AP_amplitudes'] = AP_amplitudes
    
    f.close()


    
    #############################################################
    # Plot some STUFF
    #############################################################
    
    fig = plot_figure_05(cells, benchmark_data, cmap=plt.cm.coolwarm)
    fig.savefig(os.path.join(benchmark_data.savefolder, 'figure_05.pdf'), dpi=150)


    fig = plot_figure_06(features, sp_waves, cmap=plt.cm.coolwarm)
    fig.savefig(os.path.join(savefolder, 'figure_06.pdf'), dpi=150)
    

    fig, ax = plt.subplots(1, figsize=(10, 10))
    for i, cell in cells.items():
        ax.plot(cell.somav+i*100, lw=0.5)
    plt.axis('tight')
    fig.savefig(os.path.join(benchmark_data.savefolder, 'somatraces.pdf'), dpi=300)
    plt.show()

