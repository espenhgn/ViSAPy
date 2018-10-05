#!/usr/bin/env python
'''Class definition for plotting benchmark data generated with ViSAPy'''

import os
import sys
if sys.version < '3':
    if not os.environ.has_key('DISPLAY'):
        import matplotlib
        matplotlib.use('Agg')
else:
    if 'DISPLAY' in os.environ.keys():
        import matplotlib
        matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from ViSAPy import GDF, fcorr
from mpi4py import MPI
import spike_sort
import h5py
import sys
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize


plt.rcdefaults()
plt.rcParams.update({
    'xtick.labelsize' : 8,
    'xtick.major.size': 5,
    'ytick.labelsize' : 8,
    'ytick.major.size': 5,
    'font.size' : 8,
    'axes.labelsize' : 8,
    'axes.titlesize' : 8,
    'legend.fontsize' : 8,
    'figure.subplot.wspace' : 0.4,
    'figure.subplot.hspace' : 0.4,
    'figure.subplot.left': 0.1,
})
smallfontsize=11


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()



######### Function and Class definitions #######################################

def xcorr(x, y, maxlags, normed=False, remove_zero=False):
    """
    Faster cross correlation gives central 2*maxlags+1 portion of
    scipy.correlate

    z = xcorr(x,y): if there is a peak before zero, x goes first, then y
                                  peak after zero, y goes first, then x
    """
    z = np.empty(2*maxlags+1)
    for i in range(-maxlags, 0):
        z[i+maxlags] = np.dot(y[-i:], x[:i])
    for i in range(1, maxlags+1):
        z[i+maxlags] = np.dot(x[i:], y[:-i])
    z[maxlags] = np.dot(x, y)
    if normed:
        z /= np.linalg.norm(x) * np.linalg.norm(y)
    if remove_zero: z[maxlags] = 0
    return z


def normalfun(x, xdata):
    mu = x[0]
    sigma = x[1]
    return 1 / np.sqrt(2*np.pi*sigma) * np.exp(-(xdata-mu)**2/(2*sigma**2)) 


def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis'''
    for loc, spine in ax.spines.iteritems():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    

def get_colors(num=16, cmap=plt.cm.viridis):
    '''return a list of color tuples to use in plots'''
    colors = []
    for i in range(num):
        i *= 256.
        if num > 1:
            i /= num - 1.
        else:
            i /= num
        colors.append(cmap(int(i)))
    return colors


def fetPCA(sp_waves, ncomps=2):
    '''
    Calculate principal components (PCs).
    
    Keyword arguments:
    ::

        spikes : dict
        ncomps : int, optional
            number of components to retain
    
    Returns:
    ::
    
        dict with entries
            data : np.array with principal components
            times : time vector of PCAs
            FS : sampling rate
            names : str, labels of signals
    '''

    data = sp_waves['data']
    n_channels = data.shape[2]
    pcas = np.zeros((n_channels*ncomps, data.shape[1]))
    
    for ch in range(data.shape[2]):
        _, _, pcas[ch::data.shape[2], ] = spike_sort.features.PCA(
                                                    data[:, :, ch], ncomps)

    
    names = ["ch.%d:PC%d" % (j+1, i+1)
             for i in range(ncomps) for j in range(n_channels)]

    
    outp = {}
    outp['data'] = pcas.T
    outp['time'] = sp_waves['time']
    outp['FS'] = sp_waves['FS']
    outp['names'] = names
    
    return outp


class plotBenchmarkData(object):
    
    def __init__(self, testdInst, cmap=plt.cm.viridis, TRANSIENT=500.):
        '''
        Plotting methods
        
        Keyword arguments:
        ::
            
            testdInst : ViSAPy.BenchmarkData object
            cmap : matplotlib.colors.LinearSegmentedColormap
            TRANSIENT : float, startup transient 
        
        '''
        #set some attributes
        self.testdInst = testdInst
        self.cmap = cmap
        self.TRANSIENT = TRANSIENT
        self.savefolder = self.testdInst.savefolder
        
        
        #using these colors and alphas:        
        self.colors = get_colors(self.testdInst.POPULATION_SIZE, cmap=cmap)
        
        self.alphas = np.ones(self.testdInst.POPULATION_SIZE)


        #using these colors and alphas:        
        self.electrodeColors = []
        for i in range(self.testdInst.electrodeParameters['x'].size):
            self.electrodeColors.append('k')


    def run(self, cellindices=None, ):
        '''
        Main method for plotting all output data
        
        Keyword arguments:
        ::
        
            cellindices : None, or np.ndarray with indices of cells for plots
            TRANSIENT : float, duration of startup transient
        
        '''
        if RANK == 0:
            #using cellindices throughout
            if cellindices is None:
                cellindices = np.arange(self.testdInst.POPULATION_SIZE)
            
            #get the cells
            cells = self.testdInst.read_lfp_cell_files(cellindices)
            print('cells ok')

            #remove vmem, imem if they exist, they are not needed here
            for cell in cells.itervalues():
                if hasattr(cell, 'vmem'):
                    del cell.vmem
                if hasattr(cell, 'imem'):
                    del cell.imem

            
            try:
                fig = self.testdInst.networkInstance.raster_plots(xlim=[500., 1500.])
                fig.savefig(os.path.join(self.savefolder, 'nestsimrasters.pdf'),
                            dpi=150)
                print('raster_plots() ok')
                plt.close(fig)
            except:
                print('raster_plots() not ok', sys.exc_info())
            
            
            #calculate some AP_trains
            for cell in cells.itervalues():
                setattr(cell, 'AP_train',
                        self.testdInst.return_spiketrains(cell.somav))

            try:
                fig = self.plot_figure_02(cells=cells)
                fig.savefig(os.path.join(self.savefolder, 'figure_02.pdf'),
                            dpi=150)
                print('plot_figure_02() ok')
                plt.close(fig)
            except:
                print('plot_figure_02() not ok', sys.exc_info())


            try:
                fig = self.plot_figure_07()
                fig.savefig(os.path.join(self.savefolder, 'figure_07.pdf'),
                            dpi=150)
                print('plot_figure_07() ok')
                plt.close(fig)
            except:
                print('plot_figure_07() not ok', sys.exc_info())


            try:
                fig = self.plot_figure_08(cells)
                fig.savefig(os.path.join(self.savefolder,
                                         'figure_08.pdf'), dpi=150)
                print('plot_figure_08() ok')
                plt.close(fig)
            except:
                print('plot_figure_08() not ok', sys.exc_info())


            try:
                fig = self.plot_figure_09(cells)
                fig.savefig(os.path.join(self.savefolder,
                                         'figure_09.pdf'), dpi=150)
                print('plot_figure_09() ok')
                plt.close(fig)
            except:
                print('plot_figure_09() not ok', sys.exc_info())

            
            #try:
            #    fig = self.plot_figure_10(cellindices)
            #    fig.savefig(self.savefolder + '/figure_10.pdf', dpi=150)
            #    print 'plot_figure_10() ok'
            #    plt.close(fig)
            #except:
            #    print 'plot_figure_10() not ok', sys.exc_info()


            try:
                fig = self.plot_figure_11(cells=cells)
                fig.savefig(os.path.join(self.savefolder, 'figure_11.pdf'), dpi=150)
                print('plot_figure_11() ok')
                plt.close(fig)
            except:
                print('plot_figure_11() not ok', sys.exc_info())

                    
            try:
                if len(cellindices) > 32:
                    fig = self.plot_figure_12(num_units=32)
                else:
                    fig = self.plot_figure_12(cellindices)
                fig.savefig(os.path.join(self.savefolder, 'figure_12.pdf'),
                            dpi=150)
                print('plot_figure_12() ok')
                plt.close(fig)
            except:
                print('plot_figure_12() not ok')
            
                
            try:
                fig = self.plot_population()
                fig.savefig(os.path.join(self.savefolder, 'population.pdf'),
                            dpi=150)
                print('plotPopulation() ok')
                plt.close(fig)
            except:
                print('plotPopulation() not ok', sys.exc_info())


        #resync MPI ranks
        COMM.Barrier()



    def plot_figure_02(self, cells=None, tstop=1500.):

        def schemPyr(cell, scale=1.):
            '''
            schematic representation of pyramidal cell
            that can be plotted using
            ax.plot3(), cell arg is LFPy.Cell instance
            
            returned argument is a x, y, z-tuple
            '''
            #scale 1:100
            x = np.array([-100., -20,  0,   0, -100,   0, 100,   0,  0,  20,
                          100,  20,  0,    0,  0, -20, -100])
            z = np.array([-50.,   0,  30, 500,  600, 500, 600, 500, 30,   0,
                          -50,   0,  0, -100,  0,   0,  -50])
            
            #shift midpoint of soma
            z -= 15
            
            #quite happy with the look of the cells, just rescale
            x *= scale
            z *= scale
            
            #offset using real 
            x += cell.somapos[0]
            z += cell.somapos[2]
            
            #depth
            y = np.zeros(x.size) + cell.somapos[1]
            
            return x, y, z
        
        def contPoint(radius=5, center=[0., 0.]):
            '''
            return coords of electrode contact point that can be
            plotted using ax.fill()
            
            returned arg is x, z-tuple
            '''
            
            theta = np.arange(22) / 20. * 2 * np.pi 
            
            x = radius * np.cos(theta)
            x += center[0]
            
            z = radius * np.sin(theta)
            z += center[1]
            
            return x, z
            
        def fancy_arrow(ax, x=np.random.rand(10),y=np.random.rand(10)):
            '''draw fancy arrow'''
            if len(x) > 2:
                ax.plot(x[:-1], y[:-1], 'k', lw=2,)
            if np.diff(y)[-1] != 0:
                head_width=0.01
                head_length=0.02
            else:
                head_width=0.02
                head_length=0.01
                
            ax.arrow(x[-2], y[-2], np.diff(x)[-1], np.diff(y)[-1],
                     head_width=head_width, head_length=head_length,
                     fc='k', ec='k', lw=2)
            
            return
            

        
        #get some cells
        if cells is None:
            cells = self.testdInst.read_lfp_cell_files()
        
        #load 1 s of recording
        if self.testdInst.cellParameters['tstop'] < tstop:
            tstop = self.testdInst.cellParameters['tstop']

        tinds = np.arange(self.TRANSIENT / self.testdInst.cellParameters['dt'],
                          tstop / self.testdInst.cellParameters['dt'] + 1,
                          dtype=int)
        tvec = tinds * self.testdInst.cellParameters['dt']

        
        #get some traces
        f = h5py.File(self.savefolder + '/ViSAPy_filterstep_0.h5',
                      'r')
        lfp_noisy = f['data'].value.T[:, tinds]
        f.close()
        
        #load the extracellular noise:
        f = h5py.File(os.path.join(self.savefolder, 'ViSAPy_noise.h5'))
        noise = f['data'].value[:, tinds]
        f.close()
        
        
        ##load nest spike times
        T = [self.TRANSIENT, tstop]       
        
        #scale ybar
        vlim = abs(lfp_noisy).max()
        scale = 2.**np.round(np.log2(vlim))
        
        #create figure object
        fig = plt.figure(figsize=(10, 7))
        
        
        #plot model noise
        ax0 = fig.add_axes([0.075, 0.55, 0.175, 0.4])
        yticks = []
        yticklabels = []
        i = 0
        for trace in noise:
            ax0.plot(tvec, trace - i*scale, color=self.electrodeColors[i],
                     rasterized=True, clip_on=False)
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))
            i += 1
        ax0.axis('tight')
        ax0.set_yticks(yticks)
        ax0.set_yticklabels(yticklabels)
        for loc, spine in ax0.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        ax0.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax0.set_title('model noise')
        ax0.text(-0.25, 1.0, 'a',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax0.transAxes)
        
        


        #plot rasters of the spike trains each postsynaptic cell receives
        #open db with excitatory spike trains:
        db = GDF(os.path.join(self.savefolder, 'SpTimesEx.db'),
                          new_db=False)

        ax44 = fig.add_axes([0.075, 0.3, 0.175, 0.175])
        
        yticks = []
        yticklabels = []
        #compute binned spike trains and corresponding correlation
        
        j = 0
        for i, cell in cells.iteritems():
            xe = np.array([])
            ye = np.array([])
            spiketimes = db.select_neurons_interval(self.testdInst.SpCellsEx[i][::50], T=T)
            for times in spiketimes:
                xe = np.r_[xe, times]
                ye = np.r_[ye, np.zeros(times.size) - j]
                j += 1
            yticks.append(-j + len(spiketimes) / 2)
            yticklabels.append('cell %i' % (i+1))

            ax44.plot(xe, ye, 'o',
                markersize=2,
                markerfacecolor=self.colors[i],
                markeredgecolor='none',
                alpha=1,
                clip_on=False,
                label='cell %i' % (i+1), rasterized=True)
        
        db.close()
        del xe, ye

        for loc, spine in ax44.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        ax44.xaxis.set_ticks_position('bottom')
        ax44.yaxis.set_ticks_position('left')

        ax44.axis(ax44.axis('tight'))
        ax44.set_xlim(T[0], T[1])
        
        ax44.set_yticks(yticks)
        ax44.set_yticklabels(yticklabels)
        ax44.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax44.set_xticklabels([])
        ax44.set_title('exc. spike trains')
        
        ax44.text(-0.25, 1.0, 'b',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax44.transAxes)




        #open db with inhibitory spike trains:
        db = GDF(os.path.join(self.savefolder, 'SpTimesIn.db'),
                          new_db=False)
        
        ax45 = fig.add_axes([0.075, 0.075, 0.175, 0.175])

        yticks = []
        yticklabels = []
        
        j = 0
        for i, cell in cells.iteritems():
            xi = np.array([])
            yi = np.array([])
            spiketimes = db.select_neurons_interval(self.testdInst.SpCellsIn[i][::50], T=T)
            for times in spiketimes:
                xi = np.r_[xi, times]
                yi = np.r_[yi, np.zeros(times.size)- j]
                j += 1
            yticks.append(-j + len(spiketimes) / 2)
            yticklabels.append('cell %i' % (i+1))
        
            ax45.plot(xi, yi, 'o',
                markersize=2,
                markerfacecolor=self.colors[i],
                markeredgecolor='none',
                alpha=1,
                clip_on=False,
                label='cell %i' % (i+1), rasterized=True)

        db.close()
        del xi, yi
        
        for loc, spine in ax45.spines.iteritems():
            if loc in ['right','top']:
                spine.set_color('none') # don't draw spine
        ax45.xaxis.set_ticks_position('bottom')
        ax45.yaxis.set_ticks_position('left')
        
        ax45.axis(ax45.axis('tight'))
        ax45.set_xlim(T[0], T[1])
        
        ax45.set_yticks(yticks)
        ax45.set_yticklabels(yticklabels)
        ax45.xaxis.set_major_locator(plt.MaxNLocator(2))

        ax45.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        ax45.set_title('inh. spike trains')
                



        
        
        #plot the electrode and simple cell models
        ax6 = fig.add_axes([0.275, 0.075, 0.175, 0.875],
            aspect='equal', adjustable='datalim', frame_on=False)
        
        #plot cell schematics
        for cellindex, cell in cells.iteritems():
            x, y, z = schemPyr(cell, scale=0.35)
            ax6.fill(x, z, color=self.colors[cellindex], lw=2,
                    rasterized=False, zorder=cell.somapos[1])        
        

        #draw the outer boundary of the population
        mpop = self.testdInst.populationParameters
        
        #outline of electrode        
        try:
            x_0 = mpop['r_z'][1, 1:-1]
            z_0 = mpop['r_z'][0, 1:-1]
            x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
            z = np.r_[2000, z_0[::-1], z_0[1:], 2000]
        except:
            x = np.r_[mpop['X'][0, 1:-1], mpop['X'][1, 1:-1][::-1]]
            z = np.r_[mpop['Z'][1:-1], mpop['Z'][1:-1][::-1]]            
            #filter z > 1000:
            z[z > 1000] = 1000

        y = np.zeros(x.size)
        ax6.fill(x, z, color=(0.5,0.5,0.5), lw=None, alpha=0.5, zorder=-1.)
        
        #contact points
        for i in range(len(self.electrodeColors)):
            radius = self.testdInst.electrodeParameters['r']
            center = [self.testdInst.electrodeParameters['x'][i],
                   self.testdInst.electrodeParameters['z'][i]]
            x, z = contPoint(radius, center)
            ax6.fill(x, z, color=self.electrodeColors[i], lw=0, zorder=0)
        
        #plot wire from each contact point
        X = self.testdInst.electrodeParameters['x']
        Z = self.testdInst.electrodeParameters['z']
        
        #loop over contacts, plot cables etc
        theta = np.arange(11) / 10. * np.pi / 2 + np.pi/2
        theta = theta[::-1]
        radius = 50
        ncont = len(self.electrodeColors)
        for i in range(ncont):
            x = np.r_[X[i],
                      i*10,
                      i*10,
                      i*10 + radius*np.cos(theta) + radius,
                      130]
            z = np.r_[Z[i],
                      Z[i]+i*10,
                      Z[i]+(i*0.5)*100 + 650-i*100,
                      Z[i]+(i*0.5)*100 + 650-i*100 + \
                                        radius*np.sin(theta) + radius,
                      Z[i]+(i*0.5+1)*100 + 650-i*100]
            ax6.plot(x, z, color=self.electrodeColors[i], lw=1, zorder=0)
            ax6.text(x[-1], z[-1]+10, 'ch. %i' % (i+1),
                     color=self.electrodeColors[i],
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     fontsize=smallfontsize)
        ax6.set_title('population')
        ax6.set_xticks([])
        ax6.set_xticklabels([])
        ax6.set_yticks([])
        ax6.set_yticklabels([])
        ax6.axis(ax6.axis('tight'))
        ax6.set_aspect('equal', adjustable='datalim')
        ax6.text(0.021, 1.0, 'c',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax6.transAxes)     


        
        
        ax4 = fig.add_axes([0.5, 0.55, 0.175, 0.4])

        i = 0
        yticks = []
        yticklabels = []
        
        for x in lfp_noisy:
            ax4.plot(tvec, x - i*scale, color=self.electrodeColors[i],
                     label=None, clip_on=False,
                     rasterized=True)
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))
            i += 1
        
        for loc, spine in ax4.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
        ax4.xaxis.set_ticks_position('bottom')
        ax4.yaxis.set_ticks_position('left')
        ax4.set_yticks(yticks)
        ax4.set_yticklabels(yticklabels)
        ax4.set_title('EPs + noise')
        axis = ax4.axis(ax4.axis('tight'))
        ax4.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax0.axis(ax4.axis()) #using same scale as in the first plot
        ax4.plot([tvec[-1], tvec[-1]],[axis[2], axis[2]+scale],
            lw=4, color='k', clip_on=False)
        ax4.text(axis[1]*1.01, axis[2], '%.2fmV' % scale,
                 fontsize=smallfontsize, rotation='vertical', verticalalignment='bottom')
        ax0.plot([tvec[-1], tvec[-1]],[axis[2], axis[2]+scale],
            lw=4, color='k', clip_on=False)
        ax0.text(axis[1]*1.01, axis[2], '%.2fmV' % scale,
                 fontsize=smallfontsize, rotation='vertical', verticalalignment='bottom')
        ax4.text(-0.25, 1.0, 'd',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax4.transAxes)        
        
        
        
        ax5 = fig.add_axes([0.775, 0.55, 0.175, 0.4], frame_on=False)
        
        patcheslist = []
        fancybox = patches.FancyBboxPatch(
                [-2.5, -1], 5, 2.5,
                boxstyle=patches.BoxStyle("Round", pad=0.5),
                ec=(0,0,0), fc=(1,1,1), lw=1, ls='solid', clip_on=False, )
        patcheslist.append(fancybox)
        collection = PatchCollection(patcheslist, match_original=True, clip_on=False)
        ax5.add_collection(collection)
        ax5.text(0, 0.35, "spike sorting", ha="center", fontsize=18,)
        ax5.text(0, -0.35, "evaluation", ha="center", fontsize=18,)
                 
                 
        ax5.set_xlim(-3, 3)
        ax5.set_ylim(-3, 3)
        ax5.set_xticks([])
        ax5.set_xticklabels([])
        ax5.set_yticks([])
        ax5.set_yticklabels([])
        ax5.text(-0.25, 1.0, 'e',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax5.transAxes)       
 
        
        
        
        
        ax7 = fig.add_axes([0.5, 0.075, 0.175, 0.4])
        yticklabels = []
        yticks = []
        i = 0
        scale = 100
        for cellkey, cell in cells.iteritems():
            ax7.plot(tvec, cell.somav[tinds] - i*scale,
                color = self.colors[cellkey],
                alpha = self.alphas[cellkey],
                lw=1,
                label = 'cell %i' % (cellkey+1), rasterized=True)
        
            yticklabels.append('cell %i' % (cellkey+1))
            yticks.append(-i*scale)
            i += 1
        
        for loc, spine in ax7.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax7.xaxis.set_ticks_position('bottom')
        ax7.yaxis.set_ticks_position('left')
        ax7.set_yticks(yticks)
        ax7.set_yticklabels(yticklabels)
        ax7.set_xlim(self.TRANSIENT, tstop)
        axis = ax7.axis(ax7.axis('tight'))
        ax7.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax7.plot([tvec[-1], tvec[-1]], [axis[2], axis[2]+scale],
            'k', lw=4, clip_on=False)
        ax7.text(axis[1]*1.01, axis[2], '100mV',
                 fontsize=smallfontsize, rotation='vertical', verticalalignment='bottom')
        ax7.set_title('soma voltages')
        ax7.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        ax7.axis(ax7.axis('tight'))
        ax7.text(-0.25, 1.0, 'f',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax7.transAxes)      
        
        
        
        
        ax8 = fig.add_axes([0.775, 0.075, 0.175, 0.4])

        ax8.set_title('spike raster')
        yticklabels = []
        yticks = []
        i = 0
        for cellkey, cell in cells.iteritems():
            nan_train = cell.AP_train[tinds].astype(float)
            sptimes = np.where(nan_train == 1)[0]
            nan_train *= np.nan
            nan_train[sptimes] = 1
            nan_train[sptimes-1] = 0
             
            ax8.plot(tvec, nan_train - i,
                    color = self.colors[cellkey],
                    alpha = self.alphas[cellkey], lw=2,
                    label = 'cell %i' % (cellkey+1))
            
            yticklabels.append('cell %i' % (cellkey+1))
            yticks.append(-i + 0.5)
            i += 1
        
        for loc, spine in ax8.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax8.xaxis.set_ticks_position('bottom')
        ax8.yaxis.set_ticks_position('left')
        
        ax8.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        ax8.set_yticks(yticks)
        ax8.set_yticklabels(yticklabels)
        ax8.set_xlim(self.TRANSIENT, tstop)
        ax8.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax8.set_ylim(yticks[-1]-0.5, 1)          
        
        ax8.text(-0.25, 1.0, 'g',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax8.transAxes)
        
        
        
        
        #plot arrows here
        ax = fig.add_axes([0, 0, 1, 1], axisbg='none', frame_on=False)
        
        fancy_arrow(ax, [0.25,  0.26, 0.26, 0.435, 0.435,  0.44],
                        [0.75, 0.75, 0.99,  0.99, 0.75, 0.75])
        fancy_arrow(ax, [0.26, 0.3], [0.2625, 0.2625])
        fancy_arrow(ax, [0.435, 0.44], [0.725, 0.725])
        fancy_arrow(ax, [0.40, 0.44], [0.2625, 0.2625])
        fancy_arrow(ax, [0.68, 0.715], [0.725, 0.725])
        fancy_arrow(ax, [0.68, 0.715], [0.2625, 0.2625])
        fancy_arrow(ax, [0.85, 0.85], [0.52, 0.6])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        return fig
    

    def plot_figure_07(self, T=[0., 10000.]):
        '''
        plot noise and noise properties
        '''   
        #file object generated by noise extraction and generation
        f = h5py.File(os.path.join(self.savefolder, 'ViSAPy_noise.h5'))
        
        
        #plot        
        fig = plt.figure(figsize=(10,10))
        
        #experimental traces
        ax = fig.add_axes([0.07, 0.725, 0.85, 0.25])
        ax.set_rasterization_zorder(1)
        yticks = []
        yticklabels = []
        data = f['input_data'].value.T
        tvec = np.arange(data.shape[1]) * self.testdInst.cellParameters['dt']
        #slice data and tvec and noise:
        if tvec[-1] > T[1]:
            data = data[:, (tvec <= T[1]) & (tvec >= T[0])]
            tvec = tvec[(tvec <= T[1]) & (tvec >= T[0])]
            
        tvec_noise = np.arange(f['data'].value.shape[1]) * \
                    self.testdInst.cellParameters['dt'] +\
                    self.testdInst.cellParameters['tstart']
        noise = f['data'].value[:, (tvec_noise <= T[1]) & (tvec_noise >= T[0])]
        
        
        scale = 0.25
        zips = []
        for i, trace in enumerate(data):
            zips.append(zip(tvec, trace - i*scale))
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))

        linecol = LineCollection(zips,
                                 color='k',
                                 clip_on=False,
                                 zorder=0,
                                 )
        ax.add_collection(linecol)
        
        axis = ax.axis(ax.axis('tight'))
        ax.plot([axis[1], axis[1]], [axis[2],axis[2]+scale],
            lw=4, color='k', clip_on=False)
        ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, axis[2], '%.2fmV' % scale,
                rotation='vertical', va='bottom', fontsize=smallfontsize)
        
        ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title('recorded noise')
        
        ax.text(-0.05, 1.0, 'a',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)
        
        
        #PSDs from each channel
        ax = fig.add_axes([0.07, 0.55, 0.175, 0.125])
        ax.set_rasterization_zorder(1)
        psd = f['psd'].value.mean(axis=1)
        ax.loglog(f['freqs'].value[1:2**15] * 1000 /
                  self.testdInst.cellParameters['dt'],
                  psd[1:2**15]/np.sqrt(f['NFFT'].value / self.testdInst.cellParameters['dt']),
                  color='k',
                  zorder=0,
                  rasterized=True,
                  )
        
        ax.axis(ax.axis('tight'))
        ax.grid('on')
        ax.set_ylabel(r'(mV$^2$/Hz)', ha='center', va='center')
        ax.set_title('PSD')
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.text(-0.25, 1.0, 'b',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)

        
        
        #plot the logbump filter response curves
        ax = fig.add_axes([0.07, 0.35, 0.175, 0.125])
        ax.set_rasterization_zorder(1)

        basist = 1000 / (2 * self.testdInst.cellParameters['dt'])
        #from logbumps import logbumps
        logbases = fcorr.logbumps(n=f['C'].shape[0], t=basist,
                        normalize=False, alpha=f['alpha'].value, debug=False,
                        figno='Logbumps')

        zips = []
        for i in range(f['C'].shape[0]):
            zips.append(zip(np.arange(logbases[i].size)[1:], logbases[i][1:]))
            
        line_segments = LineCollection(zips,
                                       linewidths = (1),
                                       linestyles = 'solid',
                                       #clip_on=False,
                                       cmap = plt.cm.get_cmap('viridis', f['C'].shape[0]),
                                       zorder=0,
                                       #rasterized=True,
                                       )
        line_segments.set_array(np.arange(f['C'].shape[0]))
        ax.add_collection(line_segments)
        ax.semilogx()
        #ax.set_ylim(-30., 0.)
        rect = np.array(ax.get_position().bounds)
        rect[0] += rect[2] + 0.01
        rect[2] = 0.015
        cax = fig.add_axes(rect)
        axcb = fig.colorbar(line_segments, cax=cax)
        cax.yaxis.set_ticks(np.linspace(1./f['C'].shape[0]/2, 1-1./f['C'].shape[0]/2, f['C'].shape[0])[::2])
        cax.yaxis.set_ticklabels(np.arange(f['C'].shape[0])[::2]+1)


        axcb.set_label('bump #')
        
        ax.grid('on')
        ax.axis(ax.axis('tight'))
        ax.set_xlim(20, basist)
        ax.set_ylabel('magnitude')
        ax.set_xlabel(r'$f$ (Hz)', labelpad=0.1)
        ax.set_title('filter basis')
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


        ax.text(-0.2, 1.0, 'c',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)

    
        #plot covariance matrices
        rows = 4
        cols = 0
        while rows * cols < f['C'].shape[0]: cols += 1
        height = 0.5 / rows / 1.75
        width = height
        
        #some grid for subplots
        x = []
        for i in range(rows):
            x = np.r_[x, np.linspace(0.35, 0.92-width, cols)]
        y = []
        for i in range(cols):
            y = np.r_[y, np.linspace(0.35, 0.675-height, rows)]
        y.sort()
        y = y[::-1]
        
        for i in range(f['C'].shape[0]):
            ax = fig.add_axes([x[i], y[i], width, height], frameon=False)
            im = ax.matshow(f['C'].value[i, ], rasterized=True,
                            cmap = plt.cm.get_cmap('viridis', f['C'].shape[0]))
            rect = np.array(ax.get_position().bounds)
            rect[0] += rect[2]  #- 0.02
            rect[2] = 0.015
            cax = fig.add_axes(rect)
            cax.set_rasterization_zorder(1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([f['C'].value[i, ].min(), f['C'].value[i, ].max()])
            cbar.set_ticklabels(['%.1e' % f['C'].value[i, ].min(), '%.1e' % f['C'].value[i, ].max()])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis(ax.axis('tight'))
            ax.text(0, 0.5, r'$C_{%i}$' % (i+1),
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax.transAxes)
            
            
            if i == 0:
                ax.text(-0.4, 1.0, 'd',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=18, fontweight='demibold',
                    transform=ax.transAxes)

            
            
        #model traces
        ax = fig.add_axes([0.07, 0.05, 0.85, 0.25])
        ax.set_rasterization_zorder(1)
        yticks = []
        yticklabels = []
        data = noise
        tvec = np.arange(data.shape[1]) * self.testdInst.cellParameters['dt']
        zips = []
        for i, trace in enumerate(data):
            zips.append(zip(tvec, trace - i*scale))
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))

        linecol = LineCollection(zips,
                                 color='k',
                                 clip_on=False,
                                 zorder=0,
                                 #rasterized=True,
                                 )
        ax.add_collection(linecol)
        
        axis = ax.axis(ax.axis('tight'))
        ax.plot([axis[1], axis[1]], [axis[2],axis[2]+scale],
            lw=4, color='k', clip_on=False)
        ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, axis[2], '%.2fmV' % scale,
                rotation='vertical', va='bottom', fontsize=smallfontsize)
        ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax.set_title('model noise')
        
        ax.text(-0.05, 1.0, 'e',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)
        
        
        return fig
        

    def plot_figure_08(self, cells, tstop=1500.):
    
        fig = plt.figure(figsize=(10, 12))
        fig.subplots_adjust(left=0.06, right=0.9, bottom=0.04, top=0.975, wspace=0.8, hspace=0.40)
        #fig.subplots_adjust(left=0.06, right=0.96, bottom=0.04, top=0.975, hspace=0.25, wspace=0.15)

        GS52 = GridSpec(5,6)
        GS53 = GridSpec(5,3)
        
        def plotSpCells(self, cellindices=None, T=[500., 1000.], binsize=10., cellinterval=1):
            '''plot histograms over which cells are used for input to
            the postsynaptic cells'''
            
            if cellindices is None:
                cellindices = np.arange(self.testdInst.POPULATION_SIZE)
            ygrid = np.r_[cellindices, cellindices[-1]]
            yticks = cellindices + 0.5
            yticklabels = []
            for i in cellindices:
                yticklabels.append('cell %i' % (i+1))
            
            #Excitatory
            binsize_ex = np.round(self.testdInst.nodes_ex.size / 100, 0)
            bins_ex = np.arange(np.array(self.testdInst.networkInstance.nodes_ex).min(),
                        np.array(self.testdInst.networkInstance.nodes_ex).max()+binsize_ex,
                        binsize_ex)
            histEx = np.array([])
            for x in self.testdInst.SpCellsEx:
                histEx = np.r_[histEx, np.histogram(x, bins=bins_ex)[0]]
            histEx = histEx.reshape(cellindices.size, -1)
            
            #inhibitory
            binsize_in = np.round(self.testdInst.nodes_in.size / 100, 0)
            bins_in = np.arange(np.array(self.testdInst.networkInstance.nodes_in).min(),
                        np.array(self.testdInst.networkInstance.nodes_in).max()+binsize_in,
                        binsize_in)
            histIn = np.array([])
            for x in self.testdInst.SpCellsIn:
                histIn = np.r_[histIn, np.histogram(x, bins=bins_in)[0]]    
            histIn = histIn.reshape(cellindices.size, -1)
                
        
            
            #plot rasters of the spike trains each postsynaptic cell receives
            #open db with excitatory spike trains:
            dbE = GDF(os.path.join(self.savefolder, 'SpTimesEx.db'),
                              new_db=False)
        
            ax4 = fig.add_subplot(GS52[0, :3])
            ax5 = fig.add_subplot(GS52[0, 3:])
            
            yticks = []
            yticklabels = []
            #compute binned spike trains and corresponding correlation
            bins = np.arange(0, self.testdInst.networkInstance.simtime + binsize, binsize)
            
            j = 0
            for i in cellindices:
                xe = np.array([])
                ye = np.array([])
                spiketimes = dbE.select_neurons_interval(self.testdInst.SpCellsEx[i][::cellinterval], T=T)
                for times in spiketimes:
                    xe = np.r_[xe, times]
                    ye = np.r_[ye, np.zeros(times.size) - j]
                    j += 1
                yticks.append(-j + len(spiketimes) / 2)
                yticklabels.append('cell %i' % (i+1))
        
                ax4.plot(xe, ye, 'o',
                    markersize=1,
                    markerfacecolor=self.colors[i],
                    markeredgecolor='none',
                    alpha=1,
                    label='cell %i' % (i+1), rasterized=True)
        
            
            del xe, ye
            
            
            for loc, spine in ax4.spines.iteritems():
                if loc in ['right','top']:
                    spine.set_color('none') # don't draw spine
            ax4.xaxis.set_ticks_position('bottom')
            ax4.yaxis.set_ticks_position('left')
            
            ax4.set_yticks(yticks)
            ax4.set_yticklabels(yticklabels)
            ax4.axis(ax4.axis('tight'))
            ax4.set_title('excitatory synapse spike trains')
            ax4.set_xlabel(r'$t$ (ms)', labelpad=0.1)
            
            ax4.text(-0.1, 1.0, 'a',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax4.transAxes)
            
            
            #open db with inhibitory spike trains:
            dbI = GDF(os.path.join(self.savefolder, 'SpTimesIn.db'),
                              new_db=False)
            
            
            yticks = []
            yticklabels = []
            
            j = 0
            for i in cellindices:
                xi = np.array([])
                yi = np.array([])
                spiketimes = dbI.select_neurons_interval(self.testdInst.SpCellsIn[i][::cellinterval], T=T)
                for times in spiketimes:
                    xi = np.r_[xi, times]
                    yi = np.r_[yi, np.zeros(times.size) - j]
                    j += 1
                yticks.append(-j + len(spiketimes) / 2)
                yticklabels.append('cell %i' % (i+1))
            
                ax5.plot(xi, yi, 'o',
                    markersize=1,
                    markerfacecolor=self.colors[i],
                    markeredgecolor='none',
                    alpha=1,
                    label='cell %i' % (i+1), rasterized=True)
        
            
            del xi, yi
            
            
            for loc, spine in ax5.spines.iteritems():
                if loc in ['right','top']:
                    spine.set_color('none') # don't draw spine
            ax5.xaxis.set_ticks_position('bottom')
            ax5.yaxis.set_ticks_position('left')
            
            ax5.set_yticks(yticks)
            ax5.set_yticklabels([])
            ax5.set_xlabel(r'$t$ (ms)', labelpad=0.1)
            ax5.set_title('inhibitory synapse spike trains')
            ax5.axis(ax5.axis('tight'))
            
            
            ax5.text(-0.1, 1.0, 'b',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax5.transAxes)
        
        
            #calculate spike histograms for each postsyn cells
            histEx = np.array([], dtype=int)
            for i in cellindices:
                xe = np.array([])
                spiketimes = dbE.select_neurons_interval(self.testdInst.SpCellsEx[i][::cellinterval],
                                                         T=[self.TRANSIENT,
                                                            self.testdInst.networkInstance.simtime])
                for times in spiketimes:
                    xe = np.r_[xe, times]
                
                hist = np.histogram(xe, bins)
                if histEx.size == 0:
                    histEx = np.r_[histEx, hist[0].astype(int)]
                else:
                    histEx = np.c_[histEx, hist[0].astype(int)]
        
            dbE.close()
            histEx = histEx.T
            del xe, hist
            
            
            if cellindices.size <= 1:
                print('can not plot correlations for population of size <= 1')
            else:
                #compute correlation coefficients:
                correlationsEx = np.corrcoef(histEx)
                
                
                ax1 = fig.add_subplot(GS53[1, 0])
                im1 = ax1.matshow(correlationsEx, cmap = plt.cm.get_cmap('viridis', 41))
                ax1.set_title('corr.coeff, exc. trains')
                ax1.xaxis.set_ticks_position('bottom')
                yticklabels = []
                for i in np.r_[-1, cellindices]:
                    yticklabels.append('train %i' % (i+1))
                ax1.set_xticklabels(yticklabels, rotation=45)
                ax1.set_yticklabels(yticklabels, rotation=45)
            
                rect = np.array(ax1.get_position().bounds)
                rect[0] += rect[2] + 0.01
                rect[2] = 0.015
                cax = fig.add_axes(rect)
                cax.set_rasterization_zorder(1)
            
                cbar = fig.colorbar(im1, cax=cax)
                
                cbar.set_label(r'$c_\mathrm{spike-time}$', labelpad=0.1)
                ax1.axis(ax1.axis('tight'))
            
                ax1.text(-0.2, 1.0, 'c',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=18, fontweight='demibold',
                    transform=ax1.transAxes)
            
            
            
                #calculate spike histograms for each postsyn cells
                histIn = np.array([], dtype=int)
                for i in cellindices:
                    xi = np.array([])
                    spiketimes = dbI.select_neurons_interval(self.testdInst.SpCellsIn[i][::cellinterval],
                                                             T=[self.TRANSIENT,
                                                                self.testdInst.networkInstance.simtime])
                    for times in spiketimes:
                        xi = np.r_[xi, times]
                    
                    hist = np.histogram(xi, bins)
                    if histIn.size == 0:
                        histIn = np.r_[histIn, hist[0].astype(int)]
                    else:
                        histIn = np.c_[histIn, hist[0].astype(int)]
            
                dbI.close()
                histIn = histIn.T
                del xi, hist
            
                #compute correlation coefficients:
                correlationsIn = np.corrcoef(histIn)
                
                ax3 = fig.add_subplot(GS53[1, 1])
            
                im3 = ax3.matshow(correlationsIn, cmap = plt.cm.get_cmap('viridis', 41))
                ax3.set_title('corr.coeff, inh. trains')
                ax3.xaxis.set_ticks_position('bottom')
                yticklabels = []
                for i in np.r_[-1, cellindices]:
                    yticklabels.append('train %i' % (i+1))
                ax3.set_xticklabels(yticklabels, rotation=45)
                ax3.set_yticklabels(yticklabels, rotation=45)
                
                
                rect = np.array(ax3.get_position().bounds)
                rect[0] += rect[2] + 0.01
                rect[2] = 0.015
                cax = fig.add_axes(rect)
                cax.set_rasterization_zorder(1)
                
                cbar = fig.colorbar(im3, cax=cax)
                
                cbar.set_label(r'$c_\mathrm{spike-time}$')
                ax3.axis(ax3.axis('tight'))
            
                ax3.text(-0.2, 1.0, 'd',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=18, fontweight='demibold',
                    transform=ax3.transAxes)
            
            
                #compute correlation coefficients:
                correlationsExIn = np.corrcoef(histEx, histIn)[histEx.shape[0]:, :histEx.shape[0]]
                
                
                ax4 = fig.add_subplot(GS53[1, 2])
            
                im4 = ax4.matshow(correlationsExIn, cmap = plt.cm.get_cmap('viridis', 41))
                ax4.set_title('corr.coeff, exc. and inh.')
                ax4.xaxis.set_ticks_position('bottom')
                yticklabels = []
                for i in np.r_[-1, cellindices]:
                    yticklabels.append('train %i' % (i+1))
                ax4.set_xticklabels(yticklabels, rotation=45)
                ax4.set_yticklabels(yticklabels, rotation=45)
                
                
                rect = np.array(ax4.get_position().bounds)
                rect[0] += rect[2] + 0.01
                rect[2] = 0.015
                cax = fig.add_axes(rect)
                cax.set_rasterization_zorder(1)
                
                cbar = fig.colorbar(im4, cax=cax)
                
                cbar.set_label(r'$c_\mathrm{spike-time}$')
                ax4.axis(ax4.axis('tight'))
            
                ax4.text(-0.2, 1.0, 'e',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=18, fontweight='demibold',
                    transform=ax4.transAxes)
        
        
        
        def plotSpikeTrains(self, cells,
                            bins=10**np.linspace(np.log10(1), np.log10(1E3), 100),
                            binwidth=10, maxlagms = 100, show_rates=True,
                            tstop = 1500.):
            '''
            gather action potentials from all cells, and plot spike trains
            '''        
            ax = fig.add_subplot(GS53[2, :])
        
            yticklabels = []
            yticks = []
            i = 0
            for cellkey, cell in cells.iteritems():
                tinds = np.arange(self.TRANSIENT / self.testdInst.cellParameters['dt'],
                                  tstop / self.testdInst.cellParameters['dt'] + 1,
                                  dtype=int)
                tvec = tinds * self.testdInst.cellParameters['dt']
                ax.plot(tvec, cell.somav[tinds] - i*100,
                    color = self.colors[cellkey],
                    alpha = self.alphas[cellkey],
                    lw=2,
                    label = 'cell %i' % (cellkey+1), rasterized=True,
                    clip_on=False)
        
                yticklabels.append('cell %i' % (cellkey+1))
                yticks.append(cell.somav[tinds].mean() - i*100)
                i += 1
        
            for loc, spine in ax.spines.iteritems():
                if loc in ['right', 'top']:
                    spine.set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        
        
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            
            ax.set_xlim(self.TRANSIENT, tstop)
            
            ax.axis(ax.axis('tight'))
            axis = ax.axis()
            
            ax.plot([axis[1], axis[1]], [-i*100+30, -(i-1)*100+30],
                'k', lw=4, clip_on=False)
            ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, -i*100+30, '100 mV',
                    rotation='vertical', fontsize=smallfontsize, va='bottom')
            
            ax.set_title('somatic traces')
            
            plt.axis('tight')
        
            ax.text(-0.05, 1.0, 'f',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax.transAxes)        
            
            
            
            ax = fig.add_subplot(GS52[3, :])
        
            ax.set_title('spike raster')
            yticklabels = []
            yticks = []
            i = 0
            for cellkey, cell in cells.iteritems():
                nan_train = cell.AP_train[tinds].astype(float)
                sptimes = np.where(nan_train == 1)[0]
                nan_train *= np.nan
                nan_train[sptimes] = 1
                nan_train[sptimes-1] = 0
                 
                ax.plot(tvec, nan_train - i,
                        color = self.colors[cellkey],
                        alpha = self.alphas[cellkey], lw=2,
                        label = 'cell %i' % (cellkey+1))
                
                yticklabels.append('cell %i' % (cellkey+1))
                yticks.append(-i + 0.5)
                i += 1
        
                if show_rates:
                    ax.text(tvec[-1]+10, 1.25-i,
                            '%.1f' % (cell.AP_train[int(self.TRANSIENT / cell.dt):].sum() * 1000. / cell.tstop),
                            va='bottom', ha='left', fontsize=smallfontsize)
            
            if show_rates:
                ax.text(1, 1, r'rate (s$^{-1}$)', fontsize=smallfontsize,
                        transform=ax.transAxes)
        
        
        
            for loc, spine in ax.spines.iteritems():
                if loc in ['right', 'top']:
                    spine.set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        
            ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_xlim(self.TRANSIENT, tstop)
            ax.set_ylim(yticks[-1]-0.5, 1)          
        
            ax.text(-0.05, 1.0, 'g',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax.transAxes)
        
        
        
        
            ncells = len(cells.keys())
            rows = 2
            cols = 0
            while rows * cols < ncells: cols += 1
            
            
            width = 0.7 / cols / 1.25
            height = 0.18 / rows / 1.5
            
            #some grid for subplots
            x = []
            for i in range(rows):
                x = np.r_[x, np.linspace(0.06, 0.675-width, cols)]
            y = []
            for i in range(cols):
                y = np.r_[y, np.linspace(0.04, 0.22, rows+1)[:-1]]
            y.sort()
            y = y[::-1]
            
        
            for i, (cellkey, cell) in enumerate(cells.iteritems()):
                TRANSIENT = int(self.TRANSIENT / cell.dt)
                ISI_tvec = np.arange(cell.somav[TRANSIENT:].size) * cell.dt
                ISI = np.diff(ISI_tvec[cell.AP_train[TRANSIENT:]==1])
                ax = fig.add_axes([x[i], y[i], width, height])
                try:
                    ax.hist(ISI, bins=bins,
                            color=self.colors[cellkey],
                            histtype='stepfilled',
                            alpha=self.alphas[cellkey],
                            edgecolor=self.colors[cellkey],
                            )
                except:
                    print('not enough spikes for ISI')
                ax.semilogx()
                ax.set_title(r'cell %i, %i APs' % (cellkey+1, cell.AP_train[TRANSIENT:].sum()))
                ax.axis(ax.axis('tight'))
                ax.set_xlim([bins.min(), bins.max()])
                ax.set_ylim(bottom=0)
        
                hist = np.histogram(ISI, bins=bins)
                ax.set_yticks([0, hist[0].max()])
                ax.set_yticklabels([0, int(hist[0].max())])
                
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
        
                for loc, spine in ax.spines.iteritems():
                    if loc in ['right', 'top']:
                        spine.set_color('none')
                
                if divmod(i, cols)[1] == 0:
                    ax.set_ylabel('count (-)', labelpad=0.1)
                
                if i >= cols:
                    ax.set_xlabel('ISI (ms)', labelpad=0.1)
                else:
                    ax.set_xticklabels([])
        
                if i == 0:
                    ax.text(-0.075*cols, 1.0, 'h',
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=18, fontweight='demibold',
                        transform=ax.transAxes)
        
        
            if len(cells.keys()) <= 1:
                print('can not plot correlations for population size <= 1')
            else:
                # plot Ecker et al correlations
                # load spikes
                db = GDF(':memory:')
                db.create(re=os.path.join(self.savefolder,
                                          'ViSAPy_ground_truth.gdf'))
                
                cellindices = cells.keys()
                neurons = np.array(cellindices) + 1
                N = len(neurons)
                
                # bin spike rasters
                bwidth = binwidth / self.testdInst.cellParameters['dt']
                mint = self.TRANSIENT / self.testdInst.cellParameters['dt']
                maxt = self.testdInst.cellParameters['tstop'] / self.testdInst.cellParameters['dt']
                rasters = db.select_neurons_interval(neurons,  T=[mint, maxt])
                
                bins = np.arange(mint, maxt+1, bwidth)
                spikes = np.empty((N, len(bins)-1))
                for i in range(N):
                    spikes[i], _ = np.histogram(rasters[i], bins=bins)
                
                #correlation coefficients will be put here
                correlations = np.corrcoef(spikes)
                
                #cap colormap
                vmax = 0.1
                if correlations.min() >= 0.1:
                    vmax = 1.
                
                ax = fig.add_subplot(GS53[4, 2])
                im = ax.matshow(correlations,
                                cmap=plt.cm.get_cmap('viridis', 41), vmax=vmax)
                ax.set_title('corr. coeff.')
                ax.xaxis.set_ticks_position('bottom')
                ticklabels = []
                for i in np.r_[-1, cellindices]:
                    ticklabels.append('cell %i' % (i+1))
                ax.set_xticklabels(ticklabels, rotation=45)
                ax.set_yticklabels(ticklabels, rotation=45)
                ax.axis(ax.axis('tight'))
            
                
                rect = np.array(ax.get_position().bounds)
                rect[0] += rect[2] + 0.01
                rect[2] = 0.015
                cax = fig.add_axes(rect)
                cax.set_rasterization_zorder(1)
                
            
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(r'$c_\mathrm{spike-time}$')
                
            
                ax.text(-0.075*cols, 1.0, 'i',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=18, fontweight='demibold',
                    transform=ax.transAxes)
        
        
        
        plotSpCells(self, T=[self.TRANSIENT, np.array([self.TRANSIENT, tstop]).mean()], cellinterval=20)
        plotSpikeTrains(self, cells, tstop=tstop)
        
        return fig


    def plot_figure_09(self, cells, tstop=1500., ):
        '''
        figure 09
        '''
        #load recording
        if self.testdInst.cellParameters['tstop'] < tstop:
            tstop = self.testdInst.cellParameters['tstop']
        
        tinds = np.arange(self.TRANSIENT / self.testdInst.cellParameters['dt'],
                            tstop / self.testdInst.cellParameters['dt']+1, dtype=int)
        tvec = tinds * self.testdInst.cellParameters['dt']
        
        #load some traces
        f = h5py.File(self.savefolder + '/ViSAPy_noiseless.h5', 'r')
        lfp_noiseless = f['data'].value.T[:, tinds]
        f.close()
        
        f = h5py.File(self.savefolder + '/ViSAPy_filterstep_0.h5',
                      'r')
        lfp_noisy = f['data'].value.T[:, tinds]
        f.close()
        
        f = h5py.File(self.savefolder + '/ViSAPy_filterstep_1.h5',
                      'r')
        lfp_filtered = f['data'].value.T[:, tinds]
        f.close()
        
        #plotting figure
        fig = plt.figure(figsize=(10, 12))
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.04, top=0.975, hspace=0.4, wspace=0.15)
                
        #handle for subplot locations
        GS = GridSpec(5, len(cells.keys()))
        
        
        yticks = []
        yticklabels = []
        
        vlim = abs(lfp_noisy).max()
        scale = 2.**np.round(np.log2(vlim))
        
        #subplot
        ax = fig.add_subplot(GS[0, :])
        
        for i, x in enumerate(lfp_noiseless):
            ax.plot(tvec, x - i*scale, color=self.electrodeColors[i],
                     rasterized=True)
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))
        
        
        axis = ax.axis(ax.axis('tight'))
        
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title('superimposed extracellular potentials')
        
        ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], lw=4,
            color='k', clip_on=False)
        ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, axis[2], '%.2f mV' % scale,
                 rotation='vertical', va='bottom', fontsize=smallfontsize)
        ax.set_xlim([self.TRANSIENT, tstop])
        
        ax.text(-0.05, 1.0, 'a',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)
        
        
        
        #subplot
        ax = fig.add_subplot(GS[1, :])
        
        for i, x in enumerate(lfp_noisy):
            ax.plot(tvec, x - i*scale, color=self.electrodeColors[i],
                     label=None, rasterized=True)
        
        axis=ax.axis(ax.axis('tight'))
        
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        ax.set_title('extracellular potentials + noise')
        
        ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], lw=4,
            color='k', clip_on=False)
        ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, axis[2], '%.2f mV' % scale,
                 rotation='vertical', va='bottom', fontsize=smallfontsize)
        
        ax.set_xlim([self.TRANSIENT, tstop])
        
        ax.text(-0.05, 1.0, 'b',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)        
        
        
        
        vlim = abs(lfp_filtered).max()
        scale = 2.**np.round(np.log2(vlim))
        
        #subplot
        ax = fig.add_subplot(GS[2, :])
        
        yticks = []
        for i, x in enumerate(lfp_filtered):
            ax.plot(tvec, x - i*scale, color=self.electrodeColors[i],
                     label=None, rasterized=True)
            yticks.append(-i*scale)
            i += 1
        
        axis=ax.axis(ax.axis('tight'))
        
        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        ax.set_title('filtered extracellular potentials')
        ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        
        ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], lw=4,
            color='k', clip_on=False)
        ax.text((axis[0]+np.diff(axis[:2])[0])*1.01, axis[2], '%.2f mV' % scale,
                 rotation='vertical', va='bottom', fontsize=smallfontsize)
        ax.set_xlim([self.TRANSIENT, tstop])
        
        ax.text(-0.05, 1.0, 'c',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)        
        
        
        
        def plotSpikeWaveforms(cellindices=None, num_units=None):
            '''plot spike waveforms'''
            if cellindices is None:
                cellindices = np.arange(self.testdInst.POPULATION_SIZE)
        
            #with many cells this is a bit cramped, plot only
            #num_units randomly chosen units, sorted  
            if num_units is not None:
                cellindices = np.random.permutation(cellindices)[:num_units]
                cellindices.sort()
        
        
            #get data to plot
            sp_waves, clust_idx, sp_win = self.get_sp_waves(filename=os.path.join(self.savefolder,
                                                                                  'ViSAPy_filterstep_1.h5'))
        
            #loop over cells and find template with largest amplitude
            templates = []
            for cellkey in cellindices:
                if sp_waves['data'][:, clust_idx==cellkey, :].shape[1] > 0:
                    templates.append(sp_waves['data'][:, clust_idx==cellkey, :].mean(axis=1))
            templates = np.array(templates)
            
            vlim = abs(templates).max()
        
            #vlim = abs(sp_waves['data']).max()
            scale = 2.**np.round(np.log2(vlim))
            tvec = sp_waves['time']
            xvec = np.arange(self.testdInst.TEMPLATELEN)
            
            #keep aspect ratio equal between panels
            axis = (0,
                    self.testdInst.TEMPLATELEN,
                    sp_waves['data'][:,:,-1].min() - (self.testdInst.electrodeParameters['x'].size-1)*scale,
                    sp_waves['data'][:,:,0].max())
            
        
            for count, cellkey in enumerate(cellindices):
                ax = fig.add_subplot(GS[3, count])
                zips = []
                yticks = []
                yticklabels = []
                for i in range(self.testdInst.electrodeParameters['x'].size):
                    if sp_waves['data'][:, clust_idx==cellkey, i].shape[1] > 0:
                        for j, x in enumerate(sp_waves['data'][:, clust_idx==cellkey, i].T):
                            zips.append(zip(xvec, x - i*scale))
                        
                        ax.plot(xvec, sp_waves['data'][:, clust_idx==cellkey, i].mean(axis=1) - i*scale,
                                 color='k', lw=2, clip_on=False, zorder=2)
                    
                    yticks.append(-i*scale)
                    yticklabels.append('ch. %i' % (i+1))
                    
                linecollection = LineCollection(zips,
                                                linewidths=(0.5),
                                                colors=self.colors[cellkey],
                                                rasterized=True,
                                                alpha=1,
                                                clip_on=False,
                                                zorder=0)    
                ax.add_collection(linecollection)
        
                for loc, spine in ax.spines.iteritems():
                    if loc in ['right', 'top',]:
                        spine.set_color('none')
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.set_yticks(yticks)
                ax.set_xticks([0, int(self.testdInst.TEMPLATELEN/2), self.testdInst.TEMPLATELEN])
                ax.set_xticklabels([0, int(self.testdInst.TEMPLATELEN/2)])
                if count == 0:
                    ax.set_yticklabels(yticklabels)
        
                    ax.text(-cellindices.size*0.05, 1.0, 'd',
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=18, fontweight='demibold',
                        transform=ax.transAxes)
                else:
                    ax.set_yticklabels([])
        
                if count == 0:
                    ax.set_xlabel('samples (-)', labelpad=0.1)
                
                ax.set_title('cell %i' % (cellkey+1) + '\n' + '%i APs' % sp_waves['data'][:, clust_idx==cellkey, 0].shape[1])        
                ax.axis(axis)
        
            ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], 'k', lw=4, clip_on=False)
            ax.text(axis[0]+np.diff(axis[:2])[0]*1.05, axis[2], '%.2f mV' % scale,
                     rotation='vertical', fontsize=smallfontsize, va='bottom')
        
        
        
        def plotSuperImposedTemplates(cells):
            '''plot lfp-traces with highlighted spike waveforms'''        
            #get data to plot
            sp_waves, clust_idx, sp_win = self.get_sp_waves(filename=os.path.join(self.savefolder,
                                                                                  'ViSAPy_filterstep_1.h5'))
        
            #concatenate some tvecs
            tvecs = tvec
            for i in range(lfp_filtered.shape[0]):
                if i != 0:
                    tvecs = np.r_[tvecs, tvec]
        
            #set lims in some 100 ms bin where number of spikes is maxed
            AP_trains = []
            for cell in cells.itervalues():
                AP_trains.append(cell.AP_train)
            AP_trains = np.array(AP_trains)
            AP_trains = AP_trains.sum(axis=0)
            #null out spikes at times < TRANSIENT
            AP_trains[:int(self.TRANSIENT / self.testdInst.cellParameters['dt'])] = 0
            AP_times = np.where(AP_trains >= 1)[0].astype(float)
            AP_times *= self.testdInst.cellParameters['dt']
            
            AP_bins = np.arange(0, self.testdInst.cellParameters['tstop'], 100)
            hist = np.histogram(AP_times, AP_bins)[0][:10]
            
            del AP_trains, AP_times
            ind = np.where(hist == hist.max())[0][0]
            
            tlim0 = AP_bins[ind]
            tlim1 = AP_bins[ind+1]+100
            
        
            vlim = abs(lfp_filtered[:, (tvec >= tlim0) & (tvec <= tlim1)]).max()
            scale = 2.**np.round(np.log2(vlim))
        
            
            ax = fig.add_subplot(GS[4, :])
            
            for i, x in enumerate(lfp_filtered):
                ax.plot(tvec, x - i*scale, label=None, color='k',
                         rasterized=True)
            
            for cellindex, cell in cells.iteritems():
                nanmat = np.zeros(lfp_filtered.shape)*np.nan
                spi = np.where(cell.AP_train[int(self.TRANSIENT /
                                                 self.testdInst.cellParameters['dt']):lfp_filtered.shape[1]] == 1)[0]
                for i in spi:
                    start = i - int(1.*self.testdInst.TEMPLATEOFFS *
                                    self.testdInst.TEMPLATELEN) + 1
                    end = i + int(1.*(1-self.testdInst.TEMPLATEOFFS) *
                                    self.testdInst.TEMPLATELEN)
                    inds = np.linspace(start, end,
                                       self.testdInst.TEMPLATELEN)
                    inds = np.array(inds).astype(int)
                    ind = inds >= lfp_filtered.shape[1]
                    inds[ind] = lfp_filtered.shape[1]-1
                    
                    ####load data
                    data = sp_waves['data'][:, clust_idx==cellindex, :].mean(axis=1)
                    #fill in NAN-array so it can be plotted easily
                    nanmat[:, inds[inds < lfp_filtered.shape[1]]] = data.T
        
        
                #shift the values accrd to channel
                for i in range(lfp_filtered.shape[0]):
                    nanmat[i, ] -= i*scale
                    
                nanmat = nanmat.flatten()
                
                ax.plot(tvecs, nanmat, label='cell %i' % (cellindex+1),
                        lw=2,
                        color = self.colors[cellindex],
                        alpha=self.alphas[cellindex], rasterized=True)
            
            
            for loc, spine in ax.spines.iteritems():
                if loc in ['right', 'top']:
                    spine.set_color('none')
            
            ax.plot([])
                    
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        
            ax.set_yticks(np.arange(lfp_filtered.shape[0])*-scale)
            labels = []
            for i in range(lfp_filtered.shape[0]): labels.append('ch. %i' % (i+1))
            ax.set_yticklabels(labels)
            ax.set_xlabel(r'$t$ (ms)', labelpad=0.1)
            ax.set_title('200 ms excerpt with superimposed templates')
        
            #showing just a 100 ms
            axis = ax.axis('tight')
            ax.axis([tlim0, tlim1, axis[2], axis[3]])
            
            axis = ax.axis()
            ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], 'k', lw=4, clip_on=False)
            ax.text(axis[0]+np.diff(axis[:2])[0]*1.01, axis[2], '%.2f mV' % scale,
                     rotation='vertical', fontsize=smallfontsize, va='bottom')
        
        
            ax.text(-0.05, 1.0, 'e',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=18, fontweight='demibold',
                transform=ax.transAxes)
            
        
        plotSpikeWaveforms()
        plotSuperImposedTemplates(cells)
    
        return fig


    def plot_figure_10(self, cellindices=None):
        '''
        plot PCA components etc obtained using ground truth
        spike times
        '''
        if cellindices is None:
            cellindices = np.arange(self.testdInst.POPULATION_SIZE)


        #params
        #sp_win = ((np.array([0, self.testdInst.TEMPLATELEN]) -
        #            self.testdInst.TEMPLATELEN*self.testdInst.TEMPLATEOFFS)
        #                * self.testdInst.cellParameters['dt']).tolist()
        ncomps = self.testdInst.nPCA

        ##load recording
        #f = h5py.File(os.path.join(self.savefolder, 'ViSAPy_filterstep_1.h5'))
        #sp = {
        #    'data' : f['data'].value.T.astype('float32'),
        #    'FS' : f['srate'].value,
        #    'n_contacts' : f['data'].value.shape[1],
        #}
        #f.close()
        
        ##load grount truth
        #GT = np.loadtxt(os.path.join(self.savefolder, 'ViSAPy_ground_truth.gdf')).T
        #clust_idx = GT[0, ].astype(int) - 1
        #spt = {
        #    'data' : GT[1, ] / sp['FS']*1000
        #}

        sp_waves, clust_idx, sp_win = self.get_sp_waves(filename=os.path.join(self.savefolder,
                                                                            'ViSAPy_filterstep_1.h5'))
        if sp_waves['data'].shape[1] <= 1:
            raise Exception('cant compute PCA for a single extracted spike')
            
        
        #extract waveforms and features
        #sp_waves = spike_sort.extract.extract_spikes(sp, spt, sp_win)
        features = spike_sort.features.combine(
                (
                    fetPCA(sp_waves, ncomps=ncomps),
                ),
                norm=False
        )
        
        #create figure
        fig = plt.figure(figsize=(10,10))
        
        #feature plots
        data = features['data']
        rows = cols = features['data'].shape[1]
        
        #create a 4X4 grid for subplots on [0.1, 0.5], [0.1, 0.5]
        width = 0.925 / rows * 0.99
        height= 0.925 / rows * 0.99
        x = np.linspace(0.05, 0.975-width, rows)
        y = np.linspace(0.05, 0.975-height, rows)[::-1]
        
        
        for i, j in np.ndindex((rows, rows)):
            if i < j:
                continue
            if i == j:
                bins = np.linspace(data[:, i].min(), data[:, i].max(), 50)
                ax = fig.add_axes([x[i], y[j], width, height])
                for cellkey in cellindices:

                    ax.hist(data[clust_idx==cellkey, i], bins=bins,
                            histtype='stepfilled', alpha=1,
                            edgecolor='none', facecolor=self.colors[cellkey],
                            zorder=-data[clust_idx==cellkey, i].size,
                            rasterized=True)
        
        
                    [count, bins] = np.histogram(data[clust_idx==cellkey, i],
                                                 bins=bins)
                    
                    normal = normalfun([data[clust_idx==cellkey, i].mean(),
                                        data[clust_idx==cellkey, i].std()], bins)
                    #scale to histogram:
                    normal /= normal.sum()
                    normal *= count.sum()
                    
                    #superimpose normal function
                    ax.plot(bins, normal, 'k', lw=1)
        
                ax.set_ylabel(features['names'][j])
            else: #then j > i
                ax = fig.add_axes([x[i], y[j], width, height])
                for cellkey in cellindices:
                    ax.scatter(data[clust_idx==cellkey, i],
                               data[clust_idx==cellkey, j],
                            marker='o',
                            facecolors=self.colors[cellkey], edgecolors='none',
                            s=5, alpha=1, rasterized=True,
                            zorder=-data[clust_idx==cellkey, i].size)
            
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
        
        return fig
 

    def plot_figure_11(self, cells, show_rates=True, tstop=1500.):
        '''
        gather action potentials from all cells, and plot spike trains
        '''        
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.06, right=0.91, bottom=0.05, top=0.975, hspace=0.10)
        ax = fig.add_subplot(211)
        yticklabels = []
        yticks = []
        for i, (cellkey, cell) in enumerate(cells.iteritems()):
            tinds = np.arange(self.TRANSIENT / self.testdInst.cellParameters['dt'],
                              tstop / self.testdInst.cellParameters['dt'] + 1,
                              dtype=int)
            tvec = tinds * self.testdInst.cellParameters['dt']
            ax.plot(tvec, cell.somav[tinds].astype(float) - i*100,
                color = self.colors[cellkey],
                alpha = self.alphas[cellkey],
                lw=2,
                label = 'cell %i' % (cellkey+1), rasterized=True)

            yticklabels.append('cell %i' % (cellkey+1))
            yticks.append(cell.somav[tinds].astype(float).mean() -i*100)
            
            if show_rates:
                ax.text(tvec[-1]+30, cell.somav[tinds].astype(float).mean()-i*100,
                        '%.1f' % (cell.AP_train[tinds[0]:].sum() *1000. / cell.tstop),
                        va='bottom', ha='left', fontsize=smallfontsize)
        
        if show_rates:
            ax.text(tvec[-1], 20,
                    r'rate (s$^{-1}$)', fontsize=smallfontsize,
                    va='bottom', ha='left')
            #ax.text(1, 1, r'rate (s$^{-1}$)', fontsize=smallfontsize,
            #        transform=ax.transAxes)

        for loc, spine in ax.spines.iteritems():
            if loc in ['right', 'top']:
                spine.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        ax.set_xlim(self.TRANSIENT, tstop)
        ax.set_ylim(top=20)
        
        #ax.axis(ax.axis('tight'))
        axis = ax.axis()
        ax.plot([axis[1], axis[1]], [axis[2],axis[2]+100], lw=4, color='k', clip_on=False)
        ax.text(axis[0]+np.diff(axis[:2])[0]*1.01, axis[2], '100 mV',
                 rotation='vertical', fontsize=smallfontsize, va='bottom')

        
        ax.set_title('somatic traces')
        

        ax.text(-0.05, 1.0, 'a',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax.transAxes)        
        

        '''plot the superimposed LFPs, added noise, bandpassfiltered data'''        
        #load some traces
        f = h5py.File(self.savefolder + '/ViSAPy_filterstep_1.h5',
                      'r')
        lfp_filtered = f['data'].value.T[:, tinds]
        f.close()
        
        #axis objects
        ax3 = fig.add_subplot(212)

        
        #scale=0.1
        vlim = abs(lfp_filtered).max()
        scale = 2.**np.round(np.log2(vlim))

        yticks = []
        yticklabels = []
        for i, x in enumerate(lfp_filtered):
            ax3.plot(tvec, x - i*scale, color=self.electrodeColors[i],
                     label=None, rasterized=True)
            yticks.append(-i*scale)
            yticklabels.append('ch. %i' % (i+1))
            i += 1
        
        axis=ax3.axis(ax3.axis('tight'))
        
        for loc, spine in ax3.spines.iteritems():
            if loc in ['right', 'top',]:
                spine.set_color('none')
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')
        
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(yticklabels)

        ax3.set_title('filtered extracellular potentials')
        ax3.set_xlabel(r'$t$ (ms)', labelpad=0.1)
        
        ax3.plot([axis[1], axis[1]], [axis[2],axis[2]+scale], lw=4, color='k', clip_on=False)
        ax3.text(axis[0]+np.diff(axis[:2])[0]*1.01, axis[2], '%.2f mV' % scale,
            rotation='vertical', va='bottom', fontsize=smallfontsize)

        ax3.text(-0.05, 1.0, 'b',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax3.transAxes)        
        
        ax3.set_xlim(self.TRANSIENT, tstop)

        return fig
    
    
    def plot_figure_12(self, cellindices=None, num_units=None):
        '''plot spike waveforms'''

        if cellindices is None:
            cellindices = np.arange(self.testdInst.POPULATION_SIZE)
        
        fig = plt.figure(figsize=(10*cellindices.size/16., 7))
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.07, top=0.92, wspace=0.2)
    
        #with many cells this is a bit cramped, plot only
        #num_units randomly chosen units, sorted  
        if num_units is not None:
            cellindices = np.random.permutation(cellindices)[:num_units]
            cellindices.sort()

        #spike sorting
        sp_waves, clust_idx, sp_win = self.get_sp_waves(filename=os.path.join(self.savefolder,
                                                                            'ViSAPy_filterstep_1.h5'))
            
    
        #loop over cells and find template with largest amplitude
        templates = []
        for cellkey in cellindices:
            if sp_waves['data'][:, clust_idx==cellkey, :].shape[1] > 0:
                templates.append(sp_waves['data'][:, clust_idx==cellkey, :].mean(axis=1))
        templates = np.array(templates)
        
        vlim = abs(templates).max() #*2
        #vlim = abs(sp_waves['data']).max()
        scale = 2.**np.round(np.log2(vlim))
        tvec = sp_waves['time']
        xvec = np.arange(self.testdInst.TEMPLATELEN)
        
        #keep aspect ratio equal between panels
        axis = (0,
                self.testdInst.TEMPLATELEN,
                sp_waves['data'][:,:,-1].min() - (self.testdInst.electrodeParameters['x'].size-1)*scale,
                sp_waves['data'][:,:,0].max())
        
    
        for count, cellkey in enumerate(cellindices):
            ax = fig.add_subplot(1, len(cellindices), count+1)
            zips = []
            yticks = []
            yticklabels = []
            for i in range(self.testdInst.electrodeParameters['x'].size):
                if sp_waves['data'][:, clust_idx==cellkey, :].shape[1] > 0:
                    for j, x in enumerate(sp_waves['data'][:, clust_idx==cellkey, i].T):
                        zips.append(zip(xvec, x - i*scale))
                        
                    ax.plot(xvec, sp_waves['data'][:, clust_idx==cellkey, i].mean(axis=1) - i*scale,
                             color='k', lw=1, clip_on=False, zorder=2)
                
                yticks.append(-i*scale)
                yticklabels.append('ch. %i' % (i+1))
                
            linecollection = LineCollection(zips,
                                            linewidths=(0.5),
                                            colors=self.colors[cellkey],
                                            rasterized=True,
                                            alpha=1,
                                            clip_on=False,
                                            zorder=0)    
            ax.add_collection(linecollection)
    
            for loc, spine in ax.spines.iteritems():
                if loc in ['right', 'top',]:
                    spine.set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks(yticks)
            ax.set_xticks([0, int(self.testdInst.TEMPLATELEN/2), self.testdInst.TEMPLATELEN])
            ax.set_xticklabels([0, int(self.testdInst.TEMPLATELEN/2)])
            if count == 0:
                ax.set_yticklabels(yticklabels)
            else:
                ax.set_yticklabels([])
    
            if count == 0:
                ax.set_xlabel('samples (-)', labelpad=0.1)
            
            ax.set_title('cell %i' % (cellkey+1) + '\n' + '%i APs' % sp_waves['data'][:, clust_idx==cellkey, 0].shape[1])        
            ax.axis(axis)
    
        ax.plot([axis[1], axis[1]], [axis[2], axis[2]+scale], 'k', lw=4, clip_on=False)
        ax.text(axis[0]+np.diff(axis[:2])[0]*1.03, axis[2], '%.2f mV' % scale,
                 rotation='vertical', fontsize=smallfontsize, va='bottom')
    
    
        return fig    

    
    def plot_figure_13(self,
                   cellindices=np.array([0]),
                   colors=['r'],
                   bins=10**np.linspace(np.log10(1), np.log10(1E3), 100)):
        '''plot MEA layout, templates, attenuation with distance, ISI'''
    
        #get some needed variables
        TD = self.testdInst
        cells = TD.read_lfp_cell_files(cellindices)
        markersize = 20
        xscaling = 0.5E1
        yscaling = 2E2
        
        #figure
        fig = plt.figure(figsize=(10, 10), frameon=False)
        
        #axes
        ax0 = fig.add_axes([0.025, 0.05, 0.325, 0.9], frame_on=False)
        ax1 = fig.add_axes([0.375, 0.05, 0.325, 0.9], frame_on=False)
        ax2 = fig.add_axes([0.825, 0.6, 0.15, 0.2])
        ax3 = fig.add_axes([0.825, 0.3, 0.15, 0.2])
        
        #clean up
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        remove_axis_junk(ax2)
        remove_axis_junk(ax3)
        
        
        ax0.plot(TD.paramsMapping['elec_x'], TD.paramsMapping['elec_y'],
                 'ko',
                 zorder=-1,
                 #markersize=markersize/2,
                 clip_on=False)
        
        for i in range(TD.POPULATION_SIZE):
            x = TD.pop_soma_pos[i]['xpos']
            y = TD.pop_soma_pos[i]['ypos']
            #mfc = self.colors[i]
            mfc = '0.7'
            ax0.plot(x, y, marker='*',
                     markeredgecolor='none',
                     markerfacecolor=mfc,
                     markersize=markersize,
                     zorder=0,
                     clip_on=False)
        
        for i in cellindices:
            x = TD.pop_soma_pos[i]['xpos']
            y = TD.pop_soma_pos[i]['ypos']
            mfc = colors[np.where(cellindices==i)[0]]
            ax0.plot(x, y, marker='*',
                     markeredgecolor='none',
                     markerfacecolor=mfc,
                     markersize=markersize,
                     zorder=0,
                     clip_on=False)
        
        
        axis = np.array(ax0.axis('equal'))
                        
        #annotate
        ax0.plot([80, 80], [0, 18], 'k', lw=4, clip_on=False)
        ax0.text(85, 18./2, r'18 $\mu$m',
                 va='center', rotation='vertical',
                 fontsize=smallfontsize,
                 clip_on=False)
        
        
        ax1.plot(TD.paramsMapping['elec_x'], TD.paramsMapping['elec_y'],
                 'ko',
                 clip_on=False)
        
        #extract and plot averaged spike waveforms of cell
        #params
        sp_win = ((np.array([0, self.testdInst.TEMPLATELEN]) -
                    self.testdInst.TEMPLATELEN*self.testdInst.TEMPLATEOFFS)
                * self.testdInst.cellParameters['dt']).tolist()
        
        #load recording
        f = h5py.File(os.path.join(self.savefolder, 'ViSAPy_filterstep_1.h5'))
        sp = {
            'data' : f['data'].value.T.astype('float32'),
            'FS' : f['srate'].value,
            'n_contacts' : f['data'].value.shape[1],
        }
        f.close()
        
        #load grount truth
        GT = np.loadtxt(os.path.join(self.savefolder, 'ViSAPy_ground_truth.gdf')).T
        clust_idx = GT[0, ].astype(int) - 1
        spt = {
            'data' : GT[1, ] / sp['FS']*1000
        }
        
        #extract waveforms
        sp_waves = spike_sort.extract.extract_spikes(sp, spt, sp_win)
        
        
        for cellindex in cellindices:
            cell = cells[cellindex]
            zips = []
            for x, y in cell.get_idx_polygons(projection=('x','y')):
                zips.append(zip(x, y))        
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors=colors[np.where(cellindices==cellindex)[0]], #self.colors[cellindex],
                                     clip_on=True)        
            ax1.add_collection(polycol)
        
        
            #mean waveforms of cell
            templates = sp_waves['data'][:, clust_idx==cellindex, :].mean(axis=1)
            t = np.arange(templates.shape[0]) * TD.cellParameters['dt']
            
            print(templates.min(), templates.max())
        
            #superimpose voltage trace at location of each contact
            xy = zip(TD.paramsMapping['elec_x'], TD.paramsMapping['elec_y'])
            for i, (x, y) in enumerate(xy):
                ax1.plot(x + 2 + t*xscaling, y + templates[:, i]*yscaling,
                         color='k',
                         zorder=1,
                         clip_on=False)
            
            ax1.set_title('single cell waveforms\n%i APs' % cell.AP_train.sum())
            
        
        #annotate
        ax1.plot([62, 60+t[-1]*xscaling], [-10, -10], 'k', lw=4, clip_on=False)
        ax1.text(62, -20, '%.1i ms' % t[-1], fontsize=smallfontsize, clip_on=False)
        
        ax1.plot([100, 100], [0, 20], 'k', lw=4, clip_on=False)
        ax1.text(105, 10, '%.2fmV' % (20. / yscaling),
                 va='center', rotation='vertical',
                 fontsize=smallfontsize,
                 clip_on=False)
        
        ax0.axis(axis)
        ax1.axis(axis)
        
        
        #plot decay of spike amplitudes
        for cellindex in cellindices:
            #get position of cell
            x = TD.pop_soma_pos[cellindex]['xpos']
            y = TD.pop_soma_pos[cellindex]['ypos']
            z = TD.pop_soma_pos[cellindex]['zpos']
            #contact positions
            xe = TD.paramsMapping['elec_x']
            ye = TD.paramsMapping['elec_y']
            ze = TD.paramsMapping['elec_z']
            
            #get the mean waveforms
            templates = sp_waves['data'][:, clust_idx==cellindex, :].mean(axis=1)
            xr = abs(templates).max(axis=0)
            xr /= xr.max()
            
            
            [[i]] = np.where(xr==xr.max())

            #distance to electrode with peak amplitude
            r = np.sqrt((xe-xe[i])**2 + (ye-ye[i])**2)
            sort = np.argsort(r)

            
            #sort with distance
            r = r[sort]
            xr = xr[sort]
            
            
            r_cutoff = 100
            
            ax2.plot(r, xr, '.',
                     color=colors[np.where(cellindices==cellindex)[0]],
                     clip_on=False)
        
        
            #superimpose best fit exponential function
            morefun = lambda X: np.exp(-r[r <= r_cutoff]/X[0]) * np.exp(X[1]) 
            fun = lambda X: (abs(np.log(morefun(X)) - np.log(xr[r <= r_cutoff]))).sum()

            res = minimize(fun, x0=[30., 1.], method='BFGS',)
            print('BFGS', res.x)
            
            ax2.plot(r[r <= r_cutoff], morefun(res.x), 'k',
                     label=r'$\exp(-r/%.3f) - %.3f$' % (res.x[0], res.x[1]))
            
            ax2.set_title('amplitude decay\n' + r'$\lambda_d=%.1f\mu$m' % res.x[0])
            
        
        ax2.semilogy()    
        axis = ax2.axis(ax2.axis('tight'))
        ax2.set_xlim(1)
        ax2.set_xticks(np.mgrid[0:axis[1]:50])
        ax2.set_xlabel(r'distance ($\mu$m)', labelpad=0.1)
        ax2.set_ylabel('norm. $|\mathrm{ampl.}|_\mathrm{max}$ (-)')
        
        
        #plot ISI histograms
        for cellindex in cellindices:
            cell = cells[cellindex]
            tvec = np.arange(cell.somav.size) * cell.dt
            ISI = np.diff(tvec[cell.AP_train==1])
            
            try:
                ax3.hist(ISI, bins=bins,
                        color=colors[np.where(cellindices==cellindex)[0]],
                        histtype='stepfilled',
                        alpha=self.alphas[cellindex],
                        edgecolor=colors[np.where(cellindices==cellindex)[0]],
                        )
            except:
                print('not enough spikes for ISI')
        
        ax3.semilogx()
        ax3.axis(ax3.axis('tight'))
        ax3.set_xlim(1E1, bins.max())
        ax3.set_xlabel('ISI (ms)', labelpad=0.1)
        ax3.set_ylabel('count (-)')
        
        
        
        #annotations
        ax0.set_title('MEA and population')
        ax3.set_title('ISI hist.')
        
        ax0.text(-0.025, 1.0, 'a',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax0.transAxes)
        
        
        ax1.text(-0.025, 1.0, 'b',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax1.transAxes)
        
        ax2.text(-0.3, 1.0, 'c',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax2.transAxes)
        
        ax3.text(-0.3, 1.0, 'd',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=18, fontweight='demibold',
            transform=ax3.transAxes)
        
        return fig


    def plot_population(self, cellindices=None,
                        figsize=(4, 10)):
        '''Use the pt3d information to plot the population of cells'''
        
        if cellindices is None:
            cellindices = np.arange(self.testdInst.POPULATION_SIZE)
        
        cells = {}
        for cellindex in cellindices:
            cells[cellindex] = self.testdInst.cellsim(cellindex,
                                                     return_just_cell=True)
        
        
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax = fig.add_subplot(111, frameon=False)
        
        
        
        #schematic plot of the population
        xoffs = 300
        for cellindex, cell in cells.iteritems():
            ax.plot(cell.somapos[0] - xoffs, cell.somapos[2],
                    'o', color=self.colors[cellindex],
                    markeredgecolor=self.colors[cellindex],
                    markersize=5,)
        
        #draw the outer boundary of the population
        mpop = self.testdInst.populationParameters
        
        #side view
        x = [-mpop['radius']-xoffs, mpop['radius']-xoffs,
             mpop['radius']-xoffs,  -mpop['radius']-xoffs,
             -mpop['radius']-xoffs]
        z = [ mpop['z_min'],        mpop['z_min'],
             mpop['z_max'],          mpop['z_max'],          mpop['z_min']]
        ax.plot(x, z, 'k', lw=1)
        
        #top down view
        radius = np.ones(21) * mpop['radius']
        theta = np.arange(21) * 2 * np.pi / 20.
        x = radius * np.cos(theta)
        x -= xoffs
        z = radius * np.sin(theta)
        z += mpop['z_max']
        z += 2*mpop['radius']
        
        ax.plot(x, z, 'k', lw=1)
        
        for cellindex, cell in cells.iteritems():
            ax.plot(cell.somapos[0] - xoffs,
                    cell.somapos[1] + mpop['z_max'] + 2*mpop['radius'],
                    'o', color=self.colors[cellindex],
                    markeredgecolor=self.colors[cellindex],
                    markersize=5,)
        
        
        ax.plot(self.testdInst.electrodeParameters['x'] - xoffs,
                self.testdInst.electrodeParameters['y'] + \
                                        mpop['z_max'] + 2*mpop['radius'],
                '.', marker='o', markersize=5, color='k')
        
        
        
        #contact points
        ax.plot(self.testdInst.electrodeParameters['x']-xoffs,
                self.testdInst.electrodeParameters['z'],
                '.', marker='o', markersize=5, color='k')
        
        #outline of electrode
        try:
            x_0 = mpop['r_z'][1, 1:-1]
            z_0 = mpop['r_z'][0, 1:-1]
            x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
            z = np.r_[1000, z_0[::-1], z_0[1:], 1000]
        except:
            x = np.r_[mpop['X'][0, 2:-2], mpop['X'][0, 3], mpop['X'][1, 3],
                      mpop['X'][1, 2:-2][::-1]]
            z = np.r_[mpop['Z'][2:-2], 1000, 1000, mpop['Z'][2:-2][::-1]]
        
        ax.fill(x, z, color=(0.5, 0.5, 0.5), lw=None, zorder=-1)
        
        
        #using the real morphologies 
        for cellindex, cell in cells.iteritems():
            zips = []
            for x, z in cell.get_idx_polygons():
                zips.append(zip(x, z))
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     color=self.colors[cellindex],
                                     rasterized=True,
                                     alpha=0.5,
                                     zorder=cell.somapos[1])
            ax.add_collection(polycol)
        
        #contact points
        ax.plot(self.testdInst.electrodeParameters['x'],
                self.testdInst.electrodeParameters['z'],
                '.', marker='o', markersize=5, color='k', zorder=0)
        
        #scalebar
        ax.plot([200, 200], [0, 100], 'k', lw=5)
        ax.text(200, 0, r'100 $\mu$m')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.axis(ax.axis('equal'))
        
        return fig


    def get_sp_waves(self, filename):
    
        #params
        sp_win = ((np.array([0, self.testdInst.TEMPLATELEN]) -
                    self.testdInst.TEMPLATELEN*self.testdInst.TEMPLATEOFFS)
                * self.testdInst.cellParameters['dt']).tolist()
    
        #load recording
        f = h5py.File(os.path.join(self.savefolder, 'ViSAPy_filterstep_1.h5'))
        sp = {
            'data' : f['data'].value.T.astype('float32'),
            'FS' : f['srate'].value,
            'n_contacts' : f['data'].value.shape[1],
        }
        f.close()
        
        #do not use detected spikes occurring prior to transient
        TRANSIENT = int(self.TRANSIENT * sp['FS'] / 1000)
        
        #load grount truth
        GT = np.loadtxt(os.path.join(self.savefolder, 'ViSAPy_ground_truth.gdf')).T
        inds = GT[1, ] >= TRANSIENT
        clust_idx = GT[0, inds].astype(int) - 1
        spt = {
            'data' : GT[1, inds] / sp['FS']*1000
        }
        
        #extract waveforms
        sp_waves = spike_sort.extract.extract_spikes(sp, spt, sp_win)
    
        return sp_waves, clust_idx, sp_win
