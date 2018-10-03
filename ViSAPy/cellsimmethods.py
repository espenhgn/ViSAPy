#!/usr/bin/env python
'''Class methods representing the neural populations'''
import os
import numpy as np
import LFPy
import neuron
import h5py
import scipy.signal as ss
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from ViSAPy import CorrelatedNoise, GDF, DriftCell
from matplotlib import cm
from mpi4py import MPI


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()



################## Class definitions ###########################################

class BenchmarkData(object):
    '''
    Main class object, allows set up of simulations, execute, and compile the
    results. This class is suitable for subclassing, e.g., create subclasses for
    custom cell simulation procedures
    
    Note that BenchmarkData.cellsim do not have any stimuli, it is simply a
    reference class
    '''
    def __init__(self,
                 cellParameters={
                        'v_init' : -80,
                        'passive' : False,
                        'nsegs_method' : None,
                        'dt' : 2**-5,
                        'tstart' : 0.,
                        'tstop' : 1000.,
                        'verbose' : False,
                    },
                 morphologies=[],
                 defaultrotation={None},
                 simulationParameters={},
                 populationParameters = {
                        'POPULATION_SIZE' : 6,
                        'radius' : 50,
                        'z_min' : -25,
                        'z_max' : 175,
                        'X' : np.array([ [0, 0,  0,  -40, -40, 0, 0],
                                       [0, 0,  0,   40,  40, 0, 0]]),
                        'Y' : np.array([ [0, 0, -50, -50, -50, 0, 0],
                                       [0, 0,  0,    0,   0, 0, 0]]),
                        'Z' : np.array([-np.inf, -50.01, -50, 0,
                                        1000, 1000.01, np.inf]),
                        'min_cell_interdist' : 5.,
                 },
                 electrodeParameters={
                        'x' : np.array([0, -1,
                                    np.sin(np.pi/6),np.sin(np.pi/6)]) * 25.,
                        'y' : np.array([0, 0,
                                    -np.cos(np.pi/6),np.cos(np.pi/6)]) * 25.,
                        'z' : np.array([-50., 0, 0, 0]),
                        'sigma' : 0.3,
                        'N' : np.array([[0,0,-1], [-1*np.cos(np.pi/9), 0,
                                                   -1*np.sin(np.pi/9)],
                                        [np.sin(np.pi/6) * np.cos(np.pi/9),
                                            -np.cos(np.pi/9) * np.cos(np.pi/9),
                                            -1 * np.sin(np.pi/9)],
                                        [-np.sin(np.pi/6) * np.cos(np.pi/9),
                                            -np.cos(np.pi/9) * np.cos(np.pi/9),
                                            1 * np.sin(np.pi/9)]]),
                        'r' : 7.,
                        'n' : 20,},
                 noiseFile=os.path.join('savedata', 'ViSAPy_noise.h5'),
                 filters=[{
                        'b' : np.array([1]),
                        'a' : np.array([1]),
                        'filterFun' : ss.lfilter
                    }],
                 savefolder='savedata',
                 default_h5_file = 'lfp_cell_%.3i.h5',
                 nPCA = 2,
                 TEMPLATELEN = 100,
                 TEMPLATEOFFS = 0.3,
                 spikethreshold = 3.,
                 ):
        '''
        class BenchmarkData initialization

        Main class object, allows set up of simulations, execute, and compile the
        results. This class is suitable for subclassing, e.g., create subclasses for
        custom cell simulation procedures
        
        Note that BenchmarkData.cellsim do not have any stimuli, it is simply a
        reference class
        
        kwargs:
        ::
            
            cellParameters : dict
                LFPy.TemplateCell kwargs
            morphologies : list
                list of morphologies, if more than one entry, will permute order
            defaultrotation : dict
                default rotation angles {'x' : rad, 'y' : rad}, z-axis is random
            simulationParameters : dict
                extra kwargs passed to LFPy.TemplateCell.simulate method
            populationParameters : dict
                kwargs for assessing cell locations in population
            electrodeParameters : dict
                LFPy.RecExtElectrode kwargs
            noiseFile : str
                path to hdf5 file containing extracellular noise
            filters : list of dicts
                filter coeffs; each dict must have 'b', 'a', 'filterFun' kwargs
            savefolder : str
                default folder for storing simulation results
            default_h5_file : str with %.3i
                cell simulations will be stored using this formatting
            nPCA : int
                number of principal components to be calculated
            TEMPLATELEN : int
                number of time samples per spike waveform
            TEMPLATEOFFS : float on [0., 1.]
                spike waveform offset vs spike time
            spikethreshold : float
                set spike threshold as multiple of signal rms
        '''
        self.cellParameters = cellParameters
        self.morphologies = morphologies
        self.defaultrotation = defaultrotation
        self.simulationParameters = simulationParameters
        self.populationParameters = populationParameters
        self.POPULATION_SIZE = populationParameters['POPULATION_SIZE']
        self.electrodeParameters = electrodeParameters
        self.filters = filters
        self.savefolder = savefolder
        self.nPCA = nPCA
        self.TEMPLATELEN = TEMPLATELEN
        self.TEMPLATEOFFS = TEMPLATEOFFS
        self.spikethreshold = spikethreshold
        if default_h5_file.find('%.') <= 0 or default_h5_file.find('.h5') <= 0:
            raise Exception, "%s must include '%.3i' and file-ending '.h5'" % \
                                default_h5_file
        else:
            self.default_h5_file = default_h5_file
        
        #put revision info in savefolder
        if RANK == 0:
            try:
                os.system('git rev-parse HEAD -> %s/testdataRevision.txt' % \
                        self.savefolder)
            except:
                pass
        COMM.barrier()
    

        #certain LFPy.TemplateCell attributes being stored for each instance
        self.savelist = [
            'somav',
            'dt',
            'somapos',
            'x',
            'y',
            'z',
            'LFP',
            'morphology',
            'default_rotation',
            'electrodecoeff',]


        #using these colors and alphas:        
        self.colors = []
        for i in range(self.POPULATION_SIZE):
            i *= 256.
            if self.POPULATION_SIZE > 1:
                i /= self.POPULATION_SIZE - 1.
            else:
                i /= self.POPULATION_SIZE
            self.colors.append(cm.Set1(int(i)))
        
        self.alphas = np.ones(self.POPULATION_SIZE)

        self.pop_soma_pos = self.set_pop_soma_pos()
        self.shuffled_morphologies = self.set_shuffled_morphologies()
        self.rotations = self.set_rotations()
    
    
    def run(self):
        '''
        Distribute individual cell simulations across MPI ranks
        
        This method takes no keyword arguments.
        '''
        for cellindex in range(self.POPULATION_SIZE):
            #in case more cores than cells are available,
            #start cell simulation on every fixed interval RANK
            RANKINTERVAL = divmod(SIZE, self.POPULATION_SIZE)[0]
            if SIZE >= self.POPULATION_SIZE:
                if divmod(cellindex*RANKINTERVAL, SIZE)[1] == RANK:
                    self.cellsim(cellindex)
            else:
                if divmod(cellindex, SIZE)[1] == RANK:
                    self.cellsim(cellindex)
                    
        #sync
        COMM.Barrier()

    
    def cellsim(self, cellindex, return_just_cell=False):
        '''
        dummy cell simulation without any stimulus, but can serve as a
        reference simulation case
        
        Keyword arguments:
        ::
            
            cellindex : int
                cell index between 0 and POPULATION_SIZE-1
            return_just_cell : bool
                If True, return only the LFPy.Cell object
                if False, run full simulation, return None
    
        Returns:
        ::
            
            None, if return_just_cell is False
            LFPy.Cell-object, if return_just_cell is True        
        '''
        electrode = LFPy.RecExtElectrode(**self.electrodeParameters)
        
        cellParameters = self.cellParameters.copy()
        cellParameters.update({
            'morphology' : self.shuffled_morphologies[cellindex]})
            
        cell = LFPy.Cell(**cellParameters)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])    
        
        if return_just_cell:
            return cell
        else:
            if self.simulationParameters.has_key('to_file'):
                if self.simulationParameters['to_file']:
                    cell.simulate(electrode,
                                  file_name=os.path.join(self.savefolder,
                                        self.default_h5_file) % (cellindex),
                                  **self.simulationParameters)
                else:
                    cell.simulate(electrode, **self.simulationParameters)
                    cell.LFP = electrode.LFP
            else:
                cell.simulate(electrode, **self.simulationParameters)
                cell.LFP = electrode.LFP
        
            
            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z
            cell.electrodecoeff = electrode.electrodecoeff

            #access file object
            f = h5py.File(os.path.join(self.savefolder,
                                self.default_h5_file) % (cellindex),
                          'a')
            
            if self.simulationParameters.has_key('to_file'):
                if self.simulationParameters['to_file']:
                    f['LFP'] = f['electrode000']

            
            #save stuff from savelist
            for attrbt in self.savelist:
                try:
                    del(f[attrbt])
                except:
                    pass
                try:
                    f[attrbt] = getattr(cell, attrbt)
                except:
                    f[attrbt] = str(getattr(cell, attrbt))
            
            
            print 'SIZE %i, RANK %i, Cell %i, Min LFP: %.3f, Max LFP: %.3f' % \
                        (SIZE, RANK, cellindex,
                        f['LFP'].value.min() if 'LFP' in f.keys() else f['electrode000'].value.min(),
                        f['LFP'].value.max() if 'LFP' in f.keys() else f['electrode000'].value.max())
            
            f.close()
            
            print 'Cell %s saved to file' % cellindex

    
    def set_pop_soma_pos(self):
        '''
        set pop_soma_pos using randomly drawn locations from
        method draw_rand_pos().
        
        This method takes no keyword arguments.
        '''
        if RANK == 0:
            if np.any(np.array(self.populationParameters.keys()) == 'X'):
                pop_soma_pos = self.draw_rand_pos_square()
            else:
                pop_soma_pos = self.draw_rand_pos()
        else:
            pop_soma_pos = None
        return COMM.bcast(pop_soma_pos, root=0)
        
    
    def set_shuffled_morphologies(self):
        '''
        Create list of shuffled morphologies for each cell in population.
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            list of strings, each string being path to a morphology file
        '''
        if RANK == 0:
            shuffled_morphologies = self.shufflemorphos(self.morphologies,
                                                    n = self.POPULATION_SIZE)
        else:
            shuffled_morphologies = None
        return COMM.bcast(shuffled_morphologies, root=0)
    
    
    def set_rotations(self):
        '''
        Append some random z-axis rotations for each cell in population
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            list of dicts,  on the form [{'z' : float}, ..], assigning a
            random rotation angle around z-axis applied to each cell.
        '''
        if RANK == 0:
            rotations = []
            for i in range(self.POPULATION_SIZE):
                defaultrot = self.defaultrotation.copy()
                defaultrot.update({'z' : np.random.rand() * 2 * np.pi})
                rotations.append(defaultrot)
        else:
            rotations = None
        return COMM.bcast(rotations, root=0)
    
        
    def shufflemorphos(self, files, n=100):
        '''
        reorder and elongate list files to n entries, scrambling the order
        
        Keyword arguments:
        ::
            
            files : list of strings, each string is path to a morphology file
            n : int, number of entries in output
        
        Returns:
        ::
            
            list of strings, each string is path to a morphology file
        '''
        filelist = []
        low = np.size(files)
        for i in range(n):
            filelist.append(files[np.random.randint(low)])
        return filelist

    
    def return_spiketrains(self, v, v_t=-30., TRANSIENT=500.):
        '''
        Takes voltage trace v, and some optional voltage threshold v_t,
        returning spike train array of same length as v.
        
        The spike time is identified as the local maxima of the derivative of
        the somatic traces.
        
        Keyword arguments:
        ::
            
            v : np.ndarray, vector containing somatic potential of a cell
            v_t : float, action-potential detection threshold
            TRANSIENT : float, discard detected spikes before this time 
        
        Returns:
        ::
            
            np.ndarray, binary vector, where 1 values correspond the
            spike time index, assessed as the index maximizing dv/dt on up slope
            for every event
        '''
        AP_train = np.zeros(v.shape, dtype=int)
        
        #alternative, looping over every element to check for crossings
        u = np.where((v[:-1] < v_t) & (v[1:] >= v_t))[0]

        #mask spikes occurring prior to TRANSIENT
        u = u[u >= int(TRANSIENT / self.cellParameters['dt'])]
        
        
        pre = -int(self.TEMPLATELEN * self.TEMPLATEOFFS)
        post = self.TEMPLATELEN + pre
        
        #mask spikes occurring at either start or end of signal
        u = u[(u >= -pre) & (u <= v.size-post)]
        
        #splitting u in intervals if there are more than 1 AP,
        #filling w with these intervals:
        for i in u:
            inds = np.arange(i + pre, i + post)
            w = v[inds[inds < v.size]]
            
            AP_train[inds[:-1][np.diff(w) == np.diff(w).max()]] = 1
                
        return AP_train
    
    
    def calc_min_cell_interdist(self, x, y, z):
        '''
        calculate cell interdistance from input coordinates.
        
        
        Keyword arguments:
        ::
            
            x,y,z : np.ndarrays, carthesian coordinates of center each soma
        
        Returns:
        ::
            
            np.ndarray, minimum cell-to-cell inter distance for each unit
        
        '''
        min_cell_interdist = np.zeros(self.POPULATION_SIZE)
        
        for i in range(self.POPULATION_SIZE):
            cell_interdist = np.sqrt((x[i] - x)**2
                    + (y[i] - y)**2
                    + (z[i] - z)**2)
            cell_interdist[i] = np.inf
            min_cell_interdist[i] = cell_interdist.min()
        
        return min_cell_interdist

    
    def draw_rand_pos(self):
        '''
        draw some random location within radius, z_min, z_max,
        and constrained by min_r and the minimum cell interdistance.
        Returned argument is a list of dicts [{'xpos', 'ypos', 'zpos'}, ]
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            list of dictionaries on the form
            [{'x' : x[i], 'y' : y[i], 'z' : z[i]}, ...]
            containing randomized x,y,z-coordinates of each cell body
        '''
        min_cell_interdist = self.populationParameters['min_cell_interdist']
        radius = self.populationParameters['radius']
        z_min = self.populationParameters['z_min']
        z_max = self.populationParameters['z_max']
        min_r= self.populationParameters['r_z']
        x = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        y = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        z = np.random.rand(self.POPULATION_SIZE)*(z_max - z_min) + z_min
        min_r_z = {}
        if min_r.size > 0:
            if type(min_r) == type(np.array([])):
                j = 0
                for j in range(min_r.shape[0]):
                    min_r_z[j] = np.interp(z, min_r[0,], min_r[1,])
                    if j > 0:
                        [w] = np.where(min_r_z[j] < min_r_z[j-1])
                        min_r_z[j][w] = min_r_z[j-1][w]
                minrz = min_r_z[j]
        else:
            minrz = np.interp(z, min_r[0], min_r[1])
                
        R_z = np.sqrt(x**2 + y**2)
        
        #want to make sure that no somas are in the same place.
        cell_interdist = self.calc_min_cell_interdist(x, y, z)
        
        [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
            cell_interdist < min_cell_interdist))
            
        while len(u) > 0:
            for i in range(len(u)):
                x[u[i]] = (np.random.rand()-0.5)*radius*2
                y[u[i]] = (np.random.rand()-0.5)*radius*2
                z[u[i]] = np.random.rand()*(z_max - z_min) + z_min
                if type(min_r) == type(()):
                    for j in range(np.shape(min_r)[0]):
                        min_r_z[j][u[i]] = \
                                np.interp(z[u[i]], min_r[0,], min_r[1,])
                        if j > 0:
                            [w] = np.where(min_r_z[j] < min_r_z[j-1])
                            min_r_z[j][w] = min_r_z[j-1][w]
                        minrz = min_r_z[j]				
                else:
                    minrz[u[i]] = np.interp(z[u[i]], min_r[0,], min_r[1,])
            R_z = np.sqrt(x**2 + y**2)
            
            #want to make sure that no somas are in the same place.
            cell_interdist = self.calc_min_cell_interdist(x, y, z)
        
            [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
                cell_interdist < min_cell_interdist))
    
        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'x' : x[i], 'y' : y[i], 'z' : z[i]})
        
        return soma_pos
    
    
    def draw_rand_pos_square(self):
        '''
        Draw random cell body locations, assuming an electrode occupying a
        square cross section with width and depth defined by X,Y,Z.
        
        killzone arg sets minimum distance to any contact size, presumably due
        to 
        
        locations are drawn within a cylindric volume with radius and depths as
        defined in populationParameters.
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            list of dictionaries on the form
            [{'x' : x[i], 'y' : y[i], 'z' : z[i]}, ...]
            containing randomized x,y,z-coordinates of each cell body
            
        '''

        X = self.populationParameters['X']
        Y = self.populationParameters['Y']
        Z = self.populationParameters['Z']
        if self.populationParameters.has_key('killzone'):
            killzone = self.populationParameters['killzone']
        else:
            killzone = 0.
        min_cell_interdist = self.populationParameters['min_cell_interdist']
        radius = self.populationParameters['radius']
        z_min = self.populationParameters['z_min']
        z_max = self.populationParameters['z_max']
        x = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        y = (np.random.rand(self.POPULATION_SIZE)-0.5)*radius*2
        z = np.random.rand(self.POPULATION_SIZE)*(z_max - z_min) + z_min

        #check which units are inside XYZ:
        xmin = np.interp(z, Z, X[0, ]-killzone)
        xmax = np.interp(z, Z, X[1, ]+killzone)
        ymin = np.interp(z, Z, Y[0, ]-killzone)
        ymax = np.interp(z, Z, Y[1, ]+killzone)
        inside_electrode = np.logical_and((x > xmin) & (x < xmax),
                            (y > ymin) & (y < ymax))       
   
        #want to make sure that no somas are in the same place.
        cell_interdist = self.calc_min_cell_interdist(x, y, z)

        R_z = np.sqrt(x**2 + y**2)
        
        [u] = np.where((R_z > radius) |
                        (cell_interdist < min_cell_interdist) |
                        inside_electrode) 

        j = 0
        while len(u) > 0:
            for i in range(len(u)):
                x[u[i]] = (np.random.rand()-0.5)*radius*2
                y[u[i]] = (np.random.rand()-0.5)*radius*2
                z[u[i]] = np.random.rand()*(z_max - z_min) + z_min

            R_z = np.sqrt(x**2 + y**2)

            #check which units are inside XYZ:
            xmin = np.interp(z, Z, X[0, ]-killzone)
            xmax = np.interp(z, Z, X[1, ]+killzone)
            ymin = np.interp(z, Z, Y[0, ]-killzone)
            ymax = np.interp(z, Z, Y[1, ]+killzone)
            inside_electrode = np.logical_and((x > xmin) & (x < xmax),
                                            (y > ymin) & (y < ymax))
            
            #want to make sure that no somas are in the same place.
            cell_interdist = self.calc_min_cell_interdist(x, y, z)
        
            [u] = np.where((R_z > radius) | 
                            (cell_interdist < min_cell_interdist) | 
                            inside_electrode) 
    
            j += 1
            if j == 1000:
                raise Exception, 'after %i iters, can not position somas' % j
            
        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'x' : x[i], 'y' : y[i], 'z' : z[i]})
    
        return soma_pos
    
    
    def calc_AP_trains(self, cells):
        '''
        Extract spike times of each cell in input dictionary cells from
        somatic traces.
        
        Keyword arguments:
        ::
            
            cells, dict of LFPy.Cell objects as returned by
                method read_lfp_cell_files() indexed by cellindex
                starting at zero
        
        Returns:
        ::
            
            np.ndarray, first column is cell id, second column spike time index
        
        '''
        AP_train_sparse =  lil_matrix((np.size(cells.keys()),
                            cells[cells.keys()[0]].AP_train.size), dtype=bool)
        i = 0
        for cell in cells.itervalues():
            AP_train_sparse[i, cell.AP_train.nonzero()[0]] = 1
            print '.',
            i += 1
        
        #reworking AP-train to work with page
        x = np.array([AP_train_sparse.nonzero()[1],
                      AP_train_sparse.nonzero()[0] + 1], dtype=int).T.tolist()
        x.sort()
        if np.ndim(x) < 2:
            return np.array(x)
        else:
            return np.array(np.fliplr(x))


    def calc_lfp_el_pos(self, cells):
        '''
        Sum all single-cell contributions to the extracellular potential in
        each electrode location
        
        Keyword arguments:
        ::
            
            cells : dict of LFPy.Cell-like objects as returned by
                method read_lfp_cell_files() indexed by cellindex
                starting at zero
        '''
        lfp = np.zeros(cells[cells.keys()[0]].LFP.value.shape, dtype='float32')
        for cell in cells.itervalues():
            lfp += cell.LFP.value
            #close file object
            cell.f.close()
            print '.',
        
        return lfp


    def read_lfp_cell_files(self, cellindices=None):
        '''
        Create handles to hdf5 file output from each cell and append to
        each LFPy.Cell-like object, avoiding in memory loading of e.g., LFPs.
        
        File objects may remain open but are closed by calc_lfp_el_pos()
        
        Keyword arguments:
        ::
            
            cellindices : np.ndarray, dtype int
                indices of each cell in population, or subset of cells 
        
        '''
        if cellindices is None:
            cellindices = np.arange(self.POPULATION_SIZE)
        
        cells = {}
        for cellindex in cellindices:
            cells[cellindex] = self.cellsim(cellindex, return_just_cell=True)
            
            f = h5py.File(os.path.join(self.savefolder,
                                            self.default_h5_file) % \
                                        (cellindex), 'r+')
            print(os.path.join(self.savefolder,
                               self.default_h5_file) % (cellindex))
            for k in f.iterkeys():
                if k in ['LFP', 'electrode000']:
                    setattr(cells[cellindex], 'LFP', f[k])
                else:
                    setattr(cells[cellindex], k, f[k].value)
                    try:
                        assert(hasattr(cells[cellindex], k))
                    except AssertionError as ae:
                        raise ae('cell {} do not have attribute {}'.format(cellindex, k))

            #attach file object
            setattr(cells[cellindex], 'f', f)
            
            
        
        #recalculate AP_trains:
        for cell in cells.itervalues():
            setattr(cell, 'AP_train', self.return_spiketrains(cell.somav))
                
        
        return cells    

    
    def calc_somavs(self, cells):
        '''
        put all somavs from dict cells in one big array


        Keyword arguments:
        ::
            
            cells : dict of LFPy.Cell-like objects as returned by
                method read_lfp_cell_files() indexed by cellindex
                starting at zero

        Returns:
        ::
            
            np.ndarray, somatic voltages of each cell
        '''
        somavs = np.zeros((len(cells.keys()),
                           cells[cells.keys()[0]].somav.size),
            dtype='float32')
        
        i = 0
        for cell in cells.itervalues():
            somavs[i, ] = cell.somav
            i += 1
            print '.',

        return somavs
        
        
    def collect_data(self, cellindices=None, analysis=False):
        '''
        Read results from cellsim, calculate full lfp, add noise, save to files.
        
        Keyword arguments:
        ::
            
            cellindices : np.ndarray, dtype int
                indices of each cell in population, or subset of cells
        
        File output:
        ::
            
            ViSAPy_somatraces.h5
                data : somatic traces of each cell in the population
                    in units of mV
                srate : sampling rate of signal
            ViSAPy_noiseless.h5
                data : sum of predicted extracellular potentials in units of mV
                    in each channel
                srate : sampling rate of signal
            ViSAPy_noise.h5
                data : synthesized noise in each channel in units of mV
                srate : sampling rate of signal
            ViSAPy_nonfiltered.h5
                data : predicted potentials superimposed with synthetic noise
                srate : sampling rate of signal
            ViSAPy_filterstep_*.h5 :
                data : same as ViSAPy_nonfiltered.h5, but band-pass filtered
                srate : sampling rate of signal
            ViSAPy_somapos.gdf
                text file containing x,y,z-coordinates of each cell
            ViSAPy_ground_truth.gdf
                text file with spiking ground truth, first column is unit,
                second column spike time
            ViSAPy_ground_truth_threshold.gdf
                text file with spiking ground truth, first column is unit,
                second column spike time, however with units below a certain
                amplitude threshold removed
            
        
        '''
        if RANK == 0:
            #using cellindices throughout
            if cellindices is None:
                cellindices = np.arange(self.POPULATION_SIZE)
            
            #cells = self.read_lfp_AP_trains_cell_files()
            cells = self.read_lfp_cell_files(cellindices)
            print 'cells ok'
            
            #remove vmem, imem if they exist, they are not needed here
            for cell in cells.itervalues():
                if hasattr(cell, 'vmem'):
                    del cell.vmem
                if hasattr(cell, 'imem'):
                    del cell.imem
            
        
            #calculate lfp from all cell contribs
            lfp = self.calc_lfp_el_pos(cells)    
            print 'lfp ok'
            
            #gather action potentials from all cells
            AP_trains = self.calc_AP_trains(cells)
            print 'AP_trains ok'
            
            somavs = self.calc_somavs(cells)
            print 'soma potentials ok'
            
            
            f = h5py.File(self.savefolder + '/ViSAPy_somatraces.h5', 'w')
            f.create_dataset('data', data=somavs, compression=4)
            #f['data'] = somavs
            f['srate'] = int(1000 / self.cellParameters['dt'])
            f.close()
            print 'save somatraces ok'
            
            #saving
            f = h5py.File(self.savefolder + '/ViSAPy_noiseless.h5', 'w')
            f['srate'] = int(1000 / self.cellParameters['dt'])
            f.create_dataset('data', data=lfp.T, compression=4)
            #f['data'] = lfp.T
            grp = f.create_group('electrode')
            grp['x'] = self.electrodeParameters['x']
            grp['y'] = self.electrodeParameters['y']
            grp['z'] = self.electrodeParameters['z']
            f.close()
            print 'save lfp ok'
            
            #add noise to all channels
            f = h5py.File(os.path.join(self.savefolder,
                                       'ViSAPy_noise.h5'), 'r')
            lfp += f['data'].value[:lfp.shape[0], :lfp.shape[1]]
            f.close()
            print 'noise ok'

        
            
            #save the somatic placements:
            pop_soma_pos = np.zeros((self.POPULATION_SIZE, 3))
            keys = ['x', 'y', 'z']
            for i in range(self.POPULATION_SIZE):
                for j in range(3):
                    pop_soma_pos[i, j] = self.pop_soma_pos[i][keys[j]]
            np.savetxt(self.savefolder + '/ViSAPy_somapos.gdf', pop_soma_pos)
            
            
            # save the contact placements and additional parameters:
            f = h5py.File(os.path.join(self.savefolder,
                                       'electrodeParameters.h5'), 'w')
            for key, value in self.electrodeParameters.items():
                f[key] = value
            f.close()

            
            #saving lfp before filtering
            f = h5py.File(self.savefolder + '/ViSAPy_nonfiltered.h5', 'w')
            f['srate'] = int(1000 / self.cellParameters['dt'])
            f.create_dataset('data', data=lfp.T, compression=4)
            grp = f.create_group('electrode')
            grp['x'] = self.electrodeParameters['x']
            grp['y'] = self.electrodeParameters['y']
            grp['z'] = self.electrodeParameters['z']
            f.close()
            print 'save noisy lfp ok'
    
            i = 0
            #looping over filters, and writing data, numbered
            for fltr in self.filters:
                lfp = fltr['filterFun'](fltr['b'], fltr['a'],
                                              lfp).astype('float32')
                f = h5py.File(self.savefolder +
                              '/ViSAPy_filterstep_%i.h5' % i, 'w')
                f['srate'] = int(1000 / self.cellParameters['dt'])
                f.create_dataset('data', data=lfp.T, compression=4)
                grp = f.create_group('electrode')
                grp['x'] = self.electrodeParameters['x']
                grp['y'] = self.electrodeParameters['y']
                grp['z'] = self.electrodeParameters['z']
                f.close()
                print 'save lfp filter %i ok' % i
                i += 1
                
                            
            
            np.savetxt(os.path.join(self.savefolder,
                                    'ViSAPy_ground_truth.gdf'),
                       AP_trains, fmt='%i')
            print 'save ground truth all cells ok'
            

            try:
                #gather action potential times from cells with proper amplitude
                AP_trains_threshold = []
                for x in AP_trains:
                    if spike_amplitudes[x[0]-1] >= stds[x[0] - 1
                                                        ]*self.spikethreshold:
                        AP_trains_threshold.append(x)
                    else:
                        pass
                AP_trains_threshold = np.array(AP_trains_threshold)
                print 'thresholded AP_trains ok'
                
                np.savetxt(os.path.join(self.savefolder,
                                        'ViSAPy_ground_truth_threshold.gdf'),
                           AP_trains_threshold, fmt = '%i')
            except:
                pass
        #sync
        COMM.Barrier()
              


    
    
class BenchmarkDataLayer(BenchmarkData):
    '''
    class BenchmarkDataLayer, inherites class BenchmarkData.
    
    This class expand upon the functionality of the parent class by interfacing
    spike events in e.g., a spiking-neuron network model
    
    
    
    class object used for the L5 tetrode test data.
    kwargs are passed on to parent class BenchmarkData
    '''
    def __init__(self,
                 templatefiles=[
                    ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                     'L5bPCmodelsEH/models/L5PCtemplate.hoc'],],
                 networkInstance = None,
                 synapseParametersEx={
                        'e' : 0,
                        'syntype' : 'Exp2Syn',
                        'tau1' : 1.,
                        'tau2' : 3.,
                        'weight' : 0.001,
                        'section' : 'alldend',
                        'nPerArea' : 10.0E-3,
                    },
                 synapseParametersIn={
                        'e' : -80,
                        'syntype' : 'Exp2Syn',
                        'tau1' : 1.,
                        'tau2' : 12.,
                        'weight' : 0.0025,
                        'color' : 'b',
                        'marker' : '.',
                        'section' : 'alldend',
                        'nPerArea' : 2.5E-3,
                    },
                 driftParameters = None,
                    #{
                    #    'driftInterval' : 1000,
                    #    'driftShift' : 10,
                    #    'driftOnset' : 0,
                    #},
                 **kwargs):
        '''
        class initialization. Additional kwargs passed on to
        parent class testdata.BenchmarkData
        
        Keyword arguments:
        ::
            
            templatefiles : nested list
                each entry list of 'hoc' code, permuted and passed to
                class LFPy.TemplateCell
            networkInstance : population.Network or child thereof class instance
                will use arrays networkInstance.nodes_ex and
                networkInstance.nodes_in
            synapseParametersEx : dict
                LFPy.Synapse keyword arguments for excitatory connections
                including synapse density 'nPerArea'
            synapseParametersIn : dict
                LFPy.Synapse keyword arguments for inhibitory connections
                including synapse density 'nPerArea'
            driftParameters : None/dict
                if not None, switch on discrete electrode drift with
                dict containing entries
                    driftInterval : (ms)
                    driftShift : (mum)
                    driftOnset : (mum)
                       
        '''
        #initialize parent class
        BenchmarkData.__init__(self, **kwargs)

        #set some attributes
        self.templatefiles = templatefiles
        self.networkInstance = networkInstance
        self.nodes_ex = np.array(self.networkInstance.nodes_ex).flatten()
        self.nodes_in = np.array(self.networkInstance.nodes_in).flatten()
        self.synapseParametersEx = synapseParametersEx
        self.synapseParametersIn = synapseParametersIn
        self.driftParameters = driftParameters
        
        #generate some model input
        self.shuffled_templatefiles = self.set_shuffled_templatefiles()
        self.synIdxEx, self.synIdxIn = self.fetchSynIdx()
        self.SpCellsEx, self.SpCellsIn = self.fetchSpCells()
        
        #store some additional LFPy.TemplateCell attributes
        self.savelist += ['templatefile', 'templateargs']
        
        
    def set_shuffled_templatefiles(self):
        '''
        return custom_codes repeated to POPULATION_SIZE entries, and
        in random order.
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            list of strings, each string a path to a NEURON template file        
        '''
        if RANK == 0:
            shuffled_templatefiles = self.shuffle_templatefiles()
        else:
            shuffled_templatefiles = None
        return COMM.bcast(shuffled_templatefiles, root=0)

    
    def shuffle_templatefiles(self):
        '''
        reorder and elongate list custom_codes to n entries,
        scrambling the ordering

        This method takes no keyword arguments.

        Returns:
        ::
            
            list of strings, each string a path to a NEURON template file
        '''
        n = self.POPULATION_SIZE
        codelist = []
        low = np.shape(self.templatefiles)[0]
        for i in range(n):
            codelist.append(self.templatefiles[np.random.randint(low)])
        return codelist

    
    def cellsim(self, cellindex, return_just_cell = False):
        '''
        do the actual simulations of LFP, using synaptic spike times from a
        network simulation

        Keyword arguments:
        ::
            
            cellindex : int
                cell index between 0 and POPULATION_SIZE-1
            return_just_cell : bool
                If True, return only the LFPy.Cell object
                if False, run full simulation, return None
    
        Returns:
        ::
            
            None, if return_just_cell is False
            LFPy.Cell-object, if return_just_cell is True                
        '''
        electrode = LFPy.RecExtElectrode(**self.electrodeParameters)
        
        morphology = self.shuffled_morphologies[cellindex]
        
        #check if we have drift
        if self.driftParameters is not None:
            Cell = DriftCell
        else:
            Cell = LFPy.TemplateCell
        
        cellParameters = dict(self.cellParameters)
        cellParameters.update(dict(
            morphology = self.shuffled_morphologies[cellindex],
            templatefile = self.shuffled_templatefiles[cellindex],
            templateargs = self.shuffled_morphologies[cellindex]
        ))
        
        #set up cell object
        cell = Cell(**cellParameters)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])    
        
                
        #make the axon point downwards, start at soma mid point, comp for diam
        self.point_axon_down(cell)   
                    
        if return_just_cell:
            #with several cells, NEURON can only hold one cell at the time
            allsecnames = []
            allsec = []
            for sec in cell.allseclist:
                allsecnames.append(sec.name())
                for i in range(sec.nseg):
                    allsec.append(sec.name())
            cell.allsecnames = allsecnames
            cell.allsec = allsec
            return cell
        else:        
            #insert synapses
            self.insert_synapses(cell, self.synapseParametersEx.copy(),
                        self.synIdxEx[cellindex],
                        SpCell=self.SpCellsEx[cellindex],
                        SpTimes=os.path.join(self.savefolder, 'SpTimesEx.db'))
            self.insert_synapses(cell, self.synapseParametersIn.copy(),
                        self.synIdxIn[cellindex],
                        SpCell=self.SpCellsIn[cellindex],
                        SpTimes=os.path.join(self.savefolder, 'SpTimesIn.db'))


            if self.driftParameters is not None:
                #set up a 3D array dotprodcoeffs incorporating electrode drift
                #at fixed intervals
                driftCount = int(divmod(self.cellParameters['tstop'],
                                        self.driftParameters['driftInterval'
                                                             ])[0])
                #in case driftInterval > tstop:
                if driftCount == 0:
                    driftCount += 1
                dotprodcoeffs = []
                #dummy-membrane currents and tvec:
                cell.imem = np.eye(cell.totnsegs)
                cell.tvec = np.arange(cell.totnsegs)*cell.dt
                print 'calculating dot-prod coefficients',
                for i in range(driftCount):
                    elParameters = self.electrodeParameters.copy()
                    elParameters['z'] += i * self.driftParameters['driftShift']
                    elParameters['z'] += self.driftParameters['driftOnset']
                    tempelectrode =LFPy.RecExtElectrode(cell, **elParameters)
                    tempelectrode.calc_lfp()
                    dotprodcoeffs.append(tempelectrode.LFP)
                    print '.',
                    
                dotprodcoeffs = np.array(dotprodcoeffs)
                #del dummy variables
                del cell.imem, cell.tvec
                print 'done'
            else:
                pass

            if self.driftParameters is not None:
                if self.simulationParameters.has_key('to_file'):
                    if self.simulationParameters['to_file']:
                        cell.simulate(file_name=os.path.join(self.savefolder,
                                            self.default_h5_file) % (cellindex),
                                dotprodcoeffs = dotprodcoeffs,
                                driftInterval = self.driftParameters['driftInterval'],
                                **self.simulationParameters)
                    else:
                        cell.simulate(dotprodcoeffs = dotprodcoeffs,
                                driftInterval = self.driftParameters['driftInterval'],
                                **self.simulationParameters)
                        cell.LFP = cell.dotprodresults
                else:
                    cell.simulate(dotprodcoeffs = dotprodcoeffs,
                                driftInterval = self.driftParameters['driftInterval'],
                                **self.simulationParameters)
                    cell.LFP = cell.dotprodresults
            else:
                if self.simulationParameters.has_key('to_file'):
                    if self.simulationParameters['to_file']:
                        cell.simulate(electrode,
                                      file_name=os.path.join(self.savefolder,
                                            self.default_h5_file) % (cellindex),
                                      **self.simulationParameters)
                    else:
                        cell.simulate(electrode, **self.simulationParameters)
                        cell.LFP = electrode.LFP
                else:
                    cell.simulate(electrode, **self.simulationParameters)
                    cell.LFP = electrode.LFP
        
        
            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z
            cell.electrodecoeff = electrode.electrodecoeff

            #access file object
            f = h5py.File(os.path.join(self.savefolder,
                                self.default_h5_file) % (cellindex),
                          mode='r+')
            
            if self.simulationParameters.has_key('to_file'):
                if self.simulationParameters['to_file']:
                    f['LFP'] = f['electrode000']
                    del f['electrode000']
                    try:
                        assert('LFP' in f.keys())
                    except AssertionError as ae:
                        raise ae('LFP dataset not found in {}'.format(f))

            #save stuff from savelist
            for attrbt in self.savelist:
                try:
                    del(f[attrbt])
                except:
                    pass
                try:
                    f[attrbt] = getattr(cell, attrbt)
                except:
                    try:
                        f[attrbt] = str(getattr(cell, attrbt))
                    except AttributeError:
                        import warnings
                        warnings.warn('Could not find %s in cell' % attrbt)

            #print some stuff
            print 'SIZE %i, RANK %i, Cell %i, Min LFP: %.3f, Max LFP: %.3f' % \
                        (SIZE, RANK, cellindex,
                        f['LFP'].value.min() if 'LFP' in f.keys() else f['electrode000'].value.min(),
                        f['LFP'].value.max() if 'LFP' in f.keys() else f['electrode000'].value.max())

            f.close()
            
            print 'Cell %s saved to file' % cellindex
    
    
    def insert_synapses(self, cell, synparams, idx,
                        SpCell, SpTimes):
        '''
        insert synapse with parameters=synparams on cell=cell, with
        segment indexes given by idx. SpCell and SpTimes picked from Brunel
        network simulation
        
        
        Keyword arguments:
        ::
            
            cell : LFPy.TemplateCell instance
                cell-object synapses are attached to
            synparams : dict
                parameters for LFPy.Synapse objects
            ids : int
                compartment index of postsynaptic target site
            SpCell : np.ndarray
                indices of network neurons spike trains are selected from
            SpTimes : str
                path to database file with spikes
        
        '''
        print 'inserting %i synsapses, TSA : %.3f ' % \
                (idx.size, cell.area[cell.get_idx(synparams['section'])].sum())
        
        #NEURON complain about this one
        del synparams['nPerArea'], synparams['section']
        
        #Insert synapses in an iterative fashion
        db = GDF(SpTimes, new_db=False)
        spikes = db.select(SpCell[:idx.size])
        db.close()


        #combine spike trains that end up on same compartment
        idx_hist, idx_bins  = np.histogram(idx, np.arange(cell.totnsegs+1))
        
        
        i = 0
        for j in range(cell.totnsegs):
            if idx_hist[j] > 0:
                synspikes = []
                for k in range(idx_hist[j]):
                    synspikes = np.r_[synspikes, spikes[i]]
                    i += 1
                synparams.update({'idx' : j})
                # Create synapse(s) and setting times using class LFPy.Synapse
                synapse = LFPy.Synapse(cell, **synparams)
                synspikes.sort()
                synapse.set_spike_times(synspikes + cell.tstart)
        
        
    def point_axon_down(self, cell):
        '''
        make the axon point downwards, start at soma mid point,
        comp for soma diameter
        
        Keyword arguments:
        ::
            
            cell : LFPy.TemplateCell instance
        
        '''
        iaxon = cell.get_idx(section='axon')
        isoma = cell.get_idx(section='soma')
        cell.xstart[iaxon] = cell.xmid[isoma]
        cell.xmid[iaxon] = cell.xmid[isoma]
        cell.xend[iaxon] = cell.xmid[isoma]
        
        cell.ystart[iaxon] = cell.ymid[isoma]
        cell.ymid[iaxon] = cell.ymid[isoma]
        cell.yend[iaxon] = cell.ymid[isoma]
        
        j = 0
        for i in iaxon:
            cell.zstart[i] = cell.zmid[isoma] \
                    - cell.diam[isoma]/2 - cell.length[i] * j
            cell.zmid[i] = cell.zmid[isoma] \
                    - cell.diam[isoma]/2 - cell.length[i]/2  - cell.length[i]*j
            cell.zend[i] = cell.zmid[isoma] \
                    - cell.diam[isoma]/2 - cell.length[i] - cell.length[i]*j
            j += 1
        
        ##point the pt3d axon as well
        for sec in cell.allseclist:
            if sec.name().rfind('axon') >= 0:
                x0 = cell.xstart[cell.get_idx(sec.name())[0]]
                y0 = cell.ystart[cell.get_idx(sec.name())[0]]
                z0 = cell.zstart[cell.get_idx(sec.name())[0]]
                L = sec.L
                for j in range(int(neuron.h.n3d())):
                    neuron.h.pt3dchange(j, x0, y0, z0,
                                     sec.diam)
                    z0 -= L / (neuron.h.n3d()-1)
        
        
    def fetchSynIdx(self):
        '''
        Find possible synaptic placements for each cell.
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            tuple of lists
                compartment indices of synapses for each cell in population.
                First entry is for excitatory connections, second inhibitory
                connections
        
        '''
        if RANK == 0:
            syn_idx_ex = []
            syn_idx_in = []
            
            cellParameters = self.cellParameters.copy()
            try:
                if cellParameters['pt3d']:
                    cellParameters.update({'pt3d' : False})
            except:
                cellParameters.update({'pt3d' : False})
            
               
            print 'find synaptic placements.', 
            for cellindex in range(self.POPULATION_SIZE):
                print '.',
                cellParameters = dict(self.cellParameters)
                cellParameters.update(dict(
                    morphology = self.shuffled_morphologies[cellindex],
                    templatefile = self.shuffled_templatefiles[cellindex],
                    templateargs = self.shuffled_morphologies[cellindex]
                ))
                
                
                cell = LFPy.TemplateCell(**cellParameters)
        
                section = self.synapseParametersEx['section']
                nPerArea = self.synapseParametersEx['nPerArea']
                #nPerArea may be specified as (mean, std) tuple
                if not np.isscalar(nPerArea):
                    nPerArea = np.random.normal(nPerArea[0], nPerArea[1])
                n = int(nPerArea*cell.area[cell.get_idx(section=section)].sum())
                syn_idx_ex.append(cell.get_rand_idx_area_norm(section=section,
                                                              nidx=n))
                
                section = self.synapseParametersIn['section']
                nPerArea = self.synapseParametersIn['nPerArea']
                #nPerArea may be specified as (mean, std) tuple
                if not np.isscalar(nPerArea):
                    nPerArea = np.random.normal(nPerArea[0], nPerArea[1])
                n = int(nPerArea*cell.area[cell.get_idx(section=section)].sum())
                syn_idx_in.append(cell.get_rand_idx_area_norm(section=section,
                                                              nidx=n))
                
                
            print '.'
        else:
            syn_idx_ex = None
            syn_idx_in = None
        
        syn_idx_ex = COMM.bcast(syn_idx_ex, root=0)
        syn_idx_in = COMM.bcast(syn_idx_in, root=0)
        
        return syn_idx_ex, syn_idx_in

    
    def fetchSpCells(self):
        '''
        For N excitatory and inhib networkInstance-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        these two populations and broadcast these as SpCellEx
        and SpCellIn.
        
        This method takes no keyword arguments.
        
        Returns:
        ::
            
            tuple of lists
                presynaptic neuron identifiers in network
        
        '''
        if RANK == 0:
            SpCellEx = []
            SpCellIn = []
            for i in range(self.POPULATION_SIZE):
                SpCellEx.append(np.random.randint(self.nodes_ex.min(),
                                                  self.nodes_ex.max(),
                                                  size=self.synIdxEx[i].size))
                SpCellIn.append(np.random.randint(self.nodes_in.min(),
                                                  self.nodes_in.max(),
                                                  size=self.synIdxIn[i].size))
        else:
            SpCellEx = None
            SpCellIn = None
        
        #broadcast the cell indexes to all ranks
        SpCellEx = COMM.bcast(SpCellEx, root=0)
        SpCellIn = COMM.bcast(SpCellIn, root=0)
    
        return SpCellEx, SpCellIn


class BenchmarkDataRing(BenchmarkDataLayer):
    '''
    class BenchmarkDataRing.
    same as BenchmarkDataLayer, just with it's own fetchSpCells method,
    where presynaptic cells are from a specific region on the
    network    
    '''
    def __init__(self,
                randdist=np.random.vonmises,
                sigma_EX = np.pi,
                sigma_IN = np.pi,
                 **kwargs):
        '''
        class BenchmarkDataRing initialization
        
        Similar to BenchmarkDataLayer, just with it's own fetchSpCells method,
        where presynaptic cells are from a specific region on the
        network, chosen on random using either a von Mises type probability
        distribution, or boxcar.
        
        Note that some spike trains from the networksim-instance may be chosen
        repeatedly if the size of the network is small.
        
        arguments:
        ::
            
            randdist :  function
                np.random.vonmises for a gaussian distribution along
                ring network topology or np.random.rand for a boxcar
                type sampling of spike trains
            sigma_EX :  float
                the spread of the distribution, excitatory cells
            sigma_IN : float
                the spread of the distribution, inhibitory cells
            **kwargs :  keyword args passed to parent class BenchmarkDataLayer
        '''
        self.randdist = randdist
        #centers of distributions on [-pi, pi>:
        self.loc_EX = np.arange(-np.pi, np.pi,
                    np.pi * 2 / kwargs['populationParameters'
                                       ]['POPULATION_SIZE'])
        self.loc_IN = np.arange(-np.pi, np.pi,
                    np.pi * 2 / kwargs['populationParameters'
                                       ]['POPULATION_SIZE'])
        self.sigma_EX = sigma_EX
        self.sigma_IN = sigma_IN

        #initialize the class
        BenchmarkDataLayer.__init__(self, **kwargs)

        
    def fetchSpCells(self):
        '''
        For N excitatory and inhib networkInstance-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        these two populations and broadcast these as SpCellEx
        and SpCellIn
        
        This is meant to be used with testdata.RingNetwork, and
        will pick cells located on a region of the 1D-network
        with random Von Mises distributed numbers scaled to network nodes
        '''
        if RANK == 0:
            SpCellEx = []
            SpCellIn = []
            
            for i in range(self.POPULATION_SIZE):
                if self.randdist == np.random.vonmises:
                    #pick vonmises distributed numbers on [-pi, pi] interval
                    distEx = self.randdist(self.loc_EX[i], self.sigma_EX,
                                                size = self.synIdxEx[i].size)
                elif self.randdist == np.random.rand:
                    distEx = self.randdist(self.synIdxEx[i].size)
                    #normalize to network population indices
                    distEx -= 0.5
                    distEx *= np.pi*2
                    distEx /= self.sigma_EX
                    distEx += self.loc_EX[i]
                    distEx[distEx < -np.pi] += np.pi*2
                else:
                    raise ValueError, 'randdist not vonmises() or rand()'
                #normalize to network population indices
                distEx /= 2*np.pi
                distEx += 0.5
                distEx *= self.nodes_ex.size
                distEx += self.nodes_ex[0]

                SpCellEx.append(distEx.astype(int))
                
                if self.randdist == np.random.vonmises:
                    #pick vonmises distributed numbers on [-pi, pi] interval
                    distIn = self.randdist(self.loc_IN[i], self.sigma_IN,
                                                size = self.synIdxIn[i].size)
                elif self.randdist == np.random.rand:
                    distIn = self.randdist(self.synIdxIn[i].size)
                    #normalize to network population indices
                    distIn -= 0.5
                    distIn *= np.pi*2
                    distIn /= self.sigma_IN
                    distIn += self.loc_IN[i]
                    distIn[distIn < -np.pi] += np.pi*2
                else:
                    raise ValueError, 'randdist not vonmises() or rand()'
                #normalize to network population indices
                distIn /= 2*np.pi
                distIn += 0.5
                distIn *= self.nodes_in.size
                distIn += self.nodes_in[0]
                
                SpCellIn.append(distIn.astype(int))
            
        else:
            SpCellEx = None
            SpCellIn = None
        
        #broadcast the cell indexes to all ranks
        SpCellEx = COMM.bcast(SpCellEx, root=0)
        SpCellIn = COMM.bcast(SpCellIn, root=0)
    
        return SpCellEx, SpCellIn
    
