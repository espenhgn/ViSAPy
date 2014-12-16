#!/usr/bin/env python
'''class for running testdata generation with method of images'''

from ViSAPy import BenchmarkData
from ViSAPy.cyextensions import ouProcess
from scipy.optimize import curve_fit
from MoI import MoI
import numpy as np
import h5py
import os
import LFPy
import neuron
from mpi4py import MPI


################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
MASTER_MODE = COMM.rank == 0


def find_distance_dependence(data, dist):
    """ Find normalized decay constant"""
    def func(dist, a, b):
        return np.exp(-dist*a) + b         
    popt, pcov = curve_fit(func, dist, data, p0=[1./30, 0.1])
    return popt, pcov 


def make_corr_matrix_neuron(populationParameters, exp_input_name, ext_sim_dict):
    """ Makes the correlation matrix specifying the amount of correlation
    between the input to the cells. It is distance (in electrode plane) dependent,
    following a relationship extracted from experimental recordings
    in a very crude ad hoc manner.
    """
    n_neurons = populationParameters['POPULATION_SIZE']
    # Since dictionaries are unstructered, we need to remember the order
    # the neurons appear in.
    corr_matrix = np.zeros((n_neurons, n_neurons))


    exp_data = np.load(exp_input_name)
    elec_x = ext_sim_dict['elec_x']
    elec_y = ext_sim_dict['elec_y']
    dist_matrix = np.zeros((len(elec_x), len(elec_x)))
    exp_corr = np.corrcoef(exp_data)
    for elec_1 in xrange(len(elec_x)):
        for elec_2 in xrange(len(elec_x)):                
            dist = np.sqrt((elec_x[elec_1] - elec_x[elec_2])**2 +
                            (elec_y[elec_1] - elec_y[elec_2])**2)
            dist_matrix[elec_1, elec_2] = dist

    popt, pcov = find_distance_dependence(exp_corr.flatten(), dist_matrix.flatten())

    for cell_id_1 in range(n_neurons):
        for cell_id_2 in range(n_neurons):
            if cell_id_1 == cell_id_2:
                dist = 0
                corr_matrix[cell_id_1, cell_id_2] = 1
            else:    
                pos_x_1 = populationParameters['X'][cell_id_1]
                pos_y_1 = populationParameters['Y'][cell_id_1]
                pos_x_2 = populationParameters['X'][cell_id_2]
                pos_y_2 = populationParameters['Y'][cell_id_2]
                dist = np.sqrt((pos_x_1 - pos_x_2)**2 + (pos_y_1 - pos_y_2)**2)
                #Ad hoc from experimental data (old): 8.25/(dist*1.5 + 8.43) + 0.02 
                corr_matrix[cell_id_1, cell_id_2] =  np.exp(-dist*popt[0]) + popt[1]

    return corr_matrix
    
    
class BenchmarkDataMoI(BenchmarkData):
    def __init__(self,
                 custom_codes = [],
                 paramsMapping = [],
                 paramsMoI = [],
                 gsynParams = {},
                 **kwargs):
        '''
        Initialization of class for testdata emulating MEA electrode setup
        '''
        BenchmarkData.__init__(self,
                          **kwargs)

        #set some attributes
        self.custom_codes = custom_codes
        self.gsynParams = gsynParams
        self.paramsMapping = paramsMapping
        self.paramsMoI = paramsMoI

        shuffled_models = self.shuffle_model_sets()
        self.shuffled_morphologies = shuffled_models[0]
        self.shuffled_custom_codes = shuffled_models[1]

        #create synapse conductance time envelopes
        self.gsyn = self.create_gsyn()
        
        #compute electrode cell mappings
        self.electrodecoeffs = self.compute_electrodecoeffs()
        
        

    def set_pop_soma_pos(self):
        '''set soma positions'''
        pos = {}
        for cellindex in range(self.POPULATION_SIZE):
            pos.update({cellindex : dict(
                xpos = self.populationParameters['X'][cellindex],
                ypos = self.populationParameters['Y'][cellindex],
                zpos = self.populationParameters['Z'][cellindex])})
        return pos
    
    
    def set_shuffled_morphologies(self):
        '''Monkeypatching of parent class method, returning None'''
        return

    
    def create_gsyn_correlations(self, lambda_d=10.):
        '''create correlation matrix of synapse conductance, having an
        exponential function dependence
        
        kwargs:
        ::
            lambda_d : float, exponential decay constant in um
        '''
        def r(i):
            '''compute distance'''
            x = self.populationParameters['X']
            y = self.populationParameters['Y']
            z = self.populationParameters['Z']
            return np.sqrt((x-x[i])**2 + (y-y[i])**2 + (z-z[i])**2)
        
        C = np.zeros((self.POPULATION_SIZE, self.POPULATION_SIZE))
        
        
        for i in range(self.POPULATION_SIZE):
            C[i, ] = np.exp(-r(i) / lambda_d)

        return C
    

    def create_gsyn(self):
        '''
        create correlated input conductance traces with
        cell to cell distance dependece
        
        The returned argument (ncells x ntimesteps) array with conductances
        '''
        if MASTER_MODE:
            gsynfile = os.path.join(self.savefolder, 'gsyn.h5')
            if os.path.isfile(gsynfile):
                print 'loading synapse conductances from file %s' % gsynfile
                f = h5py.File(gsynfile, 'r')
                gsyncorr = f['data'].value
                f.close()                
            else:
                corrmx_gsyn = self.create_gsyn_correlations(lambda_d=self.gsynParams['lambda_d'])
                chol = np.linalg.cholesky(corrmx_gsyn)
                
                #get uncorrelated noise
                gsyn = ouProcess(**self.gsynParams['OUParams'])
                
                #make the noise correlated
                gsyncorr = np.dot(chol, gsyn).T
                
                #set the desired means and standard deviations in each trace
                gsyncorr /= gsyncorr.std(axis=0)
                gsyncorr -= gsyncorr.mean(axis=0)
                gsyncorr *= self.gsynParams['gsyn_std']
                gsyncorr += self.gsynParams['gsyn_mean']
                
                gsyncorr = gsyncorr.T
                
                #saving to HDF5
                print 'writing synapse conductances to file %s' % gsynfile
                f = h5py.File(gsynfile, compression='gzip')
                f['data'] = gsyncorr
                f['Fs'] = self.cellParameters['timeres_python']
                f.close()
                
        else:
            gsyncorr = None

        COMM.barrier()
        return COMM.bcast(gsyncorr, root=0)


    def shuffle_model_sets(self):
        '''
        Take lists of morphologies and custom_codes, and return for each cell
        a pair of morphology and corresponding custom_code LFPy.Cell keyword
        arguments        
        '''
        if MASTER_MODE:
            multpl = self.POPULATION_SIZE / len(self.morphologies) + 1
            
            #repeating list for as many times as necessary
            pairs = zip(self.morphologies*multpl, self.custom_codes*multpl)
            
            shuffled_morphos = []
            shuffled_codes = []
            for i in range(self.POPULATION_SIZE):
                j = np.random.randint(self.POPULATION_SIZE)           
                shuffled_morphos += [pairs[j][0]]
                shuffled_codes += [pairs[j][1]]
                shuffled = [shuffled_morphos, shuffled_codes]
        else:
            shuffled = None
        
        COMM.barrier()
        return COMM.bcast(shuffled, root=0)

    
    def compute_electrodecoeffs(self):
        '''
        compute for each cell their corresponding mapping of compartment
        membrane currents to each electrode contact point
        '''
        coefffile = os.path.join(self.savefolder, 'electrodecoeffs.h5')
        
        moi = MoI(self.paramsMoI)

        coeffs = []

        if os.path.isfile(coefffile):
            if MASTER_MODE:
                print 'loading electrode mappings from file %s' % coefffile                
                f = h5py.File(coefffile, 'r')
                allcoeffs = {}
                for cellindex in range(self.POPULATION_SIZE):
                    allcoeffs.update({
                        cellindex : f['cell%.3i' % cellindex].value
                        })
                f.close()
            else:
                allcoeffs = None
            return COMM.bcast(allcoeffs, root=0)
        else:       
            for cellindex in range(self.POPULATION_SIZE):
                if divmod(cellindex, SIZE)[1] == RANK:
                    cell = self.cellsim(cellindex, return_just_cell=True)
    
                    mapping = moi.make_mapping_cython(self.paramsMapping,
                                                    xmid=cell.xmid,
                                                    ymid=cell.ymid,
                                                    zmid=cell.zmid,
                                                    xstart=cell.xstart,
                                                    ystart=cell.ystart,
                                                    zstart=cell.zstart,
                                                    xend=cell.xend,
                                                    yend=cell.yend,
                                                    zend=cell.zend,
                                                    morphology=cell.morphology)
                    coeffs.append({cellindex : mapping})
            
            #sync MPI threads, communicate coefficients between ranks
            COMM.Barrier()
            coeffs = COMM.gather(coeffs, root=0)
            if RANK == 0:
                allcoeffs = {}
                for i in range(SIZE):
                    for item in coeffs[i]:
                        allcoeffs.update(item)
                
                #write mappings to HDF5
                print 'writing electrode mappings to file %s' % coefffile
                f = h5py.File(coefffile, 'a', compression='gzip')
                for cellindex in range(self.POPULATION_SIZE):
                    f['cell%.3i' % cellindex] = allcoeffs[cellindex]
                f.close()            
            else:
                allcoeffs = None
            
            return COMM.bcast(allcoeffs, root=0)
    
    
    def cellsim(self, cellindex, return_just_cell=False):
        '''
        main cell simulation and LFP generating procedure
        '''
        
        cellParameters = self.cellParameters.copy()
        cellParameters.update(dict(morphology = self.shuffled_morphologies[cellindex],
                               custom_code = [self.shuffled_custom_codes[cellindex]])
        )
        
        cell = LFPy.Cell(**cellParameters)
        
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])    

        if return_just_cell:
            #with several cells, NEURON can only hold one cell at the time
            allsecnames = []
            allsec = []
            for sec in cell.allseclist:
                allsecnames.append(sec.name())
                for i in xrange(sec.nseg):
                    allsec.append(sec.name())
            cell.allsecnames = allsecnames
            cell.allsec = allsec
            return cell
        else:
            #set up synapse
            t = np.arange(self.gsyn[cellindex].size).astype(float)
            t *= self.cellParameters['timeres_python']
            t += self.cellParameters['tstartms']
            gsyn_t = neuron.h.Vector(t)
            
            #synapse conductance
            gsyn = neuron.h.Vector(self.gsyn[cellindex])
            
            #insert mech and play vector
            for sec in neuron.h.soma:
                sec.insert('gsyn')
                for seg in sec:
                    gsyn.play(seg._ref_g_gsyn, gsyn_t)

            #perform simulation
            if self.simulationParameters.has_key('to_file'):
                if self.simulationParameters['to_file']:
                    cell.simulate(dotprodcoeffs=self.electrodecoeffs[cellindex],
                                  file_name=os.path.join(self.savefolder,
                                        self.default_h5_file) % (cellindex),
                                  **self.simulationParameters)
                else:
                    cell.simulate(dotprodcoeffs=self.electrodecoeffs[cellindex],
                                  **self.simulationParameters)
                    cell.LFP = cell.dotprodresults[0]
            else:
                cell.simulate(dotprodcoeffs=self.electrodecoeffs[cellindex],
                              **self.simulationParameters)
                cell.LFP = cell.dotprodresults[0]
        
        
            
            cell.x = self.paramsMapping['elec_x']
            cell.y = self.paramsMapping['elec_y']
            cell.z = self.paramsMapping['elec_z']
            
            cell.custom_code = self.shuffled_custom_codes[cellindex][1]
            cell.electrodecoeff = self.electrodecoeffs[cellindex]

            #access file object
            f = h5py.File(os.path.join(self.savefolder,
                                self.default_h5_file) % (cellindex),
                          compression='gzip')
            
            if self.simulationParameters.has_key('to_file'):
                if self.simulationParameters['to_file']:
                    f['LFP'] = f['electrode000'].astype('float32')

            #save stuff from savelist
            for attrbt in self.savelist:
                try:
                    del(f[attrbt])
                except:
                    pass
                try:
                    if attrbt == 'LFP':
                        f[attrbt] = getattr(cell, attrbt).astype('float32')
                    else:
                        f[attrbt] = getattr(cell, attrbt)
                except:
                    try:
                        f[attrbt] = str(getattr(cell, attrbt))
                    except:
                        import warning
                        warning.warn('Could not find %s in cell') % attrbt

            #print some stuff
            print 'SIZE %i, RANK %i, Cell %i, Min LFP: %.3f, Max LFP: %.3f' % \
                        (SIZE, RANK, cellindex,
                        f['LFP'].value.min(), f['LFP'].value.max())

            f.close()
            
            print 'Cell %s saved to file' % cellindex            




