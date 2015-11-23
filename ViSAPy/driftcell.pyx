#!/usr/bin/env python
'''
subclass of LFPy.TemplateCell incorporating electrode drift
'''

import numpy as np
cimport numpy as np
import neuron
from time import time

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t   LTYPE_t
from LFPy import TemplateCell


def _run_simulation(cell, variable_dt=False, atol=0.001):
    '''
    Running the actual simulation in NEURON, simulations in NEURON
    is now interruptable.
    '''
    neuron.h.dt = cell.timeres_NEURON
    
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
    else:
        cvode.active(0)
    
    #initialize state
    neuron.h.finitialize(cell.v_init)
    
    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()
    
    #Starting simulation at t != 0
    neuron.h.t = cell.tstartms
    
    cell._loadspikes()
        
    #print sim.time at intervals
    cdef double counter = 0.
    cdef double interval
    cdef double tstopms = cell.tstopms
    cdef double t0 = time()
    cdef double ti = neuron.h.t
    cdef double rtfactor
    if tstopms > 1000:
        interval = 1 / cell.timeres_NEURON * 100
    else:
        interval = 1 / cell.timeres_NEURON * 10
    
    while neuron.h.t < tstopms:
        neuron.h.fadvance()
        counter += 1.
        if divmod(counter, interval)[1] == 0:
            rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
            print('t = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor))
            t0 = time()
            ti = neuron.h.t


def _run_simulation_with_electrode(cell,
                                   variable_dt=False, atol=0.001,
                                   to_memory=True, to_file=False,
                                   file_name=None,
                                   np.ndarray[DTYPE_t, ndim=3, negative_indices=False] dotprodcoeffs=None,
                                   int driftint=0):
    '''
    Running the actual simulation in NEURON.
    electrode argument used to determine coefficient
    matrix, and calculate the LFP on every time step.
    '''
    
    #c-declare some variables
    cdef int i, j, tstep, ncoeffs
    cdef int totnsegs = cell.totnsegs
    cdef double tstopms = cell.tstopms
    cdef double counter, interval
    cdef double t0
    cdef double ti
    cdef double rtfactor
    cdef double timeres_NEURON = cell.timeres_NEURON
    cdef double timeres_python = cell.timeres_python
    #cdef np.ndarray[DTYPE_t, ndim=3, negative_indices=False] dotprodcoeffs
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False] coeffs, LFP
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] imem = \
        np.empty(totnsegs)
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] area = \
        cell.area
    
    #check if h5py exist and saving is possible
    try:
        import h5py
    except:
        print('h5py not found, LFP to file not possible')
        to_file = False
        file_name = None



    # Initialize NEURON simulations of cell object    
    neuron.h.dt = timeres_NEURON
    
    #integrator
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
    else:
        cvode.active(0)
    
    #initialize state
    neuron.h.finitialize(cell.v_init)
    
    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()
    
    #Starting simulation at t != 0
    neuron.h.t = cell.tstartms
    
    #load spike times from NetCon
    cell._loadspikes()
    
    #print sim.time at intervals
    counter = 0.
    tstep = 0
    t0 = time()
    ti = neuron.h.t
    if tstopms > 1000:
        interval = 1 / timeres_NEURON * 100
    else:
        interval = 1 / timeres_NEURON * 10
        
    #temp vector to store membrane currents at each timestep
    imem = np.empty(cell.totnsegs)
    #LFPs for each electrode will be put here during simulation
    if to_memory:
        LFP = np.empty((dotprodcoeffs.shape[1],
                                    cell.tstopms / cell.timeres_NEURON + 1))
    
    #LFPs for each electrode will be put here during simulations
    if to_file:
        #ensure right ending:
        if file_name.split('.')[-1] != 'h5':
            file_name += '.h5'
        el_LFP_file = h5py.File(file_name, 'w')
        el_LFP_file['electrode%.3i' % 0] = np.empty((dotprodcoeffs.shape[1],
                                    cell.tstopms / cell.timeres_NEURON + 1))

    #multiply segment areas with specific membrane currents later:
    #mum2 conversion factor:
    area *= 1E-2
    coeffs = dotprodcoeffs[0] #not used yet
    while neuron.h.t < tstopms:
        if neuron.h.t >= 0:
            i = 0
            for sec in cell.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane
                    i += 1
            #pA/mum2 -> nA conversion
            imem *= area
            
            if divmod(tstep, driftint)[1] == 0:
                coeffs = dotprodcoeffs[divmod(tstep, driftint)[0],]
            else:
                pass
            if to_memory:
                LFP[:, tstep] = np.dot(coeffs, imem)
                    
            if to_file:
                el_LFP_file['electrode%.3i' % 0][:, tstep] = np.dot(coeffs, imem)
            
            tstep += 1
        neuron.h.fadvance()
        counter += 1.
        if divmod(counter, interval)[1] == 0:
            rtfactor = (neuron.h.t - ti) * 1E-3 / (time() - t0)
            print('t = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor))
            t0 = time()
            ti = neuron.h.t
    
    try:
        #calculate LFP after final fadvance()
        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                imem[i] = seg.i_membrane
                i += 1
        #pA/mum2 -> nA conversion
        imem *= area

        if divmod(tstep, driftint)[1] == 0:
            coeffs = dotprodcoeffs[divmod(tstep, driftint)[0], :, :]
        else:
            pass    
        if to_memory:
            LFP[:, tstep] = np.dot(coeffs, imem)
        if to_file:
            el_LFP_file['electrode%.3i' % 0][:, tstep] = np.dot(coeffs, imem)

    except:
        pass
    
    # Final step, put LFPs in the electrode object, superimpose if necessary
    # If electrode.perCellLFP, store individual LFPs
    if to_memory:
        cell.dotprodresults = LFP
    
    if to_file:
        el_LFP_file.close()




class DriftCell(TemplateCell):
    def __init__(self, **kwargs):
        '''
        Initialization of class DriftCell. Inherits class LFPy.TemplateCell.
        
        This class implements a modified simulate function allowing shifting
        the electrode recording position during the course of simulations.
        
        Keyword arguments:
        ::
            
            **kwargs : see class LFPy.Cell and LFPy.TemplateCell
        
        '''        
        #initialize the cell object
        TemplateCell.__init__(self, **kwargs)



    def simulate(self, electrode=None, rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                 rec_variables=[], variable_dt=False, atol=0.001,
                 to_memory=True, to_file=False, file_name=None,
                 dotprodcoeffs=None, driftInterval=None):
        '''
        This is the main function running the simulation of the NEURON model.
        Start NEURON simulation and record variables specified by arguments.
        
        Arguments:
        ::
            
            electrode:  Either an LFPy.RecExtElectrode object or a list of such.
                        If supplied, LFPs will be calculated at every time step
                        and accessible as electrode.LFP. If a list of objects
                        is given, accessible as electrode[0].LFP etc.
            rec_imem:   If true, segment membrane currents will be recorded
                        If no electrode argument is given, it is necessary to
                        set rec_imem=True in order to calculate LFP later on.
                        Units of (nA).
            rec_vmem:   record segment membrane voltages (mV)
            rec_ipas:   record passive segment membrane currents (nA)
            rec_icap:   record capacitive segment membrane currents (nA)
            rec_isyn:   record synaptic currents of from Synapse class (nA)
            rec_vmemsyn:    record membrane voltage of segments with Synapse(mV)
            rec_istim:  record currents of StimIntraElectrode (nA)
            rec_variables: list of variables to record, i.e arg=['cai', ]
            variable_dt: boolean, using variable timestep in NEURON
            atol:       absolute tolerance used with NEURON variable timestep 
            to_memory:  only valid with electrode, store lfp in -> electrode.LFP 
            to_file:    only valid with electrode, save LFPs in hdf5 file format 
            file_name:  name of hdf5 file, '.h5' is appended if it doesnt exist
            dotprodcoeffs :  list of N x Nseg np.ndarray. These arrays will at
                        every timestep be multiplied by the membrane currents.
                        Presumably useful for memory efficient csd or lfp calcs
            driftinterval : float
                time in ms; if dotprodcoeffs is a 3D array, switch dotprodcoeffs
                at these fixed time intervals. 
            '''
        self._set_soma_volt_recorder()
        self._collect_tvec()
        
        if rec_imem:
            self._set_imem_recorders()
        if rec_vmem:
            self._set_voltage_recorders()
        if rec_ipas:
            self._set_ipas_recorders()
        if rec_icap:
            self._set_icap_recorders()
        if len(rec_variables) > 0:
            self._set_variable_recorders(rec_variables)
        
        if driftInterval < self.timeres_python:
            raise ValueError, 'driftinterval < timeres_python!'
        if driftInterval != None:
            #convert driftinterval to tstep
            driftint = int(driftInterval / self.timeres_python)
        else:
            driftint = None
        
        #run fadvance until t >= tstopms, and calculate LFP if asked for
        if dotprodcoeffs is None:
            if not rec_imem:
                print("rec_imem = %s, membrane currents will not be recorded!" \
                                  % str(rec_imem))
            _run_simulation(self, variable_dt, atol)
        else:
            #allow using both electrode and additional coefficients:
            _run_simulation_with_electrode(self, variable_dt, atol,
                                               to_memory, to_file, file_name,
                                               dotprodcoeffs, driftint)
        #somatic trace
        self.somav = np.array(self.somav)
        
        if rec_imem:
            self._calc_imem()        
        if rec_ipas:
            self._calc_ipas()        
        if rec_icap:
            self._calc_icap()        
        if rec_vmem:
            self._collect_vmem()        
        if rec_isyn:
            self._collect_isyn()        
        if rec_vmemsyn:
            self._collect_vsyn()        
        if rec_istim:
            self._collect_istim()
        if len(rec_variables) > 0:
            self._collect_rec_variables(rec_variables)
