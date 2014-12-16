#!/usr/bin/env python
import numpy as np
from sys import stdout
import os
from os.path import join
from cython_funcs import *

class MoI:
    '''Class for calculating the potential in a semi-infinite slice of neural tissue.
    Set-up:


              SALINE -> sigma_S 

    <----------------------------------------------------> z = + a
    
              TISSUE -> sigma_T


                   o -> charge_pos = [x,y,z]


    <-----------*----------------------------------------> z = -a               
                 \-> elec_pos = [x,y,z] 

                 ELECTRODE GLASS PLATE -> sigma_G 
        

    Arguments:
        set_up_parameters = {
                     'sigma_G': 0.0, # Conductivity below electrode
                     'sigma_T': 0.3, # Conductivity of tissue
                     'sigma_S': 3.0, # Conductivity of saline
                     'slice_thickness': 200., # um
                     'steps' : 20, # How many steps to include of the infinite series
                      }

    The cython functions can be used independently of this class.
                 
    '''
    def __init__(self,
                 set_up_parameters = {
                     'sigma_G': 0.0, # Below electrode
                     'sigma_T': 0.3, # Tissue
                     'sigma_S': 3.0, # Saline
                     'slice_thickness': 300., # um
                     'steps' : 20,
                      },
                 debug = False
                 ):
        self.set_up_parameters = set_up_parameters
        try:
            self.sigma_G = set_up_parameters['sigma_G']
        except KeyError:
            self.sigma_G = 0
        self.sigma_T = set_up_parameters['sigma_T']
        self.sigma_S = set_up_parameters['sigma_S']
        self._check_for_anisotropy()
        
        self.slice_thickness = set_up_parameters['slice_thickness']
        self.a = self.slice_thickness/2.
        try:
            self.steps = set_up_parameters['steps']
        except KeyError:
            self.steps = 50

    def _anisotropic_saline_scaling(self):
        """ To make formula work in anisotropic case we scale the conductivity of the
        saline to be a scalar k times the tissue conductivity. (Wait 1990)
        """

        ratios = np.array(self.sigma_S) / np.array(self.sigma_T)

        if np.abs(ratios[0] - ratios[2]) <= 1e-15:
            scale_factor = ratios[0]
        elif np.abs(ratios[1] - ratios[2]) <= 1e-15:
            scale_factor = ratios[1]

        sigma_S_scaled = scale_factor * np.array(self.sigma_T)
        sigma_T_net = np.sqrt(self.sigma_T[0] * self.sigma_T[2])
        sigma_S_net = np.sqrt(sigma_S_scaled[0] * sigma_S_scaled[2])

        print "Sigma_T: %s, Sigma_S: %s, Sigma_S_scaled: %s, scale factor: %g" %(
            self.sigma_T, self.sigma_S, sigma_S_scaled, scale_factor)
        self.anis_W = (sigma_T_net - sigma_S_net)/(sigma_T_net + sigma_S_net)

        

    def _check_for_anisotropy(self):
        """ Checks if input conductivities are tensors or scalars
        and sets self.is_anisotropic correspondingly
        """
        sigmas = [self.sigma_G, self.sigma_T, self.sigma_S]
        anisotropy_list = []

        types = [type(self.sigma_T), type(self.sigma_S), type(self.sigma_S)]

        
        if (list in types) or (np.ndarray in types):
            self.is_anisotropic = True

            if type(self.sigma_G) in [list, np.ndarray]:
                if len(self.sigma_G) != 3:
                    raise ValueError, "Conductivity vector but not with size 3"
                self.sigma_G = np.array(self.sigma_G) #Just to be sure it's numpy array
            else:
                self.sigma_G = np.array([self.sigma_G, self.sigma_G, self.sigma_G])
            if type(self.sigma_T) in [list, np.ndarray]:
                if len(self.sigma_T) != 3:
                    raise ValueError, "Conductivity vector but not with size 3"
                self.sigma_T = np.array(self.sigma_T) #Just to be sure it's numpy array
            else:
                self.sigma_T = np.array([self.sigma_T, self.sigma_T, self.sigma_T])

            if type(self.sigma_S) in [list, np.ndarray]:
                if len(self.sigma_S) != 3:
                    raise ValueError, "Conductivity vector but not with size 3"
                self.sigma_S = np.array(self.sigma_S) #Just to be sure it's numpy array
            else:
                self.sigma_S = np.array([self.sigma_S, self.sigma_S, self.sigma_S])
       
            self._anisotropic_saline_scaling()
            if (self.sigma_G[0] == self.sigma_G[1] == self.sigma_G[2]) and \
               (self.sigma_T[0] == self.sigma_T[1] == self.sigma_T[2]) and \
               (self.sigma_S[0] == self.sigma_S[1] == self.sigma_S[2]):
                print "Isotropic conductivities can be given as scalars."         
        else:
            self.is_anisotropic = False
 
    def in_domain(self, elec_pos, charge_pos):
        """ Checks if elec_pos and charge_pos is within valid area.
        Otherwise raise exception."""

        # If inputs are single positions
        if (np.array(elec_pos).shape == (3,)) and \
          (np.array(charge_pos).shape == (3,)):
            elec_pos = [elec_pos]
            charge_pos = [charge_pos]

        for epos in elec_pos:
            if not np.abs(epos[2] + self.a) <= 1e-14:
                print "Electrode position: ", elec_pos
                raise RuntimeError("Electrode not within valid range.")
        for cpos in charge_pos:
            if np.abs(cpos[2]) >= self.a:
                print "Charge position: ", charge_pos
                raise RuntimeError("Charge not within valid range.")
        for cpos in charge_pos:
            for epos in elec_pos:
                dist = np.sqrt( np.sum( (np.array(cpos) - np.array(epos))**2 ))
                if dist < 1e-6:
                    print "Charge position: ", charge_pos, "Electrode position: ", elec_pos
                    raise RuntimeError("Charge and electrode at same position!")

    def anisotropic_saline_scaling(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential with saline conductivity
        sigma_S is scaled to k * sigma_T. There is also a much faster cython version of this"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]

        def _omega(dz):
            return 1/np.sqrt(self.sigma_T[0]*self.sigma_T[2]*(y - y0)**2 + \
                             self.sigma_T[0]*self.sigma_T[1]*dz**2 + \
                             self.sigma_T[1]*self.sigma_T[2]*(x - x0)**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += (self.anis_W)**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        phi *= 2*imem/(4*np.pi)
        return phi


    def isotropic_moi(self, charge_pos, elec_pos, imem=1):
        """ This function calculates the potential at the position elec_pos = [x, y, z]
        set up by the charge at position charge_pos = [x0, y0, z0]. To get get the potential
        from multiple charges, the contributions must be summed up.
        """
        def _omega(dz):
            return 1/np.sqrt( (y - y0)**2 + (x - x0)**2 + dz**2) 
        #self.in_domain(elec_pos, charge_pos) # Check if valid positions
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        phi = _omega(z - z0)
        n = 0
        WTS = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        WTG = (self.sigma_T - self.sigma_G)/(self.sigma_T + self.sigma_G)
        while n < self.steps:
            if n == 0:
                phi += WTS * _omega(z + z0 - (4*n + 2)*self.a) +\
                       WTG * _omega(z + z0 + (4*n + 2)*self.a)
            else:
                phi += (WTS*WTG)**n *(\
                    WTS * _omega(z + z0 - (4*n + 2)*self.a) + WTG * _omega(z + z0 + (4*n + 2)*self.a) +\
                    _omega(z - z0 + 4*n*self.a) + _omega(z - z0 - 4*n*self.a) )
            n += 1
        phi *= imem/(4*np.pi*self.sigma_T)
        return phi

    def line_source_moi(self, comp_start, comp_end, comp_length, elec_pos, imem=1):
        """ Calculate the moi line source potential at electrode plane"""
        self.in_domain(elec_pos, comp_start)
        self.in_domain(elec_pos, comp_end)
        x0, y0, z0 = comp_start[:]
        x1, y1, z1 = comp_end[:]
        x, y, z = elec_pos[:]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        a_x = x - x0
        a_y = y - y0
        W = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        def _omega(a_z):
            #See Rottman integration formula 46) page 137 for explanation
            factor_a = comp_length*comp_length
            factor_b = - a_x*dx - a_y*dy - a_z * dz
            factor_c = a_x*a_x + a_y*a_y + a_z*a_z
            b_2_ac = factor_b*factor_b - factor_a * factor_c
            if np.abs(b_2_ac) <= 1e-16:
                num = factor_a + factor_b
                den = factor_b
            else:
                num = factor_a + factor_b + \
                      comp_length*np.sqrt(factor_a + 2*factor_b + factor_c)
                den = factor_b + comp_length*np.sqrt(factor_c)
            return np.log(num/den)
        phi = _omega(-self.a - z0)
        if not phi == phi:
            set_trace()
        n = 1
        while n < self.steps:
            phi += W**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        phi *= 2*imem/(4*np.pi*self.sigma_T * comp_length)
        return phi

    def point_source_moi_at_elec(self, charge_pos, elec_pos, imem=1):
        """ Calculate the moi point source potential assuming electrode at MEA electrode plane (elec_z = -a)"""
        self.in_domain(elec_pos, charge_pos)
        x0, y0, z0 = charge_pos[:]
        x, y, z = elec_pos[:]
        W = (self.sigma_T - self.sigma_S)/(self.sigma_T + self.sigma_S)
        def _omega(dz):
            return 1/np.sqrt( (y - y0)**2 + (x - x0)**2 + dz**2) 
        phi = _omega(-self.a - z0)
        n = 1
        while n < self.steps:
            phi += W**n * (_omega((4*n-1)*self.a - z0) + _omega(-(4*n+1)*self.a - z0))
            n += 1   
        return 2*phi*imem/(4*np.pi*self.sigma_T)
    

    def potential_at_elec_big_average(self, elec_pos, r, n_avrg_points, function, func_args):
        """ Calculate the potential at electrode 'elec_index' with n_avrg_points points"""
        phi = 0
        for pt in xrange(n_avrg_points):
            pt_pos = np.array([(np.random.rand() - 0.5)* 2 * r,
                               (np.random.rand() - 0.5)* 2 * r])
            # If outside electrode
            while np.sum( pt_pos**2) > r**2:
                pt_pos = np.array([(np.random.rand() - 0.5) * 2 * r,
                                   (np.random.rand() - 0.5) * 2 * r])
            avrg_point_pos = [elec_pos[0] + pt_pos[0],
                              elec_pos[1] + pt_pos[1],
                              elec_pos[2]]           
            #phi += self.point_source_moi_at_elec(charge_pos, avrg_point_pos, imem)
            phi += function(*func_args, elec_pos=avrg_point_pos)
        return phi/n_avrg_points

    def make_mapping(self, neur_dict, ext_sim_dict):
        """ Make a mapping given two arrays of electrode positions. To find potential at elec use np.dot(mapping, imem) where imem are
        transmembrane currents"""
        print '\033[1;35mMaking mapping for %s...\033[1;m' %neur_dict["name"]
        coor_folder = join(ext_sim_dict['neural_input'],\
                                         neur_dict['name'])
        if ext_sim_dict['use_line_source']:
            comp_start = np.load(join(coor_folder, 'coor_start.npy'))
            comp_end = np.load(join(coor_folder, 'coor_end.npy'))
            comp_length = np.load(join(coor_folder, 'length.npy'))
        comp_coors = np.load(join(coor_folder, 'coor.npy'))

        try:
            if ext_sim_dict['collapse_cells']:
                pos = ext_sim_dict['collapse_pos']
                comp_start[2,:] = pos
                comp_end[2,:] = pos
                comp_coors[2,:] = pos
                length = np.sqrt(np.sum((comp_end - comp_start)**2, axis=0))
        except KeyError:
            pass
        n_compartments = len(comp_coors[0,:])
        n_elecs = ext_sim_dict['n_elecs']
        mapping = np.zeros((n_elecs,n_compartments))
        steps = ext_sim_dict['moi_steps']
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar

        for comp in xrange(n_compartments):
            if os.environ.has_key('DISPLAY'):
                percentage = (comp+1)*100/n_compartments
                stdout.write("\r%d %% complete" % percentage)
                stdout.flush()
            for elec in xrange(n_elecs):
                elec_pos = [elec_x[elec], elec_y[elec], elec_z]
                charge_pos = comp_coors[:,comp]
                if ext_sim_dict['include_elec']:
                    if ext_sim_dict['use_line_source']:
                        if comp == 0:
                            mapping[elec, comp] += self.potential_at_elec(\
                                charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                        else: 
                            mapping[elec, comp] += self.potential_at_elec_line_source(\
                                comp_start[:,comp], comp_end[:,comp],
                                comp_length[comp], elec_pos, ext_sim_dict['elec_radius'])
                    else:
                        mapping[elec, comp] += self.potential_at_elec(\
                            charge_pos, elec_pos, ext_sim_dict['elec_radius'])
                else:
                    if ext_sim_dict['use_line_source']:
                        mapping[elec, comp] += self.line_source_moi(\
                            comp_start, comp_end, comp_length, elec_pos)
                    else:
                        mapping[elec, comp] += self.isotropic_moi(\
                                charge_pos, elec_pos)
            if os.environ.has_key('DISPLAY'):                
                print ''
        try:
            os.mkdir(ext_sim_dict['output_folder'])
            os.mkdir(join(ext_sim_dict['output_folder'], 'mappings'))
        except OSError:
            pass
        np.save(join(ext_sim_dict['output_folder'], 'mappings', 'map_%s.npy' \
                %(neur_dict['name'])), mapping)
        return mapping

    def make_mapping_cython(self, ext_sim_dict, xmid=None, ymid=None, zmid=None,
                            xstart=None, ystart=None, zstart=None,
                            xend=None, yend=None, zend=None, morphology=None):
        """ Make a mapping given two arrays of electrode positions"""
        print "Making mapping. Cython style"
        elec_x = ext_sim_dict['elec_x'] # Array
        elec_y = ext_sim_dict['elec_y'] # Array
        elec_z = ext_sim_dict['elec_z'] # Scalar    

        if xmid != None:
            xmid = np.array(xmid, order='C')
            ymid = np.array(ymid, order='C')
            zmid = np.array(zmid, order='C')

        if xstart != None:
            xend = np.array(xend, order='C')
            yend = np.array(yend, order='C')
            zend = np.array(zend, order='C')
            xstart = np.array(xstart, order='C')
            ystart = np.array(ystart, order='C')
            zstart = np.array(zstart, order='C')
        
        if np.any(np.array([zstart, zend, zmid]) <= elec_z):
            raise Exception, 'cell %s, compartments cannot be below bottom boundary!' % morphology
            
        if ext_sim_dict['include_elec']:
            n_avrg_points = ext_sim_dict['n_avrg_points']
            elec_r = ext_sim_dict['elec_radius']
            if ext_sim_dict['use_line_source']:
                function =  LS_with_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, n_avrg_points, elec_r,
                        elec_x, elec_y, xstart, ystart, zstart, xend, yend, zend]
            else:
                function = PS_with_elec_mapping
                func_args = [self.sigma_T, self.sigma_S, elec_z,
                        self.steps, n_avrg_points,
                        elec_r, elec_x, elec_y, xmid, ymid, zmid]
        else:
            if ext_sim_dict['use_line_source']:
                function = LS_without_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, elec_x, elec_y,
                        xstart, ystart, zstart, xend, yend, zend]
            else:
                function = PS_without_elec_mapping
                func_args = [self.sigma_T, self.sigma_S,
                        elec_z, self.steps, elec_x, elec_y, xmid, ymid, zmid]
        mapping = function(*func_args)
        return mapping
