
from cython_funcs import *
import random
import unittest
import numpy as np
from MoI import MoI
import pylab as pl
import time
try:
    from ipdb import set_trace
except:
    pass

class TestMoI(unittest.TestCase):

    def test_cython_LS_with_electrode(self):
        """ Test if cython extentions works as expected """
        
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200.,
            'steps' : 20}

        xstart = np.array([100., 0, -100.,-110])
        xend =   np.array([100., 0, -100.,-110]) + 100.

        ystart = np.array([100., 0, -100, 200.])
        yend =   np.array([100., 0, -100, 200]) +100.
        
        zstart = np.array([10, 0, -10, -50.])
        zend =   np.array([10, 0, -10, -50]) + 10.

        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.

        #elec_x = np.array([0.])
        #elec_y = np.array([0.])

        
        elec_z = -set_up_parameters['slice_thickness']/2.
        elec_r = 1
        n_avrg_points = 100

        ext_sim_dict = {'elec_x': elec_x,
                        'elec_y': elec_y,
                        'elec_z': elec_z,
                        'include_elec': True,
                        'use_line_source': True,
                        'moi_steps': set_up_parameters['steps'],
                        'n_avrg_points': n_avrg_points,
                        'elec_radius':elec_r
                        }
        moi = MoI(set_up_parameters = set_up_parameters)
        t0 = time.time()
        mapping = LS_with_elec_mapping(set_up_parameters['sigma_T'],
                                       set_up_parameters['sigma_S'],
                                       elec_z, set_up_parameters['steps'],
                                       n_avrg_points, elec_r,
                                       elec_x, elec_y, xstart, ystart, zstart,
                                       xend, yend, zend)
        t_cy = time.time() - t0
        t0 = time.time()
        mapping2 = moi.make_mapping_standalone(ext_sim_dict, xstart=xstart, ystart=ystart, zstart=zstart,
                                            xend=xend, yend=yend, zend=zend)
        t_py = time.time() - t0
        rel_error = np.abs((mapping - mapping2)/mapping)
        print "\nLS with electrode cython speed-up: ", t_py/t_cy
        self.assertLessEqual(np.max(rel_error), 0.001)
        
    def test_cython_LS_without_electrode(self):
        """ Test if cython extentions works as expected """
        
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200,
            'steps' : 20}

        xstart = np.array([100, 0, -100,-110.])
        xend =   np.array([100, 0, -100,-110]) + 100.

        ystart = np.array([100, 0, -100, 200.])
        yend =   np.array([100, 0, -100, 200.]) +100
        
        zstart = np.array([10, 0, -10, -50.])
        zend =   np.array([10, 0, -10, -50.]) + 10

        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.

        elec_z = -set_up_parameters['slice_thickness']/2.

        ext_sim_dict = {'elec_x': elec_x,
                        'elec_y': elec_y,
                        'elec_z': elec_z,
                        'include_elec': False,
                        'use_line_source': True,
                        'moi_steps': set_up_parameters['steps'],                    
                        }
        moi = MoI(set_up_parameters = set_up_parameters)
        t0 = time.time()
        mapping = LS_without_elec_mapping(set_up_parameters['sigma_T'],
                                          set_up_parameters['sigma_S'],
                                          elec_z, set_up_parameters['steps'],
                                          elec_x, elec_y, xstart, ystart, zstart,
                                          xend, yend, zend)
        t_cy = time.time() - t0
        t0 = time.time()
        mapping2 = moi.make_mapping_standalone(ext_sim_dict, xstart=xstart, ystart=ystart, zstart=zstart,
                                            xend=xend, yend=yend, zend=zend)
        t_py = time.time() - t0
        rel_error = np.abs((mapping - mapping2)/mapping)
        print "\nLS no electrode cython speed-up: ", t_py/t_cy
        self.assertLessEqual(np.max(rel_error), 0.001)
        
    def test_cython_PS_without_electrode(self):
        """ Test if cython extentions works as expected """
        
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200,
            'steps' : 2}

        xmid = np.array([100,0,-100.])
        ymid = np.array([100,0,-100.])
        zmid = np.array([10,0,-10.])

        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.
        elec_z = -set_up_parameters['slice_thickness']/2.
        ext_sim_dict = {'elec_x': elec_x,
                        'elec_y': elec_y,
                        'elec_z': elec_z,
                        'include_elec': False,
                        'use_line_source': False,
                        'moi_steps': set_up_parameters['steps']
                        }
        moi = MoI(set_up_parameters = set_up_parameters)
        t0 = time.time()
        mapping = PS_without_elec_mapping(set_up_parameters['sigma_T'],
                            set_up_parameters['sigma_S'],
                            elec_z, set_up_parameters['steps'],
                            elec_x, elec_y, xmid, ymid, zmid)
        t_cy = time.time() - t0
        t0 = time.time()
        mapping2 = moi.make_mapping_standalone(ext_sim_dict, xmid=xmid, ymid=ymid, zmid=zmid)
        t_py = time.time() - t0
        rel_error = np.abs((mapping - mapping2)/mapping)
        print "\nPS no electrode cython speed-up: ", t_py/t_cy
        self.assertAlmostEqual(np.max(rel_error), 0.0, 6)

    def test_cython_PS_with_electrode(self):
        """ Test if cython extentions works as expected """
        
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200,
            'steps' : 20}

        xmid = 1000*np.array([0.1,0,-0.1])
        ymid = 1000*np.array([0.1,0,-0.1])
        zmid = 1000*np.array([0.01,0,-0.01])

        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.
        elec_z = -set_up_parameters['slice_thickness']/2.
        elec_r = 1
        n_avrg_points = 100
        ext_sim_dict = {'elec_x': elec_x,
                        'elec_y': elec_y,
                        'elec_z': elec_z,
                        'include_elec': True,
                        'use_line_source': False,
                        'elec_radius': elec_r,
                        'moi_steps': set_up_parameters['steps'],
                        'n_avrg_points': n_avrg_points,
                        }
        moi = MoI(set_up_parameters = set_up_parameters)
        
        t0 = time.time()
        mapping = PS_with_elec_mapping(set_up_parameters['sigma_T'], set_up_parameters['sigma_S'], elec_z,
                                       set_up_parameters['steps'], n_avrg_points, elec_r, elec_x, elec_y,
                                       xmid, ymid, zmid)
        t_cy = time.time() - t0
        t0 = time.time()
        mapping2 = moi.make_mapping_standalone(ext_sim_dict, xmid, ymid, zmid)
        t_py = time.time() - t0
        print "\nPS with electrode cython speed-up: ", t_py/t_cy
        rel_error = np.abs((mapping - mapping2)/mapping)
        #print mapping
        #print mapping2
        #print mapping- mapping2
        self.assertLessEqual(np.max(rel_error), 0.001)



    def test_cython_mapping(self):
        
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200.,
            'steps' : 20}

        xstart = np.array([100., 0, -100.,-110])
        xend =   np.array([100., 0, -100.,-110]) + 100.

        ystart = np.array([100., 0, -100, 200.])
        yend =   np.array([100., 0, -100, 200]) +100.
        
        zstart = np.array([10, 0, -10, -50.])
        zend =   np.array([10, 0, -10, -50]) + 10.



        xmid = xstart
        ymid = ystart
        zmid = zstart

        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.

        #elec_x = np.array([0.])
        #elec_y = np.array([0.])

        
        elec_z = -set_up_parameters['slice_thickness']/2.
        elec_r = 1
        n_avrg_points = 100

        ext_sim_dict = {'elec_x': elec_x,
                        'elec_y': elec_y,
                        'elec_z': elec_z,
                        'include_elec': False,
                        'use_line_source': False,
                        'moi_steps': set_up_parameters['steps'],
                        'n_avrg_points': n_avrg_points,
                        'elec_radius':elec_r
                        }
        moi = MoI(set_up_parameters = set_up_parameters)
        mapping = moi.make_mapping_standalone(ext_sim_dict, xstart=xstart,
                                              ystart=ystart, zstart=zstart,
                                              xend=xend, yend=yend, zend=zend,
                                              xmid=xmid, ymid=ymid, zmid=zmid)
        
        mapping2 = moi.make_mapping_cython(ext_sim_dict, xstart=xstart,
                                           ystart=ystart, zstart=zstart,
                                           xend=xend, yend=yend, zend=zend,
                                           xmid=xmid, ymid=ymid, zmid=zmid)
        
        error = np.abs((mapping - mapping2)/mapping2)
        
        self.assertLessEqual(np.max(error), 0.001)

    def test_homogeneous(self):
        """If saline and tissue has same conductivity, MoI formula
        should return 2*(inf homogeneous point source)."""
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 0.3, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [0, 0, -set_up_parameters['slice_thickness']/2]
        dist = np.sqrt( np.sum(np.array(charge_pos) - np.array(elec_pos))**2)
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_T'])\
                       * imem/(dist)
        returned_ans = Moi.isotropic_moi(charge_pos, elec_pos, imem)
        self.assertAlmostEqual(expected_ans, returned_ans, 6)

    def test_saline_effect(self):
        """ If saline conductivity is bigger than tissue conductivity, the
        value of 2*(inf homogeneous point source) should be bigger
        than value returned from MoI"""
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 3.0, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos = [0,0,0]
        elec_pos = [0, 0, -set_up_parameters['slice_thickness']/2]
        dist = np.sqrt( np.sum((np.array(charge_pos) - np.array(elec_pos))**2))
        expected_ans = 2/(4*np.pi*set_up_parameters['sigma_T']) * imem/dist
        returned_ans = Moi.isotropic_moi(charge_pos, elec_pos, imem)
        self.assertGreater(expected_ans, returned_ans)

    def test_charge_closer(self):
        """ If charge is closer to electrode, the potential should
        be greater"""
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 3.0, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        charge_pos_1 = [0, 0, 0]
        charge_pos_2 = [0, 0, -set_up_parameters['slice_thickness']/4]
        elec_pos = [0, 0, -set_up_parameters['slice_thickness']/2]
        returned_ans_1 = Moi.isotropic_moi(charge_pos_1, elec_pos, imem)
        returned_ans_2 = Moi.isotropic_moi(charge_pos_2, elec_pos, imem)
        self.assertGreater(returned_ans_2, returned_ans_1)
        
    def test_within_domain_check(self):
        """ Test if unvalid electrode or charge position raises RuntimeError.
        """
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 3.0, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        a = set_up_parameters['slice_thickness']/2
        invalid_positions = [[0, 0, -a - 120],
                            [0, 0, +a + 120]]
        valid_position = [0,0,0]


        with self.assertRaises(RuntimeError):
            Moi.in_domain(valid_position, [1,0,0])
        
        with self.assertRaises(RuntimeError):
            Moi.isotropic_moi(valid_position, valid_position)
            for pos in invalid_positions:
                with self.assertRaises(RuntimeError):
                    Moi.isotropic_moi(valid_position, pos)
                    Moi.isotropic_moi(pos, valid_position)

        xstart = np.array([100., 0, -100.,-110])
        xend =   np.array([100., 0, -100.,-110]) + 100.

        ystart = np.array([100., 0, -100, 200.])
        yend =   np.array([100., 0, -100, 200]) +100.
        
        zstart = np.array([10, 0, -10, -50.])
        zend =   np.array([10, 0, -10, -50]) + 10.
        
        xmid = np.array([100,0,-100.])
        ymid = np.array([100,0,-100.])
        zmid = np.array([10,0,-10.])

        
        elec_x = (np.arange(3) - 1)*50.
        elec_y = (np.arange(3) - 1)*50.
        elec_z = -1
        n_avrg_points = 1
        elec_r = 1
        with self.assertRaises(RuntimeError):
            mapping = LS_with_elec_mapping(set_up_parameters['sigma_T'],
                                           set_up_parameters['sigma_S'],
                                           elec_z, set_up_parameters['steps'],
                                           n_avrg_points, elec_r, elec_x, elec_y, xstart, ystart, zstart,
                                           xend, yend, zend)
        with self.assertRaises(RuntimeError):
            mapping = LS_without_elec_mapping(set_up_parameters['sigma_T'],
                                              set_up_parameters['sigma_S'],
                                              elec_z, set_up_parameters['steps'],
                                              elec_x, elec_y, xstart, ystart, zstart,
                                              xend, yend, zend)

        with self.assertRaises(RuntimeError):
            mapping = PS_without_elec_mapping(set_up_parameters['sigma_T'],
            set_up_parameters['sigma_S'],
            elec_z, set_up_parameters['steps'],
            elec_x, elec_y, xmid, ymid, zmid)

        with self.assertRaises(RuntimeError):
            mapping = PS_with_elec_mapping(set_up_parameters['sigma_T'], 
                                           set_up_parameters['sigma_S'], elec_z,
                                           set_up_parameters['steps'], 
                                           n_avrg_points, elec_r, elec_x, elec_y,
                                           xmid, ymid, zmid)

        elec_z = -100
        zmid += 150
        zstart += 150
        zend += 150
        with self.assertRaises(RuntimeError):
            mapping = LS_with_elec_mapping(set_up_parameters['sigma_T'],
                                           set_up_parameters['sigma_S'],
                                           elec_z, set_up_parameters['steps'],
                                           n_avrg_points, elec_r, elec_x, elec_y, xstart, ystart, zstart,
                                           xend, yend, zend)
        with self.assertRaises(RuntimeError):
            mapping = LS_without_elec_mapping(set_up_parameters['sigma_T'],
                                              set_up_parameters['sigma_S'],
                                              elec_z, set_up_parameters['steps'],
                                              elec_x, elec_y, xstart, ystart, zstart,
                                              xend, yend, zend)

        with self.assertRaises(RuntimeError):
            mapping = PS_without_elec_mapping(set_up_parameters['sigma_T'],
            set_up_parameters['sigma_S'],
            elec_z, set_up_parameters['steps'],
            elec_x, elec_y, xmid, ymid, zmid)

        with self.assertRaises(RuntimeError):
            mapping = PS_with_elec_mapping(set_up_parameters['sigma_T'], 
                                           set_up_parameters['sigma_S'], elec_z,
                                           set_up_parameters['steps'], 
                                           n_avrg_points, elec_r, elec_x, elec_y,
                                           xmid, ymid, zmid)
    def test_if_anisotropic(self):
        """ Test if it can handle anisotropies
        """
        set_up_parameters = {
            'sigma_G': [1.0, 1.0, 1.0], # Below electrode
            'sigma_T': [0.1, 0.1, 1.0], # Tissue
            'sigma_S': [0.0, 0.0, 0.0], # Saline
            'slice_thickness': 200,
            'steps' : 2}
        Moi = MoI(set_up_parameters = set_up_parameters)
        self.assertTrue(Moi.is_anisotropic)

        set_up_parameters = {
            'sigma_G': [1.0], # Below electrode
            'sigma_T': [1.0], # Tissue
            'sigma_S': [0.0], # Saline
            'slice_thickness': 200,
            'steps' : 2}
        with self.assertRaises(RuntimeError):
            Moi = MoI(set_up_parameters = set_up_parameters)

        set_up_parameters = {
            'sigma_G': [1.0, 2.0, 3.0], # Below electrode
            'sigma_T': 1.0, # Tissue
            'sigma_S': 0.0, # Saline
            'slice_thickness': 200,
            'steps' : 2}
        with self.assertRaises(RuntimeError):
            Moi = MoI(set_up_parameters = set_up_parameters)


    def atest_very_anisotropic(self):
        """ Made to find error in very anisotropic case close to upper layer
        """
        set_up_parameters = {
            'sigma_G': [1.0, 1.0, 1.0], # Below electrode
            'sigma_T': [0.1, 0.1, 1.0], # Tissue
            'sigma_S': [0.0, 0.0, 0.0], # Saline
            'slice_thickness': 200,
            'steps' : 2}
        Moi = MoI(set_up_parameters = set_up_parameters)
        imem = 1.2
        a = set_up_parameters['slice_thickness']/2.
        high_position = [0, 0, 90]
        low_position = [0, 0, -a + 10]
        x_array = np.linspace(-200, 200, 41)
        y_array = np.linspace(-100, 100, 21)
        values_high = []
        values_low = []
        for y in y_array:
            for x in x_array:
                values_high.append([x,y, Moi.ad_hoc_anisotropic(\
                    charge_pos = high_position, elec_pos = [x,y,-100])])
                values_low.append([x,y, Moi.ad_hoc_anisotropic(\
                    charge_pos = low_position, elec_pos = [x,y,-100])])
        values_high = np.array(values_high)
        values_low = np.array(values_low)
        pl.subplot(211)
        pl.scatter(values_high[:,0], values_high[:,1], c = values_high[:,2])
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(212)
        pl.scatter(values_low[:,0], values_low[:,1], c = values_low[:,2])
        pl.colorbar()
        pl.axis('equal')
        pl.show()


    def test_big_average(self):
        """ Testing average over electrode with many values"""
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 3.0, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        a = set_up_parameters['slice_thickness']/2.
        Moi = MoI(set_up_parameters = set_up_parameters)
        r = 30
        charge_pos = [0, 0, 0]
        elec_pos = [0, 0, -a]
        n_avrg_points = 100
        phi = Moi.potential_at_elec_big_average(elec_pos, r, n_avrg_points,
                                                Moi.point_source_moi_at_elec,
                                                [charge_pos])


    def atest_moi_line_source(self):
        """ Testing infinite isotropic moi line source formula"""
        set_up_parameters = {
            'sigma_G': 0.0, # Below electrode
            'sigma_T': 0.3, # Tissue
            'sigma_S': 3.0, # Saline
            'slice_thickness': 200,
            'steps' : 20}
        a = set_up_parameters['slice_thickness']/2.
        Moi = MoI(set_up_parameters = set_up_parameters)

        comp_start = [-50, -100, 90]
        comp_end = [10, 100, -90]
        comp_mid = (np.array(comp_end) + np.array(comp_start))/2
        comp_length = np.sqrt( np.sum((np.array(comp_end) - np.array(comp_start))**2))
        elec_y = np.linspace(-150, 150, 50)
        elec_x = np.linspace(-150, 150, 50)
        phi_LS = []
        phi_PS = []
        phi_PSi = []
        y = []
        x = []
        points = 200
        s = np.array(comp_end) - np.array(comp_start)
        ds = s / (points-1)
 
        for x_pos in xrange(len(elec_x)):
            for y_pos in xrange(len(elec_y)):
                phi_PS.append(Moi.isotropic_moi(comp_mid, [elec_x[x_pos], elec_y[y_pos], -100]))
                delta = 0
                for step in xrange(points):
                    pos = comp_start + ds*(step)
                    delta += Moi.isotropic_moi(\
                        pos, [elec_x[x_pos], elec_y[y_pos], -100], imem = 1./(points+1))
                phi_PSi.append(delta)
                                       
                x.append(elec_x[x_pos])
                y.append(elec_y[y_pos])
        x = np.array(x)
        y = np.array(y)
        cyth = LS_without_elec_mapping(set_up_parameters['sigma_T'],
                                       set_up_parameters['sigma_S'],
                                       -100, set_up_parameters['steps'],
                                       x, y, np.array([comp_start[0]]),
                                       np.array([comp_start[1]]), np.array([comp_start[2]]),
                                       np.array([comp_end[0]]), np.array([comp_end[1]]), np.array([comp_end[2]]))
        import pylab as pl
        pl.subplot(411)
    
        pl.scatter(x,y, c=cyth[:,0], s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(412)
        pl.scatter(x,y, c=phi_PS, s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()
        pl.subplot(413)
        pl.scatter(x,y, c=phi_PSi, s=2, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()

        
        pl.subplot(414)       
        pl.scatter(x,y, c=(np.array(cyth[:,0]) - np.array(phi_PSi)), s=1, edgecolors='none')
        pl.axis('equal')
        pl.colorbar()       
        pl.savefig('line_source_test_cython.png')
        
if __name__ == '__main__':
    unittest.main()
