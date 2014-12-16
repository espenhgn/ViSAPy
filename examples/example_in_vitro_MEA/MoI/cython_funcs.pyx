#!/usr/bin/env python
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, log
DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def PS_without_elec_mapping(float sigma_T, float sigma_S, float elec_z, int steps,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid):
    """
    Arguments: float sigma_T, float sigma_S, float elec_z, int steps,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid

    Calculates the potential at the electrode plane elec_z of the MEA from point sources at xmid, ymid, zmid.
    To get potential use np.dot(mapping, imem) where imem is the transmembrane currents of the point sources.
    The conductivities sigma are expected to be scalars
    """

    if (zmid - elec_z <= 0).any():
        raise RuntimeError
    if (zmid + elec_z >= 0).any():
        raise RuntimeError
    
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] mapping
    cdef float W, delta
    cdef int comp, elec, n

    if (zmid - elec_z <= 0).any():
        raise RuntimeError, "zmid too high"
    if (zmid + elec_z >= 0).any():
        raise RuntimeError, "zmid too low"
    
    mapping = np.zeros((len(elec_x), len(xmid)))
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    
    for comp in xrange(len(xmid)):
        for elec in xrange(len(elec_x)):
            delta = 1/sqrt(pow(elec_y[elec] - ymid[comp], 2) +
                           pow(elec_x[elec] - xmid[comp], 2) +
                           pow(elec_z - zmid[comp], 2))
            n = 1
            while n < steps:
                delta += pow(W, n) * (1/sqrt(pow(elec_y[elec] - ymid[comp], 2) +
                                             pow(elec_x[elec] - xmid[comp], 2) +
                                             pow(-(4*n-1)*elec_z - zmid[comp], 2)) +
                                      1/sqrt(pow(elec_y[elec] - ymid[comp], 2) +
                                             pow(elec_x[elec] - xmid[comp], 2) +
                                             pow((4*n+1)*elec_z - zmid[comp], 2)))
                n += 1
            mapping[elec, comp] = delta
    return 2*mapping/(4*np.pi*sigma_T)

@cython.boundscheck(False)
def anisotropic_PS_without_elec_mapping(float elec_z, int steps,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] sigma_T,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] sigma_S,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
                np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid):
    """
    Calculate the moi point source potential at the MEA plane with saline conductivity
    sigma_S is scaled to k * sigma_T. Only for point sources without electrodes.
    Arguments: float elec_z, int steps,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] sigma_T,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] sigma_S,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid
    """

    if len(sigma_T) != 3:
        raise RuntimeError, "Tissue conductivity must have length 3"
    if len(sigma_S) != 3:
        raise RuntimeError, "Saline conductivity must have length 3"
    if (zmid - elec_z <= 0).any():
        raise RuntimeError
    if (zmid + elec_z >= 0).any():
        raise RuntimeError
    
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ratios, sigma_S_scaled
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] mapping
    cdef float scale_factor, sigma_T_net, sigma_S_net, anis_W, delta
    cdef int n, comp, elec
    
    ratios = sigma_S / sigma_T
    if np.abs(ratios[0] - ratios[2]) <= 1e-15:
        scale_factor = ratios[0]
    elif np.abs(ratios[1] - ratios[2]) <= 1e-15:
        scale_factor = ratios[1]

    sigma_S_scaled = scale_factor * sigma_T
    sigma_T_net = np.sqrt(sigma_T[0] * sigma_T[2])
    sigma_S_net = np.sqrt(sigma_S_scaled[0] * sigma_S_scaled[2])
    anis_W = (sigma_T_net - sigma_S_net)/(sigma_T_net + sigma_S_net)
    mapping = np.zeros((len(elec_x), len(xmid)))

    for comp in xrange(len(xmid)):
        for elec in xrange(len(elec_x)):
            delta = 1/sqrt(sigma_T[0]*sigma_T[2]*pow(elec_y[elec] - ymid[comp], 2) + \
                           sigma_T[0]*sigma_T[1]*pow(elec_z - zmid[comp], 2) + \
                           sigma_T[1]*sigma_T[2]*pow(elec_x[elec] - xmid[comp], 2)) 
            n = 1
            while n < steps:
                delta += pow(anis_W, n) * (1/sqrt(sigma_T[0]*sigma_T[2]*pow(elec_y[elec] - ymid[comp], 2) + \
                                           sigma_T[0]*sigma_T[1]*pow(-(4*n-1)*elec_z - zmid[comp], 2) + \
                                           sigma_T[1]*sigma_T[2]*pow(elec_x[elec] - xmid[comp], 2)) +\
                                          1/sqrt(sigma_T[0]*sigma_T[2]*pow(elec_y[elec] - ymid[comp], 2) + \
                                            sigma_T[0]*sigma_T[1]*pow((4*n+1)*elec_z - zmid[comp], 2) + \
                                            sigma_T[1]*sigma_T[2]*pow(elec_x[elec] - xmid[comp], 2)))
                n += 1
            mapping[elec, comp] = delta
    return 2/(4*np.pi) * mapping

@cython.boundscheck(False)
def PS_with_elec_mapping(float sigma_T, float sigma_S, float elec_z, int steps,
                          int n_avrg_points, float elec_r,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid):
    """ 
    Arguments:
           float sigma_T, float sigma_S, float elec_z, int steps,
                          int n_avrg_points, float elec_r,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xmid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ymid,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zmid 
    """
    
    if (zmid - elec_z <= 0).any():
        raise RuntimeError
    if (zmid + elec_z >= 0).any():
        raise RuntimeError
    
    cdef float W, delta, pt_delta, xpos, ypos
    cdef int comp, elec, n

    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] mapping
    mapping = np.zeros((len(elec_x), len(xmid)))
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)

    for comp in xrange(len(xmid)):
        for elec in xrange(len(elec_x)):
            delta = 0
            for pt in xrange(n_avrg_points):
                xpos = (np.random.rand() - 0.5)* 2 * elec_r
                ypos = (np.random.rand() - 0.5)* 2 * elec_r
                # If outside electrode
                while (xpos*xpos + ypos*ypos) > elec_r*elec_r:
                    xpos = (np.random.rand() - 0.5)* 2 * elec_r
                    ypos = (np.random.rand() - 0.5)* 2 * elec_r
                
                pt_delta = 1/sqrt(pow(elec_y[elec] + ypos - ymid[comp], 2) +
                                  pow(elec_x[elec] + xpos - xmid[comp], 2) +
                                  pow(elec_z - zmid[comp], 2))
                n = 1
                while n < steps:
                    pt_delta += pow(W, n) * (
                        1/sqrt(pow(elec_y[elec] + ypos - ymid[comp], 2) +
                               pow(elec_x[elec] + xpos - xmid[comp], 2) +
                               pow(-(4*n-1)*elec_z - zmid[comp], 2)) +
                        1/sqrt(pow(elec_y[elec] + ypos - ymid[comp], 2) +
                               pow(elec_x[elec] + xpos - xmid[comp], 2) +
                               pow((4*n+1)*elec_z - zmid[comp], 2)))
                    n += 1
                delta += pt_delta 
            mapping[elec, comp] = delta / n_avrg_points
    return 2*mapping/(4*np.pi*sigma_T)

@cython.boundscheck(False)
def _LS_omega(float a_x, float a_y, float a_z, float comp_length, float dx, float dy, float dz):
    #See Rottman integration formula 46) page 137 for explanation
    cdef float factor_a, factor_b, factor_c, b_2_ac, num, den
    factor_a = comp_length*comp_length
    factor_b = - a_x*dx - a_y*dy - a_z * dz
    factor_c = a_x*a_x + a_y*a_y + a_z*a_z
    b_2_ac = factor_b*factor_b - factor_a * factor_c
    if np.abs(b_2_ac) <= 1e-16:
        num = factor_a + factor_b
        den = factor_b
    else:
        num = factor_a + factor_b + \
              comp_length*sqrt(factor_a + 2*factor_b + factor_c)
        den = factor_b + comp_length*sqrt(factor_c)
    return log(num/den)

@cython.boundscheck(False)
def LS_without_elec_mapping(float sigma_T, float sigma_S, float elec_z, int steps,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ystart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zend):
    """ 
    Arguments:
            float sigma_T, float sigma_S, float elec_z, int steps,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ystart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zend
    """
    if (zstart - elec_z <= 0).any():
        raise RuntimeError
    if (zstart + elec_z >= 0).any():
        raise RuntimeError
    if (zend - elec_z <= 0).any():
        raise RuntimeError
    if (zend + elec_z >= 0).any():
        raise RuntimeError    
    
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] mapping
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] length
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dxs
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dys
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dzs

    cdef float W, delta, a, a_x, a_y, a_z0, a_z1, a_z2
    cdef int comp, elec, n

    dxs = xend - xstart
    dys = yend - ystart
    dzs = zend - zstart
    length = (dxs**2 + dys**2 + dzs**2)**(0.5)
    a = - elec_z
    mapping = np.zeros((len(elec_x), len(xstart)))
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    for comp in xrange(len(xstart)):
        for elec in xrange(len(elec_x)):
            a_z0 = -a - zstart[comp]
            a_x = elec_x[elec] - xstart[comp]
            a_y = elec_y[elec] - ystart[comp]
            n = 1
            delta = _LS_omega(a_x, a_y, a_z0, length[comp], dxs[comp], dys[comp], dzs[comp])
            while n < steps:
                a_z1 = (4*n-1)*a - zstart[comp]
                a_z2 =-(4*n+1)*a - zstart[comp]
                delta += pow(W, n) * (_LS_omega(a_x, a_y, a_z1, length[comp], dxs[comp], dys[comp], dzs[comp]) +
                                      _LS_omega(a_x, a_y, a_z2, length[comp], dxs[comp], dys[comp], dzs[comp]))
                n += 1
            mapping[elec, comp] = delta / length[comp] 
    return 2 * mapping /(4*np.pi*sigma_T)

@cython.boundscheck(False)
def LS_with_elec_mapping(float sigma_T, float sigma_S, float elec_z, int steps,
                           int n_avrg_points, float elec_r,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ystart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zend):
    """ 
    Arguments:
            float sigma_T, float sigma_S, float elec_z, int steps,
                           int n_avrg_points, float elec_r,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_x,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] elec_y,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] ystart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zstart,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] xend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] yend,
            np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] zend
    """



    
    if (zstart - elec_z <= 0).any():
        raise RuntimeError
    if (zstart + elec_z >= 0).any():
        raise RuntimeError
    if (zend - elec_z <= 0).any():
        raise RuntimeError
    if (zend + elec_z >= 0).any():
        raise RuntimeError    
    
    cdef np.ndarray[DTYPE_t, ndim=2, negative_indices=False, mode='c'] mapping
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] length
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dxs
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dys
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False, mode='c'] dzs

    cdef float W, delta, a, a_x, a_y, a_z0, a_z1, a_z2, ypos, xpos, pt_delta
    cdef int comp, elec, n

    dxs = xend - xstart
    dys = yend - ystart
    dzs = zend - zstart
    length = (dxs**2 + dys**2 + dzs**2)**(0.5)
    a = - elec_z
    mapping = np.zeros((len(elec_x), len(xstart)))
    W = (sigma_T - sigma_S)/(sigma_T + sigma_S)
    for comp in xrange(len(xstart)):
        for elec in xrange(len(elec_x)):
            delta = 0
            for pt in xrange(n_avrg_points):
                xpos = (np.random.rand() - 0.5)* 2 * elec_r
                ypos = (np.random.rand() - 0.5)* 2 * elec_r
                # If outside electrode
                while (xpos*xpos + ypos*ypos) > elec_r*elec_r:
                    xpos = (np.random.rand() - 0.5)* 2 * elec_r
                    ypos = (np.random.rand() - 0.5)* 2 * elec_r
                a_z0 = -a - zstart[comp]
                a_x = elec_x[elec] + xpos - xstart[comp]
                a_y = elec_y[elec] + ypos - ystart[comp]
                n = 1
                pt_delta = _LS_omega(a_x, a_y, a_z0, length[comp],
                                     dxs[comp], dys[comp], dzs[comp])
                while n < steps:
                    a_z1 = (4*n-1)*a - zstart[comp]
                    a_z2 =-(4*n+1)*a - zstart[comp]
                    pt_delta += pow(W, n) * (_LS_omega(a_x, a_y, a_z1, length[comp], dxs[comp],
                                                       dys[comp], dzs[comp]) +
                                             _LS_omega(a_x, a_y, a_z2, length[comp], dxs[comp],
                                                       dys[comp], dzs[comp]))
                    n += 1
                delta += pt_delta / n_avrg_points
            mapping[elec, comp] = delta / length[comp]             
    return 2 * mapping /(4*np.pi*sigma_T) 

