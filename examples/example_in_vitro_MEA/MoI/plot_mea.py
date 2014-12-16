import numpy as np
import pylab as pl
try:
    from ipdb import set_trace
except:
    pass
import os

def plot_mea(neuron_dict, ext_sim_dict, neural_sim_dict):
    pl.close('all')
    fig_all = pl.figure(figsize=[15,15])
    ax_all = fig_all.add_axes([0.1, 0.1, 0.8, 0.8], frameon=False)
    for elec in xrange(len(ext_sim_dict['elec_z'])):
        ax_all.plot(ext_sim_dict['elec_z'][elec], ext_sim_dict['elec_y'][elec], color='b',\
                marker='$E%i$'%elec, markersize=20 )    
    legends = []
    for i, neur in enumerate(neuron_dict):
        folder = os.path.join(neural_sim_dict['output_folder'], neuron_dict[neur]['name'])
        coor = np.load(os.path.join(folder,'coor.npy'))
        x,y,z = coor
        n_compartments = len(x)
        fig = pl.figure(figsize=[10, 10])
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=False)
        # Plot the electrodes
        for elec in xrange(len(ext_sim_dict['elec_z'])):
            ax.plot(ext_sim_dict['elec_z'][elec], ext_sim_dict['elec_y'][elec], color='b',\
                   marker='$%i$'%elec, markersize=20 )
        # Plot the neuron
        xmid, ymid, zmid = np.load(folder + '/coor.npy')
        xstart, ystart,zstart = np.load(folder + '/coor_start.npy')
        xend, yend, zend = np.load(folder + '/coor_end.npy')
        diam = np.load(folder + '/diam.npy')
        length = np.load(folder + '/length.npy')
        n_compartments = len(diam)
        for comp in xrange(n_compartments):
            if comp == 0:
                xcoords = pl.array([xmid[comp]])
                ycoords = pl.array([ymid[comp]])
                zcoords = pl.array([zmid[comp]])
                diams = pl.array([diam[comp]])    
            else:
                if zmid[comp] < 0.400 and zmid[comp] > -.400:  
                    xcoords = pl.r_[xcoords, pl.linspace(xstart[comp],
                                                         xend[comp], length[comp]*3*1000)]   
                    ycoords = pl.r_[ycoords, pl.linspace(ystart[comp],
                                                         yend[comp], length[comp]*3*1000)]   
                    zcoords = pl.r_[zcoords, pl.linspace(zstart[comp],
                                                         zend[comp], length[comp]*3*1000)]   
                    diams = pl.r_[diams, pl.linspace(diam[comp], diam[comp],
                                                length[comp]*3*1000)]
        argsort = pl.argsort(-xcoords)
        ax.scatter(zcoords[argsort], ycoords[argsort], s=20*(diams[argsort]*1000)**2,
                       c=xcoords[argsort], edgecolors='none', cmap='gray')
        ax_all.plot(zmid[0], ymid[0], marker='$%i$'%i, markersize=20, label='%i: %s' %(i, neur))
        #legends.append('%i: %s' %(i, neur))
        ax.axis(ext_sim_dict['plot_range'])
        ax.axis('equal')
        ax.axis(ext_sim_dict['plot_range'])
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('y [mm]')
        fig.savefig(os.path.join(neural_sim_dict['output_folder'],\
                  'neuron_figs', '%s.png' % neur))
    ax_all.axis('equal')
    ax.axis(ext_sim_dict['plot_range'])
    ax_all.set_xlabel('z [mm]')
    ax_all.set_ylabel('y [mm]')
    ax_all.legend()
    fig_all.savefig(os.path.join(neural_sim_dict['output_folder'], 'fig.png'))
 
