import numpy as np
import os
import scipy.io as sio

def get_footprint(stat_file, iscell_file): # return a N x H x W array 
    stat = np.load(stat_file, allow_pickle = True)
    iscell = np.load(iscell_file)
    n_rois = stat.shape[0] # Total rois detected 

    # Image size in pixels, use sbx.loadmat to double-check the original scanbox image 
    height = 512
    width = 796

    sp_footprint = np.zeros((n_rois, height, width))

    for i in range(n_rois):
        sp_footprint[i, stat[i]['ypix'], stat[i]['xpix']] = stat[i]['lam']
    
    # use iscell.npy to prune the  footprint

    cell_stat= stat[iscell[:,0]>0]
    num_cell = cell_stat.shape[0]

    cell_footprint = np.zeros((num_cell, height, width))

    for i in range(num_cell):
        cell_footprint[i, cell_stat[i]['ypix'], cell_stat[i]['xpix']] = cell_stat[i]['lam']
              
    return cell_footprint
