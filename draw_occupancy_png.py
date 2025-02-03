#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:11:30 2024

@author: shengkai
"""

import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os

#%%
def save_density(data, grid_spacing, outfilename, origin=None):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = grid_spacing
        if origin is not None:
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("done")


#%%
def load_mrc(dp, box):
    if dp[-1]!="/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)

#%%

seg_4ybb_name, seg_4ybb = load_mrc('/Users/shengkai/Dropbox (Scripps Research)/Williamson Lab/Paper_Kai/PDBs_forAlignment_box160/reset_origin/',160)

pna_name, pna_mrc = load_mrc('/Users/shengkai/Dropbox (Scripps Research)/pna/maps/scaled_adjusted_rename_map/', 160)

#%%
seg_4ybb_name = [i.split(".")[0] for i in seg_4ybb_name]
pna_name = [i.split(".")[0] for i in pna_name]
#%%
pna_mrc_bi = pna_mrc > np.std(pna_mrc) * 4.5

#%%
occ_mat = np.matmul(pna_mrc_bi, seg_4ybb.T)

#%%
occ_mat_norm = occ_mat/seg_4ybb.sum(1)
#%%
occ_mat_norm = occ_mat_norm/occ_mat_norm[:,132].reshape(-1,1)

#%%
import pandas as pd
pd.DataFrame(occ_mat_norm, columns = seg_4ybb_name, index = pna_name).to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/dependency/new_occ.csv")
#%%

def mass_center(volume):
    x_ind, y_ind, z_ind = np.indices([160,160,160])
    total_mass = np.sum(volume)
    center_of_mass_x = np.sum(x_ind * volume) / total_mass
    center_of_mass_y = np.sum(y_ind * volume) / total_mass
    center_of_mass_z = np.sum(z_ind * volume) / total_mass
    mass_center_ =  np.array([int(center_of_mass_x),
                             int(center_of_mass_y),
                             int(center_of_mass_z)])
    return mass_center_



for i, seg_name in enumerate(seg_4ybb_name):
    temp_seg = seg_4ybb[i]
    temp_occ_arr = np.matmul(pna_mrc_bi, temp_seg.T)/temp_seg.sum()
    fig_name = seg_name + "_occ.png"
    fig, axs = plt.subplots(7, 7, figsize=(10, 10))

    axs = axs.flatten()
    
    seg_x, seg_y, seg_z = mass_center(temp_seg.reshape((160,160,160)))
    
    vmax_temp = temp_seg.sum() ** (1/3)
    
    for k, order in enumerate(temp_occ_arr.argsort()):
        temp_pna_name = pna_name[order]
        temp_occ = temp_occ_arr[order]
        temp_mrc = np.zeros(160**3)
        temp_mrc[temp_seg ==1] = pna_mrc_bi[order][temp_seg ==1]
        temp_mrc = temp_mrc.reshape((160,160,160))
        box_mrc = temp_mrc[seg_x-25:seg_x+25,seg_y-25:seg_y+25,seg_z-25:seg_z+25]
        axs[k].imshow(box_mrc.sum(0), vmax = vmax_temp, vmin = 0)
        axs[k].text(0.5, 0.8, '%s \n %2f'%(temp_pna_name,temp_occ), color='white', fontsize=8,
                ha='center', va='center', transform=axs[k].transAxes)
        axs[k].axis('off')  # Turn off axis labels for better visualization
    
    plt.tight_layout()
    fig.suptitle(seg_name, fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()
    plt.savefig('/Users/shengkai/Dropbox (Scripps Research)/pna/occupancy_png/' + fig_name)
    plt.close()
        
        
    


#%%
for i, seg_name in enumerate(seg_4ybb_name):
    temp_seg = seg_4ybb[i]
    temp_occ_arr = np.matmul(pna_mrc_bi, temp_seg.T)/temp_seg.sum()/np.matmul(pna_mrc_bi, seg_4ybb[132].T)*seg_4ybb[132].sum()
    fig_name = seg_name + "_occ.png"
    fig, axs = plt.subplots(7, 7, figsize=(10, 10))

    axs = axs.flatten()
    
    seg_x, seg_y, seg_z = mass_center(temp_seg.reshape((160,160,160)))
    
    vmax_temp = temp_seg.sum() ** (1/3)
    
    for k, order in enumerate(temp_occ_arr.argsort()):
        temp_pna_name = pna_name[order]
        temp_occ = temp_occ_arr[order]
        temp_mrc = np.zeros(160**3)
        temp_mrc[temp_seg ==1] = pna_mrc_bi[order][temp_seg ==1]
        temp_mrc = temp_mrc.reshape((160,160,160))
        box_mrc = temp_mrc[seg_x-25:seg_x+25,seg_y-25:seg_y+25,seg_z-25:seg_z+25]
        axs[k].imshow(box_mrc.sum(0), vmax = vmax_temp, vmin = 0)
        axs[k].text(0.5, 0.8, '%s \n %2f'%(temp_pna_name,temp_occ), color='white', fontsize=8,
                ha='center', va='center', transform=axs[k].transAxes)
        axs[k].axis('off')  # Turn off axis labels for better visualization
    
    plt.tight_layout()
    fig.suptitle(seg_name, fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()
    plt.savefig('/Users/shengkai/Dropbox (Scripps Research)/pna/occpancy_png_ul24_norm/' + fig_name)
    plt.close()
        



