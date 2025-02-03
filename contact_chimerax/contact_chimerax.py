#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:50:20 2024

@author: shengkai
"""

session = 1

#%%
import chimerax.core.commands.run as run
import numpy as np
import pandas as pd
import os

name_list = [i for i in os.listdir("/Users/shengkai/Dropbox (Scripps Research)/pna/all_seg_pdb") if i.split(".")[-1] == "pdb"]
name_list.sort()
for i in name_list:
    run(session, "open '/Users/shengkai/Dropbox (Scripps Research)/pna/all_seg_pdb/%s'"%i)

run(session, "hide all models")
mol_list = session.models.list()
#num_mol = len(name_list)
num_mol = 6
contact_mat = np.zeros([num_mol,num_mol])

for i in range(num_mol-1):
    run(session, "show #%i models"%(i+1))
    print("calculating %s"%name_list[i])
    contact_mat[i,i] = 1
    for j in range(i+1, num_mol):
        run(session, "show #%i models"%(j+1))
        contact_mat[i,j] = len(run(session, "contacts resSeparation 5 intraModel false intraMol false ignoreHiddenModels true"))
        contact_mat[j,i] = contact_mat[i,j]
        run(session, "hide #%i models"%(j+1))
    run(session, "hide all models")
    
    