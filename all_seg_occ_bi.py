#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:08:38 2024

@author: shengkai
"""
import os
import mrcfile
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy

#%%
def load_mrc(dp, box):
    if dp[-1]!="/": dp = dp + "/"
    name_list = [i for i in os.listdir(dp) if i.split(".")[-1]=="mrc"]
    name_list.sort()
    num = len(name_list)
    temp = np.zeros((num, box**3))
    for i, name in enumerate(name_list):
#        print(name)
        temp[i] = mrcfile.open(dp + name).data.reshape(-1)
    return (name_list, temp)

#%%
import networkx as nx
import matplotlib.pyplot as plt
#%%
def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current
        
def prune_DG(DG):
    ls_suc = []
    
    for i in list(DG.nodes):
        ls_suc += [list(DG.successors(i))]
    
    for t1, i in enumerate(DG.nodes):
        for t2, j in enumerate(DG.nodes):
            if j == i: continue
            if sum(np.isin(np.array(ls_suc[t1]+[i]), np.array(ls_suc[t2])))==len(ls_suc[t1])+1:
                for k in ls_suc[t1]:
                    ls_suc[t2].remove(k)
        #    print(ls_suc)
    DG_prune = nx.DiGraph()           
    for i, j in enumerate(ls_suc):
        if len(j)>0:
            for k in j:
                DG_prune.add_edge(list(DG.nodes)[i],k)
    return DG_prune

def subgraph(H):
    sink_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
    source_nodes = [node for node in H.nodes if H.in_degree(node) == 0]
    sub_list = []
    for source in source_nodes:
        for sink in sink_nodes:
            path = nx.all_simple_paths(H, source, sink)
            nodes = []
            for p in path:
                nodes  = nodes + p
            nodes = set(nodes)
            sub_list += [H.subgraph(nodes)]
    return sub_list
#%%     
def hierarchy_pos(G, root, levels=None, width=4., height=4.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

def draw_hierarchical(H, root="core"):
    pos = hierarchy_pos(H, root)
    #    plt.xlim([-1,2])
    plt.figure(figsize=(10, 10)) 
    nx.draw_networkx(H,pos,with_labels = True, edge_color='black', width = 1,  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black',boxstyle='round,pad=0.2'))


def solve_uull(u1, u2, l1, l2, i, j):
    t00 = np.logical_and(l1==0, l2==0).sum()
    t01 = np.logical_and(l1==0, u2==1).sum()
    t10 = np.logical_and(u1==1, l2==0).sum()
    t11 = np.logical_and(u1==1, u2==1).sum()
    return((i,j,t00,t01,t10,t11))

#%%
def contact_prune_DG(contact_G, DG):
    contact = set(contact_G.edges())
    contact = set.union(contact, (set(map(lambda x: x[::-1], contact_G.edges()))))
    DG_temp = nx.DiGraph()
    for i in DG.edges():
        if i in contact:
            DG_temp.add_edge(i[0],i[1])
    return DG_temp

#%%
def prune_DG2(DG):
    DG_temp = list(DG.edges())
    source = [node for node in DG if DG.in_degree(node)==0][0]
    for i in DG[source].keys():
        if DG.in_degree(i)>1:
            DG_temp.remove((source, i))
 #   print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx

#%%
def prune_DG3(DG):
    DG_temp = list(DG.edges())
    for i in DG.nodes():
        for j in DG.successors(i):
            for k in set(DG.successors(i)) & set(DG.successors(j)):
                try: DG_temp.remove((i,k))
                except: continue
    
#    print(DG_temp)
    xx = nx.DiGraph()
    xx.add_edges_from(DG_temp)
    return xx
#%%

all_seg_name, all_seg = load_mrc("/Users/shengkai/Dropbox (Scripps Research)/pna/all_seg", 160)
#%%
pna_name, pna_mrc = load_mrc('/Users/shengkai/Dropbox (Scripps Research)/pna/maps/scaled_adjusted_rename_map/', 160)
#%%
pna_mrc_bi = pna_mrc > np.std(pna_mrc) * 4.5
pna_mrc_bi = pna_mrc_bi * 1

#%%
pna_name_new, pna_mrc_new = load_mrc('/Users/shengkai/Dropbox (Scripps Research)/Williamson Lab/Paper_Kai/PNA/emdb_upload/map', 160)
pna_name = np.array([i.split(".")[0] for i in pna_name])

#%%
pna_mrc_new_bi = pna_mrc_new > np.std(pna_mrc_new) * 4.5
pna_mrc_new_bi = pna_mrc_new_bi * 1


#%%
all_seg = all_seg>0.1
all_seg = all_seg * 1
#%%
all_seg_name = np.array([i.split(".")[0] for i in all_seg_name])

pna_name_new = np.array([i.split(".")[0] for i in pna_name_new])

#%%

tr_all_seg = pd.read_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/tr_all_seg.csv")


#%%

occ = np.zeros((tr_all_seg.seg.values.shape[0],pna_mrc_new_bi.shape[0]))



#%%

h2_occ = np.matmul(pna_mrc_new_bi, all_seg[np.where(all_seg_name == "h2")[0]].T)/all_seg[np.where(all_seg_name == "h2")].sum()
uL24_occ = np.matmul(pna_mrc_new_bi, all_seg[np.where(all_seg_name == "uL24")[0]].T)/all_seg[np.where(all_seg_name == "uL24")].sum()


#%%
occ_raw = np.zeros((158,45))

for i, seg_name in enumerate(tr_all_seg.seg):
    print(seg_name)
    temp_seg = all_seg[np.where(all_seg_name == seg_name)[0]]
    temp_occ = np.matmul(pna_mrc_new_bi, temp_seg.T)/temp_seg.sum()
    if seg_name[0] == "h":
        temp_occ = temp_occ/h2_occ
    elif seg_name[1] == "L":
        temp_occ = temp_occ/uL24_occ
    
    occ_raw[i] = temp_occ.reshape(1,-1)
  #  bi_occ_temp = temp_occ >  float(tr_all_seg.threshold.values[i])
 #   bi_occ_temp = bi_occ_temp * 1
#    occ[i] = bi_occ_temp.reshape(-1)
#%%
occ_raw = np.zeros((158,45))

for i, seg_name in enumerate(tr_all_seg.seg):
    print(seg_name)
    temp_seg = all_seg[np.where(all_seg_name == seg_name)[0]]
    temp_occ = np.matmul(pna_mrc_new_bi, temp_seg.T)/temp_seg.sum()
    if seg_name[0] == "h":
        temp_occ = temp_occ/h2_occ
    elif seg_name[1] == "L":
        temp_occ = temp_occ/uL24_occ
    occ_raw[i] = temp_occ.reshape(1,-1)
    bi_occ_temp = temp_occ >  float(tr_all_seg.new_tr.values[i])
    bi_occ_temp = bi_occ_temp * 1
    occ[i] = bi_occ_temp.reshape(-1)
#%%
#df = pd.DataFrame(occ.T, columns = tr_all_seg.seg.values, index = pna_name)

#%%
#df_occtest = pd.DataFrame(occ.T, columns = tr_all_seg.seg.values, index = pna_name)
#%%
df_raw = pd.DataFrame(occ_raw.T, columns = tr_all_seg.seg.values, index = pna_name_new)

df_raw.to_csv("/Users/shengkai/Dropbox (Scripps Research)/Williamson Lab/Paper_Kai/PNA/supplementary_table/raw/raw_occ.csv")
#%%
#df = df.drop("RA20-B-c1").drop("RA20-B-c2")
df_raw = df_raw.drop("RA20-B-c1").drop("RA20-B-c2")
#%%
#df = df.drop("RA20-B-c1_nu").drop("RA20-B-c2_nu").drop("RB2860-J1_nu")
df_raw = df_raw.drop("RA20-B-c1_nu").drop("RA20-B-c2_nu").drop("RB2860-J1_nu")

#%%
# Melting the dataframes
df_melted = df.reset_index().melt(id_vars='index', var_name='seg', value_name='bi')
df_raw_melted = df_raw.reset_index().melt(id_vars='index', var_name='seg', value_name='raw')

# Merging the melted dataframes
df_merged = pd.merge(df_melted, df_raw_melted, on=['index', 'seg'])

#%%
"""
max0 = np.zeros(df_raw.shape[1])
min1 = np.ones(df_raw.shape[1])

for i, temp_seg in enumerate(np.unique(df_melted.seg)):
    df_temp = df_merged[df_merged.seg == temp_seg]
    if len(df_temp.raw[df_temp.bi == 0])!=0:
        max0[i] = df_temp.raw[df_temp.bi == 0].max()
    if len(df_temp.raw[df_temp.bi == 1])!=0:
        min1[i] = df_temp.raw[df_temp.bi == 1].min()
#%%
tr = (min1 + max0)/2
order = df.values.sum(0)

df_order = pd.DataFrame({"tr":tr,"order":order})


#%%
sorted_df_order = df_order.sort_values(by=['order', 'tr'])

combined_order = sorted_df_order.index
"""
#%%



# Plotting violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='seg', y='raw', hue='bi',
               data=df_merged[df_merged.seg.isin(tr_all_seg.seg[order.argsort()][0:10].values)], split=True,
               inner="point",bw_adjust=1, density_norm="count")
plt.title('Violin Plot for 0 and 1 Classes')
plt.xlabel('Column')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()
#%%
# Plotting violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='seg', y='raw', hue='bi',
               data=df_merged[df_merged.seg== "bL36"], split=True,
               inner="box", gap = 0.1, density_norm="width", bw_method = "silverman")
plt.title('Violin Plot for 0 and 1 Classes')
plt.xlabel('Column')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()
#%%
# Plotting violin plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='seg', y='raw', hue='bi',
               data=df_merged, order = df.columns[order.argsort()],
               dodge=True)
plt.title('Violin Plot for 0 and 1 Classes')
plt.xlabel('Column')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.show()

#%%
# Plotting violin plot
plt.figure(figsize=(12, 6))
sns.stripplot(x='seg', y='raw', hue='bi',
               data=df_merged, order = df.columns[order.argsort()],
               dodge=True)
plt.title('Violin Plot for 0 and 1 Classes')
plt.xlabel('Column')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.show()

#%%
# Plotting violin plot
plt.figure(figsize=(12, 6))
sns.stripplot(x='seg', y='raw', hue='bi',
               data=df_merged, order = df.columns[combined_order],
               dodge=True)
plt.title('Violin Plot for 0 and 1 Classes')
plt.xlabel('Column')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.show()
#%%
df.to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/all_seg_occ_bi.csv")
#%%
df = pd.read_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/all_seg_occ_bi.csv", index_col = 0)
df = df.drop("RA20-B-c1").drop("RA20-B-c2").drop("RB2860-J1")
#%%

c = sns.clustermap(df)

#%%
clusters = hierarchy.cut_tree(c.dendrogram_col.linkage, height = 0.1)

#%%

for i in np.unique(clusters):
    print(tr_all_seg.seg.values[np.where(clusters == i)[0]])

#%%
block_dict = {"block_"+ tr_all_seg.seg.values[np.where(clusters == i)[0]][0]:tr_all_seg.seg.values[np.where(clusters == i)[0]] for i in np.unique(clusters)}
#%%

contact_df = pd.read_csv('/Users/shengkai/Dropbox (Scripps Research)/pna/contact_matrix.csv', index_col = 0, header = 0)

contact_df_bi = (contact_df>1)*1

#%%
contact_df_bi.to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/contact_df_bi.csv")
#%%

def get_contact_seg(contact_df, seg1, seg2):
    if contact_df[seg1][seg2]!=0:
        return (1,"%s_%s"%(seg1, seg2))
    else:
        return (0,"no contact")

def get_contact_blocks(contact_df, block1, block2):
    contact_cont = 0
    contact = []
    for i in block1:
        for j in block2:
            #print(i)
            #print(j)
            contact_num, contact_temp = get_contact_seg(contact_df, i, j)
            #print(contact_num)
            #print(contact_temp)
            contact_cont = contact_cont + contact_num
            if contact_num != 0:
                contact += [contact_temp]
    return (contact_cont, contact)

#%%

block_contact_mat = np.zeros((len(block_dict),len(block_dict)))


#%%
for i, i_name in enumerate(block_dict.keys()):
    for j, j_name in enumerate(block_dict.keys()):
        if i < j: continue  
        block_contact_mat[i,j],_ = get_contact_blocks(contact_df, block_dict[i_name], block_dict[j_name])
        block_contact_mat[j,i] = block_contact_mat[i,j]


#%%
df_occ = df.T

#%%
block_list = np.array([i for i in block_dict.keys()])
df_occ["block"] = np.array([block_list[i][0] for i in clusters])

#%%
block_contact_df = pd.DataFrame(block_contact_mat, columns = block_list, index = block_list)            
#%%

df_occ_block = df_occ.groupby("block").mean()

#%%
s_upper = df_occ_block.T.values
s_lower = df_occ_block.T.values

seg_num = df_occ_block.shape[0]

qua = np.zeros((int(seg_num*(seg_num-1)/2),6))
count = 0
for i in range(seg_num-1):
    for j in range(i+1, seg_num):
        qua[count] = solve_uull(s_upper[:,i],s_upper[:,j],s_lower[:,i],s_lower[:,j], i,j)
        count += 1
    dependency = qua[np.logical_and(qua[:,3]==0, qua[:,4]>0)][:,0:2] 
    dependency = np.vstack((dependency,qua[np.logical_and(qua[:,4]==0, qua[:,3]>0)][:,0:2][:,::-1]))
    cor = qua[np.logical_and(qua[:,3]==0, qua[:,4]==0)][:,0:2]

DG=nx.DiGraph()
DG.add_edges_from(dependency.astype(int))
DG_prune = prune_DG(DG)
#%%
plt.figure()
pos = nx.kamada_kawai_layout(DG_prune)  # Compute layout
nx.draw_networkx(DG_prune, pos, with_labels=True)
plt.show()  # Don't forget to show the plot

#%%
contact = np.zeros((int(block_contact_mat.shape[0]*(block_contact_mat.shape[0]-1)/2),3))
count = 0
for i in range(block_contact_mat.shape[0]-1):
    for j in range(i+1, block_contact_mat.shape[0]):
       #conif i > j: continue
        contact[count] = [i, j, block_contact_mat[i,j]]
        count += 1
        
GG=nx.Graph()
GG.add_weighted_edges_from(contact[contact[:,2]!=0])
mapper = {i:j.split("_")[-1] for i,j in enumerate(block_list)}


plt.figure()
nx.draw_networkx(GG, with_labels = True)

#%%
DG_prune_contact = contact_prune_DG(GG,DG)
#%%
plt.figure()
nx.draw_networkx(nx.relabel_nodes(DG_prune_contact,mapper), with_labels = True)
#%%
H = nx.relabel_nodes(prune_DG3(prune_DG2(prune_DG(DG_prune_contact))),mapper)
#%%
s_H = subgraph(H)

#%%
for i, H_temp in enumerate(s_H):
   # plt.figure()
    draw_hierarchical(H_temp, "h18")
    plt.savefig("sub_%i.png"%i)
    plt.close()
    
#%%
block_csv = pd.DataFrame(block_list)
block_csv["content"] = [block_dict[i] for i in block_list]

block_csv["name"] = [i.split("_")[-1] for i in block_list]
block_csv.to_csv("block_list.csv")

#%%
block_csv = pd.read_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/sub_dependency/block_list_ann_2.csv", header = 0, index_col = 0)

mask_nodes = block_csv.name[~block_csv.domain.isin([0, 1, 2, 3, 6])].values

domain1236_G = H.copy()
domain1236_G.remove_nodes_from(mask_nodes)


#%%
list_to_remove = ["uL14", "bL27", "h67", "bL9", "bL25", "h38t", "h69t", "uL5"]

domain1236_G_remove = domain1236_G.copy()
domain1236_G_remove.remove_nodes_from(list_to_remove)

list2_to_remove = ["h72b", "h73b", "h73t", "h102t", "h42", "h43", "h31", "uL15",
                   "bL33", "h89t"]
domain1236_G_remove.remove_nodes_from(list2_to_remove)
#%%
d_s =subgraph(domain1236_G_remove)

#%%
for i, d_temp in enumerate(d_s):
   # plt.figure()
    draw_hierarchical(d_temp, "h18")
    plt.savefig("sub_D_%i.png"%i)
    plt.close()

#%%
domain_color_map = {
    0: 'grey',
    1: 'red',
    2: 'orange',
    3: 'yellow',
    5: 'lightblue',
    6: 'green'
}
#%%
domain1236_domain = [block_csv.domain[np.where(block_csv.name == i)[0]].values[0] for i in domain1236_G_remove.nodes]

domain1236_node_color = [domain_color_map[i] for i in domain1236_domain]
#%%

pos = hierarchy_pos(domain1236_G_remove, "h18")

plt.figure(figsize=(10, 10)) 
nx.draw_networkx(domain1236_G_remove,pos,with_labels = True, edge_color='black', width = 1,  node_color=domain1236_node_color)


#%%
import grandalf
from grandalf.layouts import SugiyamaLayout


g = grandalf.utils.convert_nextworkx_graph_to_grandalf(domain1236_G_remove)  # undocumented function

class defaultview(object): # see README of grandalf's github
    w, h = 10, 10


for v in g.C[0].sV:
    v.view = defaultview()

sug = SugiyamaLayout(g.C[0])
sug.init_all() # roots=[V[0]])
sug.draw()
# This is a bit of a misnomer, as grandalf doesn't actually come with any visualization methods.
# This method instead calculates positions

poses = {v.data: (v.view.xy[0], v.view.xy[1]) for v in g.C[0].sV} # Extracts the positions
nx.draw_networkx(domain1236_G_remove, pos=poses, with_labels=True,node_color=domain1236_node_color)

plt.show()


#%%
poses_inverted_y = {node: (x, -y) for node, (x, y) in poses.items()}
plt.figure()
nx.draw_networkx(domain1236_G_remove, pos=poses_inverted_y, with_labels=True,node_color=domain1236_node_color)



#%%
poses_inverted_y = {node: (x, -y) for node, (x, y) in poses.items()}

for i in [13,12,22,10,6,21]:
    poses_inverted_y["h%i"%i] = poses_inverted_y["h%i"%i] * np.array([-1,1])

for i in ["h27","bL20","h26","h32b","h32s","h33b","h102t","h72b"]:
    poses_inverted_y[i] = poses_inverted_y[i] + np.array([50,0])

poses_inverted_y["bL34"] = poses_inverted_y["bL34"] * np.array([-1,1]) + np.array([-80,0])
poses_inverted_y["h27"] = poses_inverted_y["h27"]  + np.array([70,0])
poses_inverted_y["h23"] = poses_inverted_y["h23"]  + np.array([-100,0])

plt.figure()
nx.draw_networkx(domain1236_G_remove, pos=poses_inverted_y, with_labels=True,node_color=domain1236_node_color)

#%%
def get_contact_in_block(contact_df, block):
    G_temp = nx.DiGraph()
    num = len(block)
    edge_list = []
    for i in range(num-1):
        for j in range(i+1,num):
            temp_num, temp_contact = get_contact_seg(contact_df, block[i], block[j])
            if temp_num != 0:
                edge_list += [temp_contact.split("_")]
             #   print(edge_list)
            #    print(temp_contact)
    G_temp.add_edges_from(edge_list)
    return G_temp
            
#%%
aa = get_contact_in_block(contact_df, block_dict["block_bL17"])
g = grandalf.utils.convert_nextworkx_graph_to_grandalf(aa)  # undocumented function

class defaultview(object): # see README of grandalf's github
    w, h = 10, 10


for v in g.C[0].sV:
    v.view = defaultview()

sug = SugiyamaLayout(g.C[0])
sug.init_all() # roots=[V[0]])
sug.draw()
# This is a bit of a misnomer, as grandalf doesn't actually come with any visualization methods.
# This method instead calculates positions



plt.figure()
poses = {v.data: (v.view.xy[0], v.view.xy[1]) for v in g.C[0].sV} # Extracts the positions
nx.draw_networkx(aa, pos=poses, with_labels=True)

plt.show()



#%%

h2_occ = np.matmul(pna_mrc_new_bi, all_seg[np.where(all_seg_name == "h2")[0]].T)/all_seg[np.where(all_seg_name == "h2")].sum().reshape(1,-1)
uL24_occ = np.matmul(pna_mrc_new_bi, all_seg[np.where(all_seg_name == "uL24")[0]].T)/all_seg[np.where(all_seg_name == "uL24")].sum().reshape(1,-1)

#%%
tr_all_seg = pd.read_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/tr_all_seg.csv")
print(np.where(tr_all_seg.seg.values == "h69b"))

#%%
for i, seg_name in [(100,"h69b")]:#enumerate(tr_all_seg.seg.values):
 #   print(seg_name)
    tr_temp = tr_all_seg.new_tr[i]
    temp_seg = all_seg[np.where(all_seg_name == seg_name)[0]].reshape(-1)
    temp_occ_arr = np.matmul(pna_mrc_new_bi, temp_seg.T)/temp_seg.sum()
    if seg_name[0] == "h":
        temp_occ_arr = temp_occ_arr/h2_occ.reshape(1,-1)
    elif seg_name[1] == "L":
        temp_occ_arr = temp_occ_arr/uL24_occ.reshape(1,-1)
   
    temp_occ_arr = temp_occ_arr.reshape(-1)
    fig_name = seg_name + "_occ.png"
    fig, axs = plt.subplots(7, 7, figsize=(10, 10))

    axs = axs.flatten()
    
    seg_x, seg_y, seg_z = mass_center(temp_seg.reshape((160,160,160)))
    
    vmax_temp = temp_seg.sum() ** (1/3)
    
    for k, order in enumerate(temp_occ_arr.argsort()):
   #     print(k)
   #     print(order)
        temp_pna_name = pna_name_new[order]
        temp_occ = temp_occ_arr[order]
        temp_mrc = np.zeros(160**3)
        pna_temp = pna_mrc_new_bi[order].reshape(-1)
        temp_mrc[temp_seg ==1] = pna_temp[temp_seg ==1]
        temp_mrc = temp_mrc.reshape((160,160,160))
        box_mrc = temp_mrc[seg_x-25:seg_x+25,seg_y-25:seg_y+25,seg_z-25:seg_z+25]
        axs[k].imshow(box_mrc.sum(0), vmax = vmax_temp, vmin = 0)
        if temp_occ >= tr_temp:
            axs[k].text(0.5, 0.8, '%s \n %2f'%(temp_pna_name,temp_occ), color='white', fontsize=8,
                ha='center', va='center', transform=axs[k].transAxes)
        if temp_occ < tr_temp:
            axs[k].text(0.5, 0.8, '%s \n %2f'%(temp_pna_name,temp_occ), color='red', fontsize=8,
                ha='center', va='center', transform=axs[k].transAxes)
        axs[k].axis('off')  # Turn off axis labels for better visualization
    #axs[47].hist()
    axs[45].axis('off')
    axs[46].axis('off')
    axs[47].axis('off')
    axs[48].axis('off')
    
    plt.tight_layout()
    fig.suptitle(seg_name, fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()
    plt.savefig('/Users/shengkai/Dropbox (Scripps Research)/pna/nu_refine_occ/' + fig_name)
    plt.close()

#%%
df_test = (df_raw > tr_all_seg.new_tr.values)*1


occ_dict = {0: "Not Occupied", 1: "Occupied"}
#%%
# Melting the dataframes
df_test_melted = df_test.reset_index().melt(id_vars='index', var_name='seg', value_name='bi')
df_raw_melted = df_raw.reset_index().melt(id_vars='index', var_name='seg', value_name='raw')

# Merging the melted dataframes
df_merged = pd.merge(df_test_melted, df_raw_melted, on=['index', 'seg'])
df_merged["occ"] = [occ_dict[i] for i in df_merged.bi]

#%%
df_merged.to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/nu_refine_occ/occ_stripplot.csv")
#%%
df_test.to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/nu_refine_occ/binarized_occ.csv")
df_raw.to_csv("/Users/shengkai/Dropbox (Scripps Research)/pna/nu_refine_occ/raw_occ.csv")
#%%

max0 = np.zeros(df_raw.shape[1])
min1 = np.ones(df_raw.shape[1])

for i, temp_seg in enumerate(np.unique(df_melted.seg)):
    df_temp = df_merged[df_merged.seg == temp_seg]
    if len(df_temp.raw[df_temp.bi == 0])!=0:
        max0[i] = df_temp.raw[df_temp.bi == 0].max()
    if len(df_temp.raw[df_temp.bi == 1])!=0:
        min1[i] = df_temp.raw[df_temp.bi == 1].min()
#%%
tr = (min1 + max0)/2
order = df.values.sum(0)

df_order = pd.DataFrame({"tr":tr,"order":order})


#%%
sorted_df_order = df_order.sort_values(by=['order', 'tr'])

combined_order = sorted_df_order.index



#%%

plt.figure(figsize=(20, 6))
sns.stripplot(x='seg', y='raw', hue='occ',
               data=df_merged, order = df.columns[combined_order],
               dodge=True)

plt.xlabel('Structure Elements')
plt.ylabel('Mean electron density')
plt.xticks(rotation=90)
plt.show()

#%%


#%%
# Set the font size and font family
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

# Create the plot
plt.figure(figsize=(15, 6))
sns.stripplot(x='seg', y='raw', hue='occ',
              data=df_merged, order=df.columns[combined_order],
              dodge=True, palette={'Occupied': 'darkred', 'Not Occupied': 'darkblue'})

# Set ticks inside
plt.xlabel('Structure Elements', size = 14)
plt.ylabel('Mean electron density', size = 14, ha='right')
plt.tick_params(axis='x', direction='in', length=2)
plt.tick_params(axis='y', direction='in', length=2, right=True, left=False, labelright=True, labelleft=False)  # Put y-ticks on the right and remove ticks on the left
plt.xticks(rotation=90)
plt.yticks(rotation=90)

#%%
# Set the font size and font family
plt.rcParams.update({'font.size': 8, 'font.family': 'Arial'})

# Create the plot
plt.figure(figsize=(6,9))
sns.stripplot(x='raw', y='seg', hue='occ',
              data=df_merged, order=df.columns[combined_order[::-1]],
              dodge=True, palette={'Occupied': 'darkred', 'Not Occupied': 'darkblue'}
              )

# Set ticks inside
plt.xlabel('Structure Elements', size = 14)
plt.ylabel('Mean electron density', size = 14)
plt.xticks(rotation=90)
plt.tick_params(axis='x', direction='in', length=5)
plt.tick_params(axis='y', direction='inout', length=5, right=True, left=False, labelright=True, labelleft=False)  # Put y-ticks on the right and remove ticks on the left

#%%
# Set the font size and font family
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

plt.subplots_adjust(left=0.1, bottom=0.01, right=0.9, top=0.99, wspace=0.2, hspace=0.3)

# Create the plot
plt.figure(figsize=(6,9))
sns.stripplot(x='raw', y='seg', hue='occ',
              data=df_merged, order=df.columns[combined_order[::-1]],
              dodge=True, palette={'Occupied': 'darkred', 'Not Occupied': 'darkblue'}
              )

# Set ticks inside
plt.xlabel('Structure Elements', size = 14)
plt.ylabel('Mean electron density', size = 14)
plt.xticks(rotation=90)
plt.tick_params(axis='x', direction='in', length=5)
plt.tick_params(axis='y', direction='inout', length=5)

#%%

c = sns.clustermap(df_test)

#%%
clusters = hierarchy.cut_tree(c.dendrogram_col.linkage, height = 0.01)

#%%

for i in np.unique(clusters):
    print(tr_all_seg.seg.values[np.where(clusters == i)[0]])

#%%
bL34_1 = df_raw[df == 1]["bL34"].dropna().values
bL34_0 = df_raw[df == 0]["bL34"].dropna().values

h103_1 = df_raw[df == 1]["h103"].dropna().values
h103_0 = df_raw[df == 0]["h103"].dropna().values
#%%
plt.figure()
plt.hist(bL34_1, bins = 30)
plt.hist(bL34_0, bins = 30)

#%%
plt.figure()
plt.hist(h103_1, bins = 30)
plt.hist(h103_0, bins = 30)
#%%
plt.figure()
plt.scatter(df_raw["h34b"].values, df_raw["h33t"].values, s =2)


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Your observed data
data1 = bL34_1  # Replace [...] with your actual data for sample 1
data2 = bL34_0  # Replace [...] with your actual data for sample 2

# Fit Gaussian distributions to the data
mean1, std1 = norm.fit(data1)
mean2, std2 = norm.fit(data2)

# Determine the number of bins using the Freedman-Diaconis rule
bins1 = int(np.ceil((2 * len(data1))**(1/3)))
bins2 = int(np.ceil((2 * len(data2))**(1/3)))

plt.figure()
# Plot histograms of the data with the fitted Gaussian distributions
plt.hist(data1, bins=bins1, density=True, alpha=0.6, color='r', label='Sample 1')
plt.hist(data2, bins=bins2, density=True, alpha=0.6, color='b', label='Sample 2')

# Plot the fitted Gaussian distributions
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)


p1 = norm.pdf(x, mean1, std1)
plt.plot(x, p1, 'g', linewidth=2)

p2 = norm.pdf(x, mean2, std2)
plt.plot(x, p2, 'b', linewidth=2)

# Add labels and title
plt.xlabel('Mean electron density')
plt.ylabel('pdf')
#plt.title('Histogram of Data with Fitted Gaussian Distributions')

plt.show()

#%%

h103_1 = df_raw[df == 1]["h32b"].dropna().values
h103_0 = df_raw[df == 0]["h32b"].dropna().values

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Your observed data
data1 = h103_1  # Replace [...] with your actual data for sample 1
data2 = h103_0  # Replace [...] with your actual data for sample 2

# Fit Gaussian distributions to the data
mean1, std1 = norm.fit(data1)
mean2, std2 = norm.fit(data2)

# Determine the number of bins using the Freedman-Diaconis rule
bins1 = int(np.ceil((2 * len(data1))**(1/3)))
bins2 = int(np.ceil((2 * len(data2))**(1/3)))

plt.figure()
# Plot histograms of the data with the fitted Gaussian distributions
plt.hist(data1, bins=bins1, density=False, alpha=0.6, color='r', label='Sample 1')
plt.hist(data2, bins=bins2, density=False, alpha=0.6, color='b', label='Sample 2')

# Plot the fitted Gaussian distributions
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p1 = norm.pdf(x, mean1, std1)
plt.plot(x, p1 * len(data1) * (xmax - xmin) / bins1, 'r', linewidth=2)

p2 = norm.pdf(x, mean2, std2)
plt.plot(x, p2 * len(data2) * (xmax - xmin) / bins2, 'b', linewidth=2)

# Add labels and title
plt.xlabel('Mean electron density')
plt.ylabel('Counts')
#plt.title('Histogram of Data with Fitted Gaussian Distributions')

plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



x = df_raw["bL34"].values 
y = df_raw["h103"].values

bins1 = int(np.ceil((2 * len(x))**(1/3)))
bins2 = int(np.ceil((2 * len(y))**(1/3)))

fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    
ax_main.scatter(x,y,marker='.')
ax_main.set(xlabel="x data", ylabel="y data")

ax_xDist.hist(x,bins=bins1,align='mid')
ax_xDist.set(ylabel='count')
ax_xCumDist = ax_xDist.twinx()
ax_xCumDist.hist(x,bins=bins1,density=True,color='r',align='mid')
ax_xCumDist.tick_params('y', colors='r')
ax_xCumDist.set_ylabel('cumulative',color='r')

ax_yDist.hist(y,bins=bins2,orientation='horizontal',align='mid')
ax_yDist.set(xlabel='count')
ax_yCumDist = ax_yDist.twiny()
ax_yCumDist.hist(y,bins=bins2,density=True,color='r',align='mid',orientation='horizontal')
ax_yCumDist.tick_params('x', colors='r')
ax_yCumDist.set_xlabel('cumulative',color='r')

plt.show()
#%%
# Your observed data

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

x_data = df_raw["bL34"]
y_data = df_raw["h103"]

# Plot 2D KDE with customized appearance
ax = sns.kdeplot(x=x_data, y=y_data, cmap="Blues", fill=True, thresh=0.05, n_levels=10)
ax = plt.scatter(x=x_data, y=y_data,s = 10, color = "r")

plt.xlabel('Occupancy of bL34', fontsize=16)
plt.ylabel('Occupancy of h103', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(tr_all_seg[tr_all_seg.seg == "bL34"].threshold.values, linestyle='--', c = 'r')
plt.axhline(tr_all_seg[tr_all_seg.seg == "h103"].threshold.values, linestyle='--', c = 'r')

plt.grid(True, linestyle='--', alpha=0.5)

#plt.scatter(x,y, s = 20, alpha = 0.2, c = "black")


plt.show()

#%%
# Your observed data

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

x_data = df_raw["bL34"]
y_data = df_raw["h103"]

# Plot 2D KDE with customized appearance
ax = plt.scatter(x=x_data, y=y_data,s = 10)

plt.xlabel('Occupancy of bL34', fontsize=16)
plt.ylabel('Occupancy of h103', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.axvline(tr_all_seg[tr_all_seg.seg == "bL34"].threshold.values, linestyle='--', c = 'r')
plt.axhline(tr_all_seg[tr_all_seg.seg == "h103"].threshold.values, linestyle='--', c = 'r')

plt.grid(True, linestyle='--', alpha=0.5)

#plt.scatter(x,y, s = 20, alpha = 0.2, c = "black")


plt.show()

