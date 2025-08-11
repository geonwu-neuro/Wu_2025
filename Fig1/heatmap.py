import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.dpi'] = 600
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.autolayout'] = True

def sort_trace(df):
    seg_1 = df[(df.c_ext == True) & (df.l_ext == False) & (df.t_ext == False)].reset_index(drop = True)
    seg_2 = df[(df.c_ext == False) & (df.l_ext == True) & (df.t_ext == False)].reset_index(drop = True)
    seg_3 = df[(df.c_ext == True) & (df.l_ext == True) & (df.t_ext == False)].reset_index(drop = True)
    seg_4 = df[(df.c_ext == True) & (df.l_ext == True) & (df.t_ext == True)].reset_index(drop = True)
    seg_5 = df[(df.c_ext == True) & (df.l_ext == False) & (df.t_ext == True)].reset_index(drop = True)
    seg_6 = df[(df.c_ext == False) & (df.l_ext == True) & (df.t_ext == True)].reset_index(drop = True)
    seg_7 = df[(df.c_ext == False) & (df.l_ext == False) & (df.t_ext == True)].reset_index(drop = True)
    seg_8 = df[(df.c_ext == False) & (df.l_ext == False) & (df.t_ext == False) & (df.t_inh == True)].reset_index(drop = True)
    seg_9 = df[(df.c_ext == False) & (df.l_ext == False) & (df.t_ext == False)& (df.t_inh == False)].reset_index(drop = True)
    
    sorted_df = pd.DataFrame()

    num = 0
    seg_count = []
    for i in [seg_1, seg_2, seg_3,seg_4, seg_5,seg_6,seg_7,seg_8,seg_9]:
        sorted_df = pd.concat([sorted_df, i], axis = 0)
        num = num + len(i)
        seg_count.append(num)

    sorted_df = sorted_df.reset_index(drop = True)
    sorted_df = sorted_df.drop(columns = ['c_ext', 'l_ext','session','cell_id']).reset_index(drop = True)

    return sorted_df, seg_count


def convert_frame(trace):  # Change the column name from frame num to seconds 
    trace = trace.drop(columns = {'animalID', 'cs_id','t_ext','t_inh','aveT'}, axis = 1)
    time_count = list(np.arange(0, 644*0.03, 0.03))
    new_co = {}
    for i in range(len(trace.columns)):
        new_co[str(i)] = round(time_count[i],2)
    trace = trace.rename(columns = new_co)
    
    return trace

#
cst_df = pd.read_csv("filepath")
sorted_cst, seg = sort_trace(cst_df)
cst_trace = convert_frame(sorted_cst)
             
# Plot sorted heatmap 
fig, ax = plt.subplots(1,2, figsize=(5.5,5.5), gridspec_kw= {'width_ratios' : (1, 0.08)})
sns.heatmap(cst_trace, cmap = 'magma', cbar_kws = {r'label': '$\Delta$F/F (z-scores)'}, 
    cbar_ax = ax[1],  ax = ax[0], vmin = -5, vmax = 20)

cbar = ax[0].collections[0].colorbar
cbar.ax.yaxis.set_label_position('left')

ax[0].vlines([154,216,231,343] ,0,len(cst_trace), color = ['lightgrey','lightgrey','yellow','yellow'], 
             ls= '--', lw = 2)

ax[0].hlines(seg[-2], 0, len(cst_trace.columns),
             color = 'w', lw = 2)

ax[0].hlines(seg[2], 0, len(cst_trace.columns),
             color = 'w', lw = 2) 

ax[0].set_ylabel('cell ID', fontsize = 20)
ax[0].set_xlabel('Second', fontsize = 20)

# Set x-ticks
xticks = [ i for i in list(np.arange(0, 644, 60))]
ax[0].tick_params('x', rotation = 45)
ax[0].set_xticks(xticks, cst_trace.columns[::60]) # Reduce the number of x-ticks 

# Set y-ticks 
yticks = [ i for i in list(np.arange(0, len(cst_trace), 20))]
ax[0].set_yticks(yticks, cst_trace.index[::20])

plt.suptitle('CST', fontsize = 24, fontweight = 'bold', color= 'navy',  y= 0.95)    
plt.tight_layout()
plt.show()

# Print the total cell number for each session
print(f'{len(cst_trace)} cells')
    






