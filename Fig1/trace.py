import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def getBoundedMean(data):
    mean = data.mean(axis = 0)
    se = data.std(axis = 0) / np.sqrt(len(data))
    
    return mean, mean-se, mean+se


cst1_df_wt = pd.read_csv(filepath_wt)
sorted_cst1_wt, seg1 = sort_trace(cst1_df_wt)
cst1_df_ko = pd.read_csv(filepath_ko)
sorted_cst1_ko, seg1 = sort_trace(cst1_df_ko)

# Plot overlayed trial_averaged traces activated across animals
# If plotting suppressed traces, set ["t_inh"] == True 
wt1 = sorted_cst1_wt[sorted_cst1_wt['t_ext'] == True].reset_index(drop = True)
ko1 = sorted_cst1_ko[sorted_cst1_ko['t_ext'] == True].reset_index(drop = True)


wt1_trace = wt1.drop(columns = ['animalID', 'cs_id','t_ext', 't_inh', 'aveT'])
time_count = list(np.arange(0, 644*0.03, 0.03))
new_co = {}
for i in range(len(wt1_trace.columns)):
    new_co[str(i)] = round(time_count[i],2)

wt1_trace = wt1_trace.rename(columns = new_co)
wt1_mean, wt1_lower, wt1_upper = getBoundedMean(wt1_trace)

ko1_trace = ko1.drop(columns = ['animalID', 'cs_id', 't_ext', 't_inh', 'aveT'])
ko1_trace = ko1_trace.rename(columns = new_co)
ko1_mean, ko1_lower, ko1_upper = getBoundedMean(ko1_trace)



fig, ax = plt.subplots(figsize = (2.5,2.5))
ax.plot(time_count, wt1_mean, color = 'navy')
ax.fill_between(time_count, wt1_lower, wt1_upper, color = 'lightgray')
ax.plot(time_count, ko1_mean, color = 'darkgoldenrod')
ax.fill_between(time_count, ko1_lower, ko1_upper, color = 'lightgray')
ax.set_title('CST1', fontsize = 20, y = 1.05)


ax.set_ylim(-2,8)      # If plot suppressed traces, set at(-4, 1)
ax.set_xlabel('Second')
ax.set_ylabel('$\Delta$F/F (z scores)')
ax.axvline(round(155*0.03, 2), ls = '--', color = 'darkgray')
ax.axvline(round(217*0.03, 2), ls = '--', color = 'darkgray')
ax.axvline(round(232*0.03, 2), ls = '--', color = 'orange')
ax.axvline(round(344*0.03, 2), ls = '--', color = 'orange')
ax.set_xticks(np.linspace(0,20,5))  
   
     
plt.tight_layout()
plt.ioff()
plt.show()
