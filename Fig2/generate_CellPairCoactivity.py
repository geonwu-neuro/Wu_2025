import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
import itertools

def corr_cellpair(data):
    # make sure cell_id are column names
    correlation = {}
    columns = data.columns.tolist()
    
    for col_a, col_b in itertools.combinations(columns, 2):
        correlation[f'{col_a}_{col_b}'] = pearsonr(data.loc[:, col_a], data.loc[:, col_b])
                
    result = pd.DataFrame.from_dict(correlation, orient = 'index')
    result.columns = ['PCC', 'p-value']
    
    return result['PCC'].to_frame()

def distance_cellpair(data):
    distance = {}
    columns = data.columns.tolist()
    
    for col_a, col_b in itertools.combinations(columns, 2):
        a = np.array(data[col_a])
        b = np.array(data[col_b])
        distance[f'{col_a}_{col_b}'] = np.linalg.norm(a-b)
                
    result = pd.DataFrame.from_dict(distance, orient = 'index')
    result.columns = ['distance']
    
    return result

# Function to extract trial-by-trial taste-evoked values from each session 
def extract_TrialbyTrial_value(folder):
    for roots, dirs, files in os.walk(folder):
        for file in files:
            cond_list = ['cst1','cst2']
            for session in cond_list:
                if f'{session}_activity_ID_byTrial' in file:
                    print(roots)
                    animal = roots.split('\\')[-2]
                    df = np.load(os.path.join(roots, file), allow_pickle = True).item()
                    trial_order = sorted(df.keys())
                    id_list = df[trial_order[0]].cell_id
                    combined = pd.DataFrame(columns = trial_order, index = id_list)
                    combined_bs = pd.DataFrame(columns = trial_order, index = id_list)
                    for trial in trial_order:
                        ind_df = df[trial].copy()
                        for cell in ind_df.cell_id:
                            combined.loc[cell, trial] = ind_df.loc[ind_df.cell_id == cell, 'z_mean_taste'].values[0]
                            combined_bs.loc[cell, trial] = ind_df.loc[ind_df.cell_id == cell, 'z_mean_baseline'].values[0]
                    
                    combined2 = combined.copy().T # change cell_id to columns, making it easier to calculate pairwise correlation later 
                    combined2_bs= combined_bs.copy().T
                    
                    combined2.to_csv(os.path.join(roots, f'{animal}_{session}_trialbytrial_aveT2.csv'), index = False)
                    combined2_bs.to_csv(os.path.join(roots, f'{animal}_{session}_trialbytrial_bs_aveT2.csv'), index = False)
#               
def shuffle_data(data):
    df = pd.DataFrame()
    for column in data.columns:
        
        df[column] = data[column].sample(frac = 1, replace = False).values
    return df            
#
def pooled_cst_corr(folder, animal_list, group, session, shuffled = False):
    
    outputfolder = r'G:\{folder}\pooled_cst_data'
    for i in ['trialbytrial_aveT2', 'trialbytrial_bs_aveT2']:
        cst_df = pd.DataFrame()
        cst_df_bs = pd.DataFrame()
        
        for roots, dirs, files in os.walk(folder):
            for file in [f for f in files if (i in f) and ('act' not in f)]:
                print(file)
                animal = file.split('_')[0]
                sessionID = file.split('_')[1]
                if (animal in animal_list) and (sessionID == session):
                    df = pd.read_csv(os.path.join(roots, file))
                    if shuffled == True:
                        df = shuffle_data(df)
                    else:
                        pass
                    corr_df = corr_cellpair(df)
                    corr_df = corr_df.reset_index()
                    corr_df = corr_df.rename(columns = {'index': 'cell_pair'})
                    corr_df['animalID'] = animal
                    corr_df['session'] = session
                    
                    # Get the distance beween cell pairs 
                    try:
                        center_df = pd.read_csv(fr"G:\2P_outputdata\{animal}\{session}\cell_center.csv")
                        distance_df = distance_cellpair(center_df.T)
                        distance_df = distance_df.reset_index()
                        distance_df = distance_df.rename(columns = {'index': 'cell_pair'})
                        # Merge the corr and distance matrix
                        corr_df = pd.merge(corr_df, distance_df, on = ['cell_pair'])
                        print(f'pulling cell center for {animal} {session}')
                    except:
                        pass
                    
                    if i == 'trialbytrial_aveT2':
                        cst_df = pd.concat([cst_df, corr_df], axis = 0)
                    elif i == 'trialbytrial_bs_aveT2':
                        cst_df_bs = pd.concat([cst_df_bs, corr_df], axis = 0)    
             
        if shuffled == True:
            cst_df.to_csv(fr'{outputfolder}\{group}_cst_avecorr_sh.csv', index = False)
            cst_df_bs.to_csv(fr'{outputfolder}\{group}_cst_bs_avecorr_sh.csv', index = False)
            
        else:
            cst_df.to_csv(fr'{outputfolder}\{group}_cst_avecorr.csv', index = False)
            cst_df_bs.to_csv(fr'{outputfolder}\{group}_cst_bs_avecorr.csv', index = False)
     
#
WT_list = ['CWP16','CWP17', 'CWP20', 'CWP27','CWP28', 'CWP29', 'CWP39','CWP47']
KO_list = ['CWShkP7', 'CWShkP8', 'CWShkP12', 'CWShkP14', 'CWShkP16', 'CWShkP18', 'CWShkP26']
folder = ''
session = ''
extract_TrialbyTrial_value(folder)
pooled_cst_corr(folder, WT_list, group = 'CTAWT', session = session)
pooled_cst_corr(folder, KO_list, group = 'CTAKO', sessio = session)
