"""
Created on Tue Jun 11 16:14:31 2024
This script pools and generates average evoked trace and concatenated "noise" trace for cells
registered across sessions
@author: chwu
"""

import numpy as np
import pandas as pd


def get_cValTrace(folder, animal, session, subtract = True):
    print(f'now do {animal}')
    aveDelta = pd.read_csv(fr"{folder}\{animal}\{session}\{animal}_{session}_TrialAverageDelta.csv")
    aveDelta = aveDelta.drop(columns={aveDelta.columns[0]})
    # change the column type from string to int
    for i, p in enumerate(list(aveDelta.columns)):
        aveDelta = aveDelta.rename(columns = {p:i})
    dat = np.load(fr"{folder}\{animal}\{session}\{animal}_{session}_deltaF_trace_byTrial.npy", 
                  allow_pickle = True).item() 
    
    concat_df = pd.DataFrame()
    for trial in dat.keys():
        print(f'processing - trial {trial}')
        df = dat[trial].copy()
        df = df.drop(columns = {'cell_id'}) # this file includes 644 frames
        if len(df.columns) < 644:
            print(f'{trial} - failed')
            continue
        else:
            if subtract == True:
                minus = df - aveDelta
            elif subtract == False:
                minus = df.copy()
            
            minus = minus.iloc[:, 232:345 ] # 232-345 is the whole 3 seconds of taste delivery            
            concat_df = pd.concat([concat_df, minus], axis = 1) # concat trials together 
    
    # reset column names so it doen't repeat every 114 frames 
    concat_df = concat_df.T.reset_index(drop = True).T
    
    return concat_df

def pooledRegTrace(folder, session,  animal_list, trace_type,  reg_type):
    # This function get cells that can be identified in all test sessions, 
    # containing cellstats and activity percentage
    data = pd.DataFrame()
    data2 = pd.DataFrame()
    #animalID = 'NA'
    for animal in animal_list:
        print(animal)
        
        # Get cross-session cell id
        reg_session = ['cst1','cst2','licl1','licl2','test1', 'test2', 'test3','test4', 'test5']
        footprint_id = np.load(fr'{folder}\{animal}\footprint\cs1_us1_cs2_us2_test\align_index.npy')
        
        # In the previous script, the reg session is test only
        # reg_session = ['test1', 'test2', 'test3','test4', 'test5']
        # footprint_id = np.load(fr'{folder}\{animal}\footprint\test_only\align_index.npy')
        
        footprint_dat = pd.DataFrame(footprint_id, columns = reg_session)
        
        if reg_type == 'nonreg':
            overlap = footprint_dat[ (footprint_dat[session] != -1)].dropna().reset_index(drop = True)
        
        elif reg_type == 'reg':
            
            # use the cells that can by tracked throughout all the test sessions
            overlap = footprint_dat[ (footprint_dat['cst1'] != -1) & (footprint_dat['cst2'] != -1)].dropna().reset_index(drop = True)
        
                   
        overlap = overlap.astype(int)
        
        for i in range(len(overlap)):
            overlap.loc[i, 'cs_id'] = f'{animal}_cs_{i}'
        
        # Remap cell_id to the trial-average delta trace file    
        if trace_type == 'signal':
            df = pd.read_csv(fr"{folder}\{animal}\{session}\{animal}_{session}_TrialAverageDelta.csv")
            df = df.drop(columns={df.columns[0]})
            df = df.iloc[:, 232:345]# 232-345 is the whole 3 seconds of taste delivery
            df['animalID'] = animal
            df['session'] = session
            
        
        elif trace_type == 'noise':
            # for the indicated animal in the session, get the concatenated variability traces 
            df = get_cValTrace(animal, session, subtract = True)
            df['animalID'] = animal
            df['session'] = session
            
        
        elif trace_type == 'concat':
            # for the indicated animal in the session, get the concatenated variability traces 
            df = get_cValTrace(animal, session, subtract = False)
            df['animalID'] = animal
            df['session'] = session
            
        for i in df.index:
            if i in overlap[session].values:
                df.loc[i, 'cs_id'] = overlap.loc[overlap[session] == i, 'cs_id' ].values[0]
                
        df.dropna(inplace = True)
        df.reset_index(drop = True, inplace = True)
        
        data = pd.concat([data, df], axis = 0)
        
        # Remap cell_id to the cellstats file 
        ind_dat = pd.DataFrame()
        df2 = pd.read_csv(fr"{folder}\{animal}\{session}\{animal}_{session}_cellstats.csv")
        for i in df2.index:
            if i in overlap[session].values:
                df2.loc[i, 'cs_id'] = overlap.loc[overlap[session] == i, 'cs_id' ].values[0]
            
        df2.dropna(inplace = True)
        df2.reset_index(drop = True, inplace = True)
        df2.rename(columns = {df2.columns[0]: 'cell_id'}, inplace = True)

        ind_dat['cs_id'] = df2['cs_id']
        ind_dat['aveT'] = df2['aveT']
        ind_dat['t_ext'] = df2['t_ext']
        ind_dat['t_inh'] = df2['t_inh']
        ind_dat['animalID'] = animal
        data2 = pd.concat([data2, ind_dat], axis = 0)
    
    # Concatenate data for all animals
    data.reset_index(drop = True, inplace = True)
    data2.reset_index(drop = True, inplace = True)
    
    # Sort by animalID to ensure correct order
    data = data.sort_values(by='animalID').reset_index(drop=True)
    data2 = data2.sort_values(by='animalID').reset_index(drop=True)

    # Merge and ensure no sorting happens during the merge
    all_data = pd.merge(data2, data, on = ['animalID', 'cs_id'], how = 'inner', sort = False)
    
    # Final sort for consistency
    all_data = all_data.sort_values(by=['animalID', 'session', 'cs_id']).reset_index(drop=True)
    
    first_column = all_data['animalID']
    second_column = all_data['session']
    third_column = all_data['cs_id']
    all_data.drop(columns = ['animalID', 'session', 'cs_id'], inplace = True )
    all_data.insert(0, 'animalID', first_column)
    all_data.insert(1, 'session', second_column)
    all_data.insert(2, 'cs_id', third_column)
                      
    return all_data

#%% Sort the cells based on the responses in test 1 
cta_list = ['CWP16','CWP17', 'CWP20', 'CWP27','CWP28', 'CWP29', 'CWP39', 'CWP47']
csonly_list = ['CWP35','CWP36', 'CWP37', 'CWP40','CWP42', 'CWP51','CWP52']
ko_list = ['CWShkP7', 'CWShkP8', 'CWShkP12', 'CWShkP14', 'CWShkP16', 'CWShkP18', 'CWShkP26']

folder = ''
outfolder = r"{folder}\pooled_cst_data\signal_noise_corr"
session = ''
cond_dict ={'CTA': cta_list, 'CSonly': csonly_list, 'KO': ko_list}
cond_dict ={'KO': ko_list}

for cond in cond_dict.keys():
    for k in ['signal', 'noise', 'concat']:
        dat = pooledRegTrace(folder, session, cond_dict[cond],
                                        trace_type = k, reg_type = 'nonreg')
        np.save(rf'{outfolder}\nonreg_{session}_{k}_trace_{cond}', dat)
                 
            
            
                 
    
    
    
    

