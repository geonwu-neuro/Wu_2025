
import numpy as np
import pandas as pd

def pooledRegData(folder, session, animal_list):
    # This function get cells that can be identified in all test sessions, 
    # containing cellstats and activity percentage
    data = pd.DataFrame()
    
    for animal in animal_list:
        
        # Get cross-session cell id
        reg_session = ['cst1','cst2','licl1','licl2','test1', 'test2', 'test3','test4', 'test5']
        footprint_id = np.load(fr'{folder}\{animal}\footprint\align_index.npy')
        footprint_dat = pd.DataFrame(footprint_id, columns = reg_session)
        

        # use the cells that can by tracked throughout all the test sessions
        overlap = footprint_dat[ (footprint_dat['test1'] != -1) & (footprint_dat['test2'] != -1) & 
                                (footprint_dat['test3'] != -1) & (footprint_dat['test4'] != -1) 
                                &(footprint_dat['test5'] != -1)].dropna().reset_index(drop = True)
       
                   
        overlap = overlap.astype(int)
        for i in range(len(overlap)):
            overlap.loc[i, 'cs_id'] = f'{animal}_cs_{i}'
        
        # Remap cell_id to the trial-average delta trace file    
        
        dat_s = np.load(fr"G:\{folder}\{animal}\{session}\{animal}_{session}_activity_ID_byTrial_sacc.npy", 
                      allow_pickle = True).item()
        dat_w = np.load(fr"G:\{folder}\{animal}\{session}\{animal}_{session}_activity_ID_byTrial_water.npy", 
                      allow_pickle = True).item()
        
        dat_dict = {'sacc': dat_s, 'water': dat_w}
        
        for taste in dat_dict:
            for trial in dat_dict[taste].keys():
                ind_df = dat_dict[taste][trial]
                ind_df['animalID'] = animal
                ind_df['session'] = session
                ind_df['taste'] = taste
                ind_df['trial'] = trial
                for i in ind_df.index:
                    if i in overlap[session].values:
                        ind_df.loc[i, 'cs_id'] = overlap.loc[overlap[session] == i, 'cs_id' ].values[0]
            
                ind_df.dropna(inplace = True)
                ind_df.reset_index(drop = True, inplace = True)
                ind_df = ind_df.rename(columns = {'z_mean_taste': 'aveT'})
                data = pd.concat([data, ind_df[['cs_id','animalID','session', 'trial','aveT', 'taste']]], axis = 0)
        
       
    data.reset_index(drop = True, inplace = True)
    
    return data

session_list = ['test1','test2', 'test3','test4', 'test5']
folder = "folderpath"
animal_list =[] # List include all the animalIDs to be included and analyzed
df_dict = {}
for session in session_list:
    df_dict[session] = pooledRegData(folder, session, animal_list)
            

