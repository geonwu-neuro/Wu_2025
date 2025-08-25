import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_normalized_euclid(df, genotype):
    session_list = ['test1','test2','test3','test4','test5']
    combined = pd.DataFrame()
    df_gr = df.groupby('animalID')
    for animal in df.animalID.unique():
        ind_df = df_gr.get_group(animal)
        ind_dict ={}
        ind_dict['animalID'] = animal
        for session in session_list:
            s_vec = np.array(ind_df.loc[ind_df.session == session, 'aveT_s'])
            w_vec = np.array(ind_df.loc[ind_df.session == session, 'aveT_w'])
            
            ind_dict[session] = np.linalg.norm(s_vec - w_vec) / np.sqrt(len(s_vec)) # Normalizing
        
        df_toadd = pd.DataFrame(ind_dict, index = [0])
        combined = pd.concat([combined, df_toadd], axis = 0)
    combined['genotype'] = genotype 
    
    # Change the dataframe into a long form for linear model fitting
    combined_melt = pd.melt(frame = combined, id_vars = ['animalID','genotype'],
                      value_vars = wt_df.columns, var_name = 'session', value_name = 'euclid')
    
    return combined_melt
