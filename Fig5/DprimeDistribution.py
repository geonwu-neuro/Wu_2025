import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def dprime(animal_list, session):
    folderpath = "folderpath" # where all the files are stored
    pooled_dat = pd.DataFrame()
    
    for animal in animal_list:
        for roots, dirs, files in os.walk(folderpath): 
            for file in [f for f in files if (session in f) and ('reg' not in f) and (animal in f) and \
                         ('activity_ID' in f) and ('sacc' in f)]:
            
                dat_dict = np.load(os.path.join(roots, file), allow_pickle = True).item()
                mean_dat = pd.DataFrame()
                animal_dat = pd.DataFrame()
                for k in dat_dict.keys(): # k is trial num
                    new_col = f'sacc_{k}'
                    dat_dict[k].rename(columns = {'z_mean_taste': new_col}, inplace = True)
                    animal_dat = pd.concat([animal_dat, dat_dict[k][new_col]], axis = 1)
                mean_dat['mean_sacc'] = animal_dat.mean(axis = 1)   
                mean_dat['std_sacc'] = animal_dat.std(axis = 1)    
                mean_dat.reset_index(inplace = True)
                mean_dat = mean_dat.rename(columns = {'index': 'cell_id'})
                
                
                mean_dat['animalID'] = animal 
                first_column = mean_dat['animalID']
                mean_dat.drop(columns = ['animalID'], inplace = True )
                mean_dat.insert(0, 'animalID', first_column)
                
             
            for file in [f for f in files if (session in f) and ('reg' not in f) and (animal in f) and \
                         ('activity_ID' in f) and ('water' in f)]:
            
               dat_dict2 = np.load(os.path.join(roots, file), allow_pickle = True).item()
               mean_dat2 = pd.DataFrame()
               animal_dat2 = pd.DataFrame()
               for k in dat_dict2.keys(): # k is trial num
                   new_col = f'water_{k}'
                   dat_dict2[k].rename(columns = {'z_mean_taste': new_col}, inplace = True)
                   animal_dat2 = pd.concat([animal_dat2, dat_dict2[k][new_col]], axis = 1)
               mean_dat2['mean_water'] = animal_dat2.mean(axis = 1)   
               mean_dat2['std_water'] = animal_dat2.std(axis = 1)   
               mean_dat2.reset_index(inplace = True)
               mean_dat2 = mean_dat2.rename(columns = {'index': 'cell_id'})
               
               mean_dat2['animalID'] = animal
               first_column = mean_dat2['animalID']
               mean_dat2.drop(columns = ['animalID'], inplace = True )
               mean_dat2.insert(0, 'animalID', first_column)
        
        combined = pd.merge(mean_dat, mean_dat2, on = ['animalID', 'cell_id'])
        pooled_dat = pd.concat([pooled_dat, combined], axis = 0)
    
    pooled_dat = pooled_dat.reset_index(drop = True)
    pooled_dat = pooled_dat.dropna() 
    pooled_dat['d-prime'] = (pooled_dat.mean_sacc - pooled_dat.mean_water) /
                           np.sqrt(0.5*(pooled_dat.std_sacc*pooled_dat.std_sacc 
                                        + pooled_dat.std_water*pooled_dat.std_water))
    pooled_dat['abs_d'] = pooled_dat['d-prime'].apply(lambda x: np.abs(x))
                      
    return pooled_dat


def f_test(data1, data2):
    if np.var(data1) > np.var(data2):
        F = np.var(data1) / np.var(data2)
        df1 = len(data1) - 1
        df2 = len(data2) - 1
    elif np.var(data2) > np.var(data1):
        F = np.var(data2) / np.var(data1)
        df1 = len(data2) - 1
        df2 = len(data1) - 1
    
    p_val = 1 - f.cdf(F, df1, df2)
    return F, p_val 


def plot_hist(animal_list, title, comp1, comp2, palette):
    df1 = dprime(animal_list, comp1)
    df2 = dprime(animal_list, comp2)
    
    # Stats for variance
    
    F, p =  f_test(df1['d-prime'], df2['d-prime'])
    print(f'{comp1}: {len(df1)} cells')
    print(f'{comp2}: {len(df2)} cells')
    print(f'{comp1} vs {comp2}, f-test: p = {p}')
    
    fig, ax = plt.subplots(figsize = (4,2.5))
    ax.hist(df1['d-prime'], bins = 30 , color = palette[0], alpha = 0.5, label = 'T1')
    ax.hist(df2['d-prime'], bins = 30, color = palette[1], alpha = 1, label = 'T2')
    ax.set_xlim(-3,3)
    ax.legend(loc= 'center right', bbox_to_anchor = (1,0.7), fontsize = 12)
    ax.set_ylabel('Neurons', fontsize = 16)
    ax.set_xlabel('Discriminability index', fontsize = 16)
    ax.tick_params('both', labelsize = 14)
    
    plt.suptitle(title, fontsize = 20)
    plt.tight_layout()
    plt.show()
