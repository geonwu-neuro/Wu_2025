import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def shape_data_byanimal(data, animal):
    grouped = data.groupby('animalID')
    num1 = grouped.get_group(animal)
    num1_label = num1.loc[~num1.trial.duplicated(), ['trial','taste']]
    num1_label = num1_label.sort_values('trial', ascending = True).reset_index(drop = True)
    num1_label = num1_label.set_index('trial')
    
    num1_piv = num1.pivot(columns = 'cs_id', index = 'trial', values = 'aveT')
    
    dat = pd.concat([num1_label, num1_piv], axis = 1)
    
    return dat

session_list = ['test1','test2', 'test3','test4', 'test5']
accuracy_df = pd.DataFrame()

for session in session_list:
    df = pd.read_csv('filepath') # open the training session file
    animal_list = list(df.animalID.unique())
    for animal in animal_list:
        result = {}
        result['training_session'] = session
        result['animalID'] = animal
        
        ind_df = shape_data_byanimal(df, animal)
        cell_order = list(ind_df.columns) # feature orders
              
        #
        # Get X and y in array format
        X = ind_df.drop(columns = 'taste').values # Add standardization 
        y = ind_df.taste.values
        class_le = LabelEncoder()
        y = class_le.fit_transform(y)
        
        # A simple code to trainin and generate a SVC (with accuracy score)
        #  for a single animal in a given session
        kfold = KFold(n_splits=10, shuffle=True, random_state = 42)
        clf =SVC(kernel = 'linear',  random_state = 42)
        score = cross_val_score(estimator = clf, X=X, y=y, cv = kfold, scoring = 'accuracy') # use total data or splitted data?
        print(f'{session}, {animal}, accuracy: {score.mean()}')
        result[session] = score.mean()
        
        # train a model in order to predict another session:
        clf2 = SVC(kernel = 'linear', random_state = 42)
        clf2.fit(X, y)
        
        # get each test data, selecting the animal, reorder the cell oroder to train model 
        for test in [s for s in session_list if s != session]:
            test_df = pd.read_csv('filepath') # open the specific session file)
            test_df2 = shape_data_byanimal(test_df, animal)
            test_df2 = test_df2[cell_order]
            
            X2 = test_df2.drop(columns = 'taste').values 
            y2 = test_df2.taste.values
            class_le2 = LabelEncoder()
            y2 = class_le.fit_transform(y2)
            
            y2_predict = clf2.predict(X2)
            cs_score = accuracy_score(y2, y2_predict)
            result[test] = cs_score
        
        add = pd.DataFrame(result, index = [0])
        accuracy_df = pd.concat([accuracy_df, add], axis = 0)
        
    
# Plot heatmap
df_to_plot = accuracy_df.drop(columns = {'animalID', 'training_session'})
fig, ax = plt.subplots()
sns.heatmap(df_to_plot)
plt.show()

