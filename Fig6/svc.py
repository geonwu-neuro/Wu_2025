import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def shape_data_byanimal(data, animal):
    grouped = data.groupby('animalID')
    num1 = grouped.get_group(animal)
    num1_label = num1.loc[~num1.trial.duplicated(), ['trial','taste']]
    num1_label = num1_label.sort_values('trial', ascending = True).reset_index(drop = True)
    num1_label = num1_label.set_index('trial')
    
    num1_piv = num1.pivot(columns = 'cs_id', index = 'trial', values = 'aveT')
    
    dat = pd.concat([num1_label, num1_piv], axis = 1)
    
    return dat
#
session_list = ['test1','test2', 'test3','test4', 'test5']

accuracy_df = pd.DataFrame(columns = session_list)
accuracy_shuffled = pd.DataFrame(columns = session_list)
conf_matrix_summary = {}

for session in session_list:
    df = pd.read_csv("filepath") # point to the specific session file
    byanimal_dict = {}
    for animal in list(df.animalID.unique()):
        ind_df = shape_data_byanimal(df, animal)
    
        # Get X and y in array format
        X = ind_df.drop(columns = 'taste').values 
        y = ind_df.taste.values
        class_le = LabelEncoder()
        y = class_le.fit_transform(y)
        
        # Step 1: A simple code to generate a SVC and the accuracy score. 
        #         Instead of splitting data into training and test dataset, run 10-fold cv
        #         and get an averaged accuracy score
        
        # Initialize Kfold and classifier
        kfold = KFold(n_splits=10, shuffle=True, random_state = 42)
        clf =SVC(kernel = 'linear',  random_state = 42)
        
        score = cross_val_score(estimator = clf, X=X, y=y, cv = kfold, scoring = 'accuracy') # use total data or splitted data?
        print(f'{session}, {animal}, accuracy: {score.mean()}')
        accuracy_df.loc[animal, session] = score.mean()
        
        # Step 2: Compute summed confusion matrix
        num_classes = len(np.unique(y))  # Number of classes
        summed_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)  # Train model
            y_pred = clf.predict(X_test)  # Predict
            
            # Compute confusion matrix and accumulate
            conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            summed_conf_matrix += conf_matrix
        
        print(f'{session}, {animal}, Summed Confusion Matrix:\n{summed_conf_matrix}')
        
        #Get class names in the correct order
        class_names = class_le.classes_
           
        # Convert summed confusion matrix to a labeled DataFrame
        conf_matrix_df = pd.DataFrame(summed_conf_matrix, 
                              index=class_names, 
                              columns=class_names)
        
        conf_matrix_summary[fr'{animal}_{session}'] = conf_matrix_df
        
        
        # Step 3: shuffled data and generate the distribution and mean of the chance score 
        iteration = 1000
        boot_list = []
        for i in range(iteration):
            y_sh = np.random.permutation(y)
            # use random.permutation instead of shuffle, so not modifying the orinal labels in place
            kfold2 = KFold(n_splits=10, shuffle=True, random_state = 42)
            clf2 = SVC(kernel = 'linear', C = 1, random_state = 42)
            score_sh = cross_val_score(estimator = clf2, X=X, y=y_sh, cv = kfold2, scoring = 'accuracy')
            boot_list.append(score_sh.mean())
        
        boot_mean = np.mean(boot_list)
        print(f'{session}, {animal}, shuffled_accuracy: {boot_mean}')
        accuracy_shuffled.loc[animal, session] = boot_mean
        
