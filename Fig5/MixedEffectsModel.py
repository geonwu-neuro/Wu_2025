import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
from lme_two_categories import *
from DprimeDistribution import dprime


# Generate a long-form dataframe for linear model fitting, 
# Dependent valuable: absolute dprime

# Provide animalID for each group
cs_list = []
wt_list = []
ko_list = []
allgroup = {"CSonly": cs_list, "CTAWT": wt_list, "CTAKO": ko_list}

abs_all = pd.DataFrame()
for group in allgroup.keys():
    for i, session in enumerate(['test1','test2','test5']):
        df = dprime(allgroup[group]. session)
        sub_df = df[['animalID', 'abs_d']].copy()
        sub_df['genotype'] =group
        sub_df['session'] = session
        abs_all = pd.concat([abs_all, sub_df], axis = 0)
abs_all = abs_all.reset_index(drop = True)    


# Perform mixed_effect models comparing csonly vs. cta (wt) vs. cta (ko)
# Set the CSonly and test1 as the reference group
abs_all['genotype'] = pd.Categorical(abs_all['genotype'], categories = ['CSonly','CTAWT','CTAKO'],
                                      ordered = True)


mixed_model = mixedlm("abs_d ~ C(genotype)*C(session)", 
                        groups = abs_all['animalID'], data = abs_all).fit(reml = False)
print(mixed_model.summary())

# Save the summary of the model to a file
summary_df = pd.DataFrame({
    'coef': mixed_model.params.values,
    'std_err': mixed_model.bse,
    'z-value': mixed_model.tvalues,
    'p-value': mixed_model.pvalues})

# Posthoc pairwise tests using linear constrast from the model  

p_unc = []

# Step 1 : building a contrast matrix for each pair
# Compute specific contrast from the fitted model

design_info = mixed_model.model.data.design_info

def design_row(session, condition):
    formula = "C(session, Treatment(reference = 'test1'))*C(genotype, Treatment(reference = 'CSonly'))"
    df_tmp = pd.concat([abs_all, pd.DataFrame([[session, condition]], 
                          columns = ['session', 'genotype'])], ignore_index = True)    
    dm = dmatrix(formula, df_tmp, return_type = 'dataframe')
    
    return dm.iloc[[-1], :]

contrast  = [
    ('WT t2 vs t1', ('test2', 'CTA'), ('test1', 'CTA')),
    ('KO t2 vs t1', ('test2', 'KO'), ('test1', 'KO')),
    ('WT vs KO at t1', ('test1', 'CTA'), ('test1', 'KO')),
    ('WT vs KO at t2', ('test2', 'CTA'), ('test2', 'KO')),
    ('WT vs KO at t5', ('test5', 'CTA'), ('test5', 'KO'))]


# Step 2 : calculate posthoc test for each contrast pair
for name, (t1, c1), (t2, c2) in contrast:
    row1 = design_row(t1,c1)
    row2 = design_row(t2,c2)
    contrast_vector = row1 - row2
    
    test = mixed_model.t_test(contrast_vector)
    p_unc.append(test.pvalue)
    

# Step 3: multiple comparsion correction (FDR)
reject, p_corrected, _, _ = multipletests(p_unc, method = 'fdr_bh')
print(p_corrected)


# 
# Reorder the fixed effect parameters for plotting
desired_order = ['Intercept', 'C(session)[T.test2]',
                 'C(session)[T.test5]',
                 'C(genotype)[T.CTA]',
                 'C(genotype)[T.CTA]:C(session)[T.test2]',
                 'C(genotype)[T.CTA]:C(session)[T.test5]', 
                 'C(genotype)[T.KO]',
                 'C(genotype)[T.KO]:C(session)[T.test2]',
                 'C(genotype)[T.KO]:C(session)[T.test5]']

fixed_effect = mixed_model.fe_params.reindex(desired_order)
stderr = mixed_model.bse.reindex(desired_order)
label = list(fixed_effect.index)

# Calculated the fitted mean (group level) 
fitted_mean = []
fitted_se = list(stderr)[:9]
current_mean = 0

for i, key in enumerate(label):
    if i == 0:
        fitted_mean.append(fixed_effect[key])
    elif i >0 and i <= 2:
        fitted_mean.append(fitted_mean[0] + fixed_effect[key])
    elif i >2 and i <=5:
        num = i - 3
        fitted_mean.append(fitted_mean[num] + fixed_effect[key])
    elif i >5 and i <=8:
        num = i - 6
        fitted_mean.append(fitted_mean[num] + fixed_effect[key])


# Plotting the fitted  means (group and individual animal levels)
primary_x = ['T1','T2','T5']*3
color = ['teal']*3 + ['navy']*3 + ['darkgoldenrod']*3
fig, ax, _ = plot_lme_two_categories(lme_model = mixed_model, 
                   newtable = abs_all, 
                   cat1_name = 'genotype', 
                   cat2_name = 'session', 
                   y_name = 'abs_d', 
                   condition_order = desired_order, 
                   fitted_mean = fitted_mean, 
                   fitted_se = fitted_se, 
                   reference_category1 = 'CSonly', 
                   reference_category2 = 'test1',
                   group = 'animalID',
                   color = color,
                   xtick_labels = primary_x,
                   figsize = (12,5))

ax.set_title('Linear mixed-effects model', y = 1.2, fontsize = 24)
ax.set_ylabel('Absolute discriminability index')
plt.show()
