# Adapted from Van Hooser Lab (github:) 

import numpy as np
import matplotlib.pyplot as plt


def plot_lme_two_categories(lme_model, newtable, cat1_name, cat2_name, y_name, condition_order, fitted_mean, 
                            fitted_se, reference_category1, reference_category2, group, 
                            xtick_labels = ['T1','T2', 'T5']*2, **kwargs):
    """
    Plot linear mixed-effects model predictions for two categorical variables.
    Args:
        lme_model: Fitted statsmodels.MixedLMResults object.
        newtable: Pandas DataFrame containing the data.
        cat1_name: Name of the first categorical variable.
        cat2_name: Name of the second categorical variable.
        y_name: Name of the response variable.
        reference_category1: Reference category for the first categorical variable.
        reference_category2: Reference category for the second categorical variable.
        group: Name of the grouping variable (random effects).
        kwargs: Optional keyword arguments for customization.
    Returns:
        fig, ax: Matplotlib figure and axis objects.
        plot_info: Dictionary containing plot-related data.
    """
    print(kwargs)
    # Extract unique categories
    cat1_levels = list(newtable[cat1_name].unique())
    cat2_levels = list(newtable[cat2_name].unique())
    random_effects = lme_model.random_effects
    #fixed_effects = lme_model.fe_params.reindex(condition_order)
    
    # order the levels
    cat1_levels.remove(reference_category1) # Genotype
    cat1_levels.insert(0, reference_category1)
    
    cat2_levels = sorted(cat2_levels) # Session


    # Initialize plot variables
    x = []
    y = []
    x_centers = []
    group_labels = []
    condition_effects = [] # stores the effects of each condition (category condition)
    condition_boundaries = [] # store the boundaries of each condition (used to defien the space between conditions )
    current_spot = 1

    # Create figure and axis
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
    
    cond = 0
    # Iterate over combinations of cat1 and cat2
    for i, cat1 in enumerate(cat1_levels):
        for j, cat2 in enumerate(cat2_levels):
            group_spot_start = current_spot
            subset = newtable[(newtable[cat1_name] == cat1) & (newtable[cat2_name] == cat2)]
            
            if subset.empty:
                continue

            for group_label, group_data in subset.groupby(group):
                if not group_data.empty:
                    # Jittered x-coordinates for each group
                    random_offsets = (np.random.rand(len(group_data)) - 0.5) / 2
                    x_coords = current_spot + random_offsets
                    x.append(x_coords)
                    y.append(group_data[y_name].values)
                    x_centers.append(current_spot)

                    # Add group labels and effects
                    group_labels.append(group_label)
                    group_effect = random_effects[group_label][0]
                    
                    condition_effect = fitted_mean[cond]
                    condition_effects.append(group_effect + condition_effect)

                    # Increment for the next group
                    current_spot += kwargs.get('within_category_space', 1)

            # Save category boundaries
            condition_boundaries.append((group_spot_start, current_spot))
            current_spot += kwargs.get('across_category_space', 1)
            cond += 1
            
    # Plot data points
    for i, (x_vals, y_vals) in enumerate(zip(x, y)):
        ax.scatter(x_vals, y_vals, label=f'Group {group_labels[i]}', alpha=0.7, color = 'grey')

    # Plot group means
    for i, (x_center, group_mean) in enumerate(zip(x_centers, condition_effects)):
        ax.plot([x_center - 0.5, x_center + 0.5], [group_mean, group_mean], color='k', lw=3)
    
    colormap = kwargs.get('color', ['blue']*len(fitted_mean))
   
    # Plot category means and standard errors
    for (start, end), mean, se, c in zip(condition_boundaries, fitted_mean, fitted_se, colormap):
        
        ax.plot([start, end], [mean, mean], color= c, lw=5)
        ax.plot([0.5 * (start + end)] * 2, [mean - se, mean + se], color= c, lw=5)

    # Customize plot
    xtick_labels = kwargs.get('xtick_labels', xtick_labels)
    ax.set_xticks([0.5 * (start + end) for start, end in condition_boundaries])
    ax.set_xticklabels(xtick_labels, fontsize = 20)
    # ax.set_xlabel(f'{cat1_name} and {cat2_name}')
    ax.set_ylabel(kwargs.get('ylabel', 'Fitted absolute index'), fontsize = 20)
    # ax.set_ylim(-20,20)
    # ax.legend(loc='best')
    # ax.grid(True)
    plt.tight_layout()
    

    # Return figure and additional info
    plot_info = {
        'x': x,
        'y': y,
        'x_centers': x_centers,
        'condition_effects': condition_effects,
        'condition_boundaries': condition_boundaries,
    }
    return fig, ax, plot_info
