import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Calculate the rate (derivative) of the logistic function
def logistic_derivative(x, L, k, x0):
    return (L * k * np.exp(-k * (x - x0))) / (1 + np.exp(-k * (x - x0)))**2


# Initial guess for parameters: [L (max value), k (growth rate), x0 (midpoint)]
def curve_smoothing(df, initial_guess = [1, 0.5, 3]):
    animalID = df.animalID.unique()[0]
    # Change the session for string to int  
    df["session"] = df["session"].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(by = 'session', ascending = True)
    # Determine x and y
    x =  df["session"].to_numpy() # Days
    y = df["lickratio"].to_numpy()  # Continuous values  
    
    
    # Fit the logistic curve to both datasets
    params, _ = curve_fit(logistic, x, y, p0=initial_guess)
    L, k, x0 = params # x0 is the inflection point
    
    # Get the fitted logistic values for plotting
    #fitted = logistic(x, *params)
    time = np.linspace(1, 5, 100)
    fitted = logistic(time, *params)
    
    # Apply Savitzky-Golay filter to smooth the fitted logistic curve (window_length must be odd)
    smoothed = savgol_filter(fitted, window_length=3, polyorder=2) 
    # window_length: must be odd num, polyorder, usually 2 or 3
    
    # load it to dataframe
    smoothed_df = pd.DataFrame(list(smoothed), index = [np.linspace(1,5,100)])
    smoothed_df = smoothed_df.T.copy()
    smoothed_df.index = [animalID]
    
    # Calculate the rate (derivative) 
    #rate = logistic_derivative(x, *params)
    rate_100 = logistic_derivative(time, *params)
    
    # Apply Savitzky-Golay filter to smooth the fitted logistic curve (window_length must be odd)
    smoothed_rate = savgol_filter(rate_100, window_length=3, polyorder=2)
    
    # load it to the a combined file
    smoothed_rate_df = pd.DataFrame(list(smoothed_rate), index = [np.linspace(1,5,100)])
    smoothed_rate_df = smoothed_rate_df.T.copy()
    smoothed_rate_df.index = [animalID]
    
    return smoothed_df, smoothed_rate_df
