# -*- coding: utf-8 -*-

#import cupy as cp

"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random, keyboard, time
from scipy.spatial.distance import cdist

def filter_cwt(X, thr):
    
    X_copy = X.copy() # Convert input array to CuPy array
    
    X_copy[X_copy < thr] = 0
    
    return X_copy # Convert the filtered array back to NumPy array and return

def make_confusion_matrix_final(
    cf, ax=None,
    cbar=True,
    cmap='Blues',
    annot=True,
    xticklabels='auto',
    yticklabels='auto',
    xyplotlabels=True,
    sum_stats=True,
):
    '''
    Pretty plot of sklearn confusion matrix
    
    cf : confusion matrix
    figsize : Tuple representing the figure size
    cbar : If True, show the color bar. Default value is True. 
    cmap : Colormap of the values displayed. Default value is 'Blues'
    annot : If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate heatmap instead of data.
    xticklabels : List of strings containing the categories to be displayed on the x axis. Default value is 'auto' (densely plot non-overlapping lables)
    yticklabels : List of strings containing the categories to be displayed on the y axis. Default value is 'auto' (densely plot non-overlapping lables)
    xyplotlabels : If True, show 'True Label' and 'Predicted Label' on the figure. Default value: True
    sumstats : If Tue, display summary statistics below the figure
    '''
    
    if sum_stats:
        accuracy = np.trace(cf) / float(np.sum(cf))
        stats_text = '\nAccuracy={:0.3f}'.format(accuracy)
    else:
        stats_text = ''
    #fig = plt.figure(figsize=figsize)
    
    sns.heatmap(cf, cmap=cmap, cbar=cbar, annot=annot, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    
    if xyplotlabels:
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label' + stats_text)
    else:
        ax.set_xlabel(stats_text)

def testing(le, model, X_test, y_test):
    #predict_proba = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        predict_proba = model.predict_proba(X_test)
        y_pred = np.argmax(predict_proba, axis=1)
        
    else:
        predict_proba = model.predict(X_test)
        y_pred = np.argmax(predict_proba, axis=1)
    
    categories = le.inverse_transform(np.union1d(np.unique(y_pred), np.unique(y_test)))
    labels_test = le.inverse_transform(y_test)
    labels_pred = le.inverse_transform(y_pred)
    cf = confusion_matrix(labels_test, labels_pred)
    
    fig, ax = plt.subplots(figsize=(10,8))
    return make_confusion_matrix_final(cf, ax=ax, cbar=False, xticklabels=categories, yticklabels=categories)


def explore_dataset(X, freq, title):
    paused = False
    fig, ax = plt.subplots(figsize=(10, 6))
    
    while True:
        ax.clear()
        rand_idx = random.randint(0, len(X)-1)
        #print(rand_idx)
        signal = X[rand_idx]
        
        ax.plot(freq, signal)
        ax.set_title(title[rand_idx])
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        
        plt.pause(1)
        
        if keyboard.is_pressed(' '):
            paused = not paused
            while paused:
                if keyboard.is_pressed(' '):
                    paused = False
                time.sleep(0.1)
        elif keyboard.is_pressed('q'):
            break
            
        time.sleep(0.1)
        
        plt.close(fig)
        
###############################################################################

def correct_predictions(y_pred, y_test): # Used to correct invalid codes
    yt_unique = np.unique(y_test, axis=0)
    yt_set = set(map(tuple, yt_unique)) # Converting arrys to sets of tuples for easy comparison
    valid_mask = np.array([tuple(row) in yt_set for row in y_pred]) # Valid predictions
    
    # Finding indices of invalid rows
    invalid_indices = np.where(~valid_mask)[0]
    
    if len(invalid_indices) == 0:
        return y_pred
    
    # Computing Hamming distances between wrong predictions and yt_unique
    h_distances = cdist(y_pred[invalid_indices], yt_unique, metric='hamming')
    
    # Nearest valid code
    nearest_indices = h_distances.argmin(axis=1)
    #print(nearest_indices)
    #print(h_distances.shape)
    corrected_y_pred = y_pred.copy()
    corrected_y_pred[invalid_indices] = yt_unique[nearest_indices]
    
    return corrected_y_pred

# Correcting NOT only with Hamming Distance
def correct_predictions_2(y_pred_proba, y_test):
    yt_unique = np.unique(y_test, axis=0)
    y_pred_bit = y_pred_proba.round().astype(int)
    
    yt_set = set(map(tuple, yt_unique)) # Converting arrays to sets of tuples for easy comparison
    valid_mask = np.array([tuple(row) in yt_set for row in y_pred_bit]) # Valid predictions
    
    # Finding indices of invalid rows
    invalid_indices = np.where(~valid_mask)[0]
    
    if len(invalid_indices) == 0:
        return y_pred_bit
    
    # Computing Hamming distances between wrong predictions and yt_unique
    h_distances = cdist(y_pred_bit[invalid_indices], yt_unique, metric='hamming')
    
    min_indices = [np.where(row == row.min())[0] for row in h_distances]
    
    # Nearest valid code
    nearest_indices = []
    
    for i in range(len(min_indices)):
        if len(min_indices[i]) == 1:
            #print(min_indices[i])
            nearest_indices.append(min_indices[i][0])
        else:
            P = y_pred_proba[invalid_indices][i]
            P_rounded = P.round().astype(int)
           
            mismatches = yt_unique[min_indices[i]] != P_rounded # Where the nearest true labels bits are different to the predicted bits
            
            distance_sums = np.sum(np.abs(P - 0.5) * mismatches, axis=1) # Sum of the distance to 0.5 where there is a mismatch
            nearest_indices.append(min_indices[i][distance_sums.argmin()])
            #print(nearest_indices)
            #print(invalid_indices)
    
    corrected_y_pred = y_pred_bit.copy()
    corrected_y_pred[invalid_indices] = yt_unique[nearest_indices]
    
    return corrected_y_pred

# Correcting base only on probabilities
def correct_predictions_3(y_pred_proba, y_test):
    yt_unique = np.unique(y_test, axis=0)
    y_pred_bit = y_pred_proba.round().astype(int)
    
    yt_set = set(map(tuple, yt_unique)) # Converting arrays to sets of tuples for easy comparison
    valid_mask = np.array([tuple(row) in yt_set for row in y_pred_bit]) # Valid predictions
    
    # Finding indices of invalid rows
    invalid_indices = np.where(~valid_mask)[0]
    
    if len(invalid_indices) == 0:
        return y_pred_bit
    
    # Computing Hamming distances between wrong predictions and yt_unique
    #h_distances = cdist(y_pred_bit[invalid_indices], yt_unique, metric='hamming')
    
    min_indices = [np.arange(0,len(yt_unique))]*len(invalid_indices)
    
    # Nearest valid code
    nearest_indices = []
    
    for i in range(len(min_indices)):
        if len(min_indices[i]) == 1:
            #print(min_indices[i])
            nearest_indices.append(min_indices[i][0])
        else:
            P = y_pred_proba[invalid_indices][i]
            P_rounded = P.round().astype(int)
           
            mismatches = yt_unique[min_indices[i]] != P_rounded # Where the nearest true labels bits are different to the predicted bits
            
            distance_sums = np.sum(np.abs(P - 0.5) * mismatches, axis=1) # Sum of the distance to 0.5 where there is a mismatch
            nearest_indices.append(min_indices[i][distance_sums.argmin()])
            #print(nearest_indices)
            #print(invalid_indices)
    
    corrected_y_pred = y_pred_bit.copy()
    corrected_y_pred[invalid_indices] = yt_unique[nearest_indices]
    
    return corrected_y_pred

def testing_bit_wise(X_test, y_test, model, show_cm=True, correct_pred = False, figsize=(10,8)):
    y_pred = model.predict(X_test)
    y_pred = np.squeeze(np.array(y_pred)) #Squeeze is used to remove the last dimension
    y_pred = y_pred.T
    
    if correct_pred:
        #corrected_y_pred = correct_predictions(y_pred.round().astype(int), y_test)
        corrected_y_pred = correct_predictions_3(y_pred, y_test)
    
        y_test_cm = [''.join(map(str, row)) for row in y_test] #Making a labels list made of strings
        
        y_test_cm = [''.join(map(str, row)) for row in y_test]
        y_pred_cm = [''.join(map(str, row)) for row in corrected_y_pred]
    
        categories = np.union1d(np.unique(y_pred_cm), np.unique(y_test_cm))
    
        cm = confusion_matrix(y_test_cm, y_pred_cm)
    
    else:
        y_test_cm = [''.join(map(str, row)) for row in y_test] #Making a labels list made of strings
        
        y_test_cm = [''.join(map(str, row)) for row in y_test]
        y_pred_cm = [''.join(map(str, row)) for row in y_pred.round().astype('uint8')]
    
        categories = np.union1d(np.unique(y_pred_cm), np.unique(y_test_cm))
    
        cm = confusion_matrix(y_test_cm, y_pred_cm)
    
   
    proba_array = y_pred.ravel().reshape(-1,1)
    proba_array = np.concatenate((proba_array, proba_array.round().astype('uint8')), axis=1)
    bool_idx = proba_array[:,1] == y_test.ravel()
    pred_qual = np.array(['Wrong'] * proba_array.shape[0]).reshape(-1,1)
    pred_qual[bool_idx] = 'Right'

    df_pred = pd.DataFrame(data=proba_array, columns=['Probabilities', 'Bits'])
    df_pred['Probabilities'] = df_pred['Probabilities'].astype(float)
    df_pred['Bits'] = df_pred['Bits'].astype('uint8')

    df_pred['Quality'] = pred_qual.astype(str)
    
    # Plotting confusion matrix
    if show_cm:
        _, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[0.75, 0.25]})
        make_confusion_matrix_final(cm, xticklabels=categories, yticklabels=categories, ax=axes[0], cbar=False)
        sns.boxplot(data=df_pred, x='Bits', y='Probabilities', 
                    hue='Quality', 
                    ax=axes[1])
    
    return df_pred, y_pred, y_test_cm, y_pred_cm

def plot_signal(ax, freq, signal, code_length, bandwidth = np.array([1.5 * 10**9, 5 * 10**9]), title=''):
    width = bandwidth[1] - bandwidth[0]
    sub_width = width / code_length
    
    start_points = np.arange(bandwidth[0], bandwidth[1], sub_width, dtype=float)

    vline_points = list(start_points)
    vline_points = np.array(vline_points[1:])
    vline_points = vline_points / 10**9
    
    for x in vline_points:
        ax.axvline(x, color='black')
        ax.axvline(x, color='black')

        ax.axvline(x, color='black')
        ax.axvline(x, color='black')

    ax.set_xlim(bandwidth/10**9)
    ax.plot(freq / 10**9, signal, color='blue')
    ax.set_title(title)

def observe_wrong_pred(y_test, y_pred, code_length, X_test, y_test_cm, y_pred_cm, freq):
    dum = y_pred.round() == y_test
    dum = dum.sum(axis=1)
    wrong_pred_idx = np.where(dum != code_length)[0]

    rand_idx = wrong_pred_idx[random.randint(0, len(wrong_pred_idx)-1)]
    signal = X_test[rand_idx]
    title = y_test_cm[rand_idx] + '(Ground Truth)_' + y_pred_cm[rand_idx] + '(Prediction)'

    _, ax = plt.subplots(figsize=(8,4))
    plot_signal(ax=ax, freq=freq, signal=signal, title=title)
    print('Decisive probabilities: ', y_pred[rand_idx])