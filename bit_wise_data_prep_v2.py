# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:18:14 2024

@author: jfodopsokou and maxhilding
"""

import pandas as pd
import numpy as np
import random

import os
import glob

import matplotlib.pyplot as plt

def read_list_of_empty_sav_files(sav_files, freq=False):
    file = sav_files[0]
    data = pd.read_csv(file, header=None)
    #freq = data.iloc[:,0]
    x_real = data.iloc[:,3].values.reshape(1,-1) 
    x_imag = data.iloc[:,4].values.reshape(1,-1)

    sources = [file[-7:-4]] #Indication the parameter considered (S11, S21, ...)

    for file in sav_files[1:]:
        data = pd.read_csv(file, header=None)
        x_real = np.vstack((x_real, data.iloc[:,3].values.reshape(1,-1)))
        x_imag = np.vstack((x_imag, data.iloc[:,4].values.reshape(1,-1)))
    
        sources += [file[-7:-4]]
    
    X = np.stack((x_real, x_imag), axis=2)
    
    return X, sources

def find_positions(string, char):
    positions = []
    for i in range(len(string)):
        if string[i] == char:
            positions.append(i)
    return positions

def read_list_of_sav_files(folder, sav_files, X_empty, sources_empty):
    pos = find_positions(folder, '_')
    
    file = sav_files[0]
    data = pd.read_csv(file, header=None)
    freq = data.iloc[:,0]
    sources = [] #The parameter (S11, S21, S12 or S22)
    
    source = file[-7:-4]
    sources += [source]
    
    source_idx = sources_empty.index(source) # source index in the sources_empty list
    x = data.iloc[:,3].values
    x = x - X_empty[source_idx, :, 0]        # applying background subtraction on the real part
    x_real = x.reshape(1,-1) 
    
    x = data.iloc[:,4].values
    x = x - X_empty[source_idx, :, 1]        # applying background subtraction on the imaginary part
    x_imag = x.reshape(1,-1)

    for file in sav_files[1:]:
        #print(file)
        data = pd.read_csv(file, header=None)

        source = file[-7:-4]
        sources += [source]

        source_idx = sources_empty.index(source)
        x = data.iloc[:,3].values
        x = x - X_empty[source_idx, :, 0]        
        x_real = np.vstack((x_real, x.reshape(1,-1)))

        x = data.iloc[:,4].values
        x = x - X_empty[source_idx, :, 1]        
        x_imag = np.vstack((x_imag, x.reshape(1,-1)))
    
    X = np.stack((x_real, x_imag), axis=2)
    
    m = X.shape[0]
    code = folder[pos[0]+1:pos[1]]
    codes = [code] * m

    interval = folder[pos[1]+1:-2]
    intervals = [interval] * m
    
    return X, codes, intervals, sources, freq

def real_all_tags_in_folder(reading_path):
    os.chdir(reading_path)
    folders = [entry for entry in os.listdir(reading_path) if os.path.isdir(entry)] #list of folders

    folder = folders[0]
    #pos = find_positions(folder, '_')

    path = os.path.join(reading_path, folder)
    os.chdir(path)
    sav_files = glob.glob('*.sav')
    empty_files = [file for file in sav_files if 'empty' in file]

    sav_files = list(set(sav_files) - set(empty_files))

    X_empty, sources_empty = read_list_of_empty_sav_files(empty_files)
    X, codes, intervals, sources, freq = read_list_of_sav_files(folder=folder, sav_files=sav_files, X_empty=X_empty, 
                                                                sources_empty=sources_empty)

    for folder in folders[1:]:
        path = os.path.join(reading_path, folder)
        os.chdir(path)
        sav_files = glob.glob('*.sav')

        empty_files = [file for file in sav_files if 'empty' in file]

        sav_files = list(set(sav_files) - set(empty_files))
        X_empty, sources_empty = read_list_of_empty_sav_files(empty_files)
        X_dum, codes_dum, intervals_dum, sources_dum, _ = read_list_of_sav_files(folder=folder, sav_files=sav_files, 
                                                                                 X_empty=X_empty, sources_empty=sources_empty)

        X = np.vstack((X, X_dum))
        sources += sources_dum

        codes += codes_dum
        intervals += intervals_dum
    
    return X, codes, sources, intervals, freq

def plot_tag(freq, x, xlim=[2.8, 7.3], vline_points=[3.2, 3.65, 3.9, 4.2, 4.6, 5.5, 6.5], figsize=(8,8), 
             title='', scaled=False):
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    #code = codes[1]


    for vline in vline_points:
        axes[0].axvline(vline, color='black')
        axes[1].axvline(vline, color='black')

    #ax.plot(freq/10**9, 20*np.log10(np.absolute(x)), color='blue')
    x_abs = np.absolute(x)
    if scaled:
        x_abs = (x_abs - x_abs.min()) / (x_abs.max() - x_abs.min())
        
    axes[0].plot(freq/10**9, x_abs, color='blue')
    axes[0].set_xlim(xlim)
    axes[0].set_xlabel('Frequency (GHz)', color='blue')
    axes[0].set_ylabel('Amplitude (...)')

    axes[1].plot(freq/10**9, np.angle(x), color='green')
    axes[1].set_xlim(xlim)
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Angle (rd)', color='green')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.95)
    
def prep_set_for_time_gating(path):
    X, codes, _, _, freq = real_all_tags_in_folder(reading_path=path)
    X_complex = X[:,:,0] + 1j*X[:,:,1] 
    X_complex = X_complex.reshape(X_complex.shape + (1,))

    X_mag = np.absolute(X_complex)
    X_ang = np.angle(X_complex)

    y = [np.array([int(i) for i in code]) for code in codes]
    y = np.array(y)
    
    dataset = {'dataMag': X_mag[:,:,0], 'dataAng': X_ang[:,:,0], 'labels': y, 'freq': freq.values}
    
    return dataset

def plot_random(X_to_use, codes_to_use, freq, freq_limits):
    idx = random.randint(0, X_to_use.shape[0]-1)
    print(idx)
    _, ax = plt.subplots(figsize=(10,6))

    ax.plot(freq/10**9, X_to_use[idx])

    for vline in freq_limits[1:-1]:
        ax.axvline(vline/10**9, color='black')
        ax.axvline(vline/10**9, color='black')

    ax.set_xlim([2.8, 7.3])
    ax.set_title(codes_to_use[idx])