# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:09:12 2024

@author: jfodopsokou
"""
import numpy as np
from scipy import signal
import random
import matplotlib.pyplot as plt

ALPHA_1 = 0.08
ALPHA_2 = 0.10

def impulse_response(b, a, fs, vline=False, vline_point=None, xlim=[1.5, 5], mode='log', show=False, worN=512):
    freq, h = signal.freqz(b, a, fs=fs, worN=worN)
    
    # Filtering in time domain
    T = 1.463 * 10**(-7) # Duration of the signal in seconds.
    input_signal = np.zeros(int(fs * T))
    input_signal[0] = 1 # Impulse at the begining of the signal duration

    output_signal = signal.lfilter(b, a, input_signal)
    t = np.arange(0, len(output_signal)/fs, 1 / fs, dtype=float) * 10**6 # Time axis values in nanoseconds
    
    '''
    # The result of the following two lines show that the fft of the time domain impulse response is equivalent to the impulse 
    #response compute above. 
    start = 220 # This is compute so that the corresponding sample in the frequency domain is 1.5 GHz (Not necessary, just set
    xlim will be sufficient)
    output_signal_freq = fft.fft(output_signal)[start:start+512]
    freq_red = fft.fftfreq(len(output_signal), d=1/fs)[start:start+512]
    '''
    
    # Plotting
    if show:
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))
        
        ax[0].set_title('Impulse Response')
        ax[0].plot(t, output_signal)
        ax[0].set_xlabel('Time (microseconds)')
        ax[0].set_xlim([-0.001, t[-1]])
        ax[0].grid(True)
        
        if mode == 'log':
            ax[1].plot(freq / 10**9, 20*np.log10(abs(h)), color='blue')
            ax[1].set_ylabel('Amplitude (dB)', color='blue')
        else:
            
            '''
            h_abs = abs(h)
            min_val = min(h_abs)
            max_val = max(h_abs)
            ax[1].plot(freq / 10**9, (h_abs - min_val) / (max_val - min_val), color='blue')
            '''
            ax[1].plot(freq / 10**9, abs(h), color='blue')
            ax[1].set_ylabel('Amplitude', color='blue')
        
        ax[1].set_xlim(xlim)
        #ax[1].set_ylim([-70, 5])
        ax[1].grid(True)
    
        ax[2].plot(freq / 10**9, np.angle(h), color='green')
        ax[2].set_ylabel('Angle (rad)', color='green')
        ax[2].set_xlabel('Frequency (GHz)')
        ax[2].set_xlim(xlim)
        #ax[2].set_ylim([-np.pi/2, np.pi/2])
        ax[2].grid(True)
        
        
        if vline:
            for x in vline_point:
                ax[1].axvline(x, color='black')
                ax[2].axvline(x, color='black')
            
    return freq, h, output_signal

def random_resonance(fs=10**10, f1=2 * 10**9, f2=4.5 * 10**9, 
                     Q1=10, Q2=50, 
                     show_params=True, mode='peak'):
    
    f0 = random.uniform(f1, f2)
    #print(f0)
    Q = random.uniform(Q1, Q2)
    
    if show_params:
        print('Cutoff frequency: {:.2f}'.format(f0/10**9))
        print('Q factor: {:.1f}'.format(Q))

    #digiltal filter coefficients
    if mode == 'peak':
        #print(fs)
        b1, a1 = signal.iirpeak(w0=f0, Q=Q, fs=fs)
    
    else:
        b1, a1 = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    
    return b1, a1, f0

def noising(h, alpha1=0.16, alpha2=0.32, step=1):
    array_length = len(h)
    loc = 0.2
    scale = 0.2
    size = array_length//step
    real_part = np.random.normal(loc=loc, scale=scale, size=size)
    imag_part = np.random.normal(loc=loc, scale=scale, size=size)

    complex_noise = abs(real_part) + 1j * abs(imag_part)
    alpha = random.uniform(alpha1, alpha2)
    noisy_h = h
    noisy_h[::step][:size] = noisy_h[::step][:size] + complex_noise * alpha
    
    return noisy_h

def noising_abs(h_abs, alpha1=0.16, alpha2=0.32, step=1):
    array_length = len(h_abs)
    loc = 0.2
    scale = 0.2
    size = array_length//step
    noise = np.random.normal(loc=loc, scale=scale, size=size)

    alpha = random.uniform(alpha1, alpha2)
    noisy_h_abs = h_abs
    noisy_h_abs[::step][:size] = noisy_h_abs[::step][:size] + noise * alpha
    
    return noisy_h_abs

def random_tag(freq_limits = [2.8*10**9, 3.7*10**9, 4.6*10**9, 5.5*10**9, 6.5*10**9, 7.3*10**9],
               step=2*10**6,
               fs=10**10,
               amp1 = 0.6,
               amp2 = 1,
               #nb_bits = 6,
               code = '101011',
               mode_res = 'peak',
               mode_plot = 'linear',
               Q1=10,
               Q2=50,
               alpha1 = ALPHA_1,
               alpha2 = ALPHA_2,
               delta=0, #how far is the generated peaks from the borders
               step_noise=1,
               worN=512,
               show=False):
    '''
    

    Parameters
    ----------
    bandwidth : TYPE, optional
        DESCRIPTION. The default is np.array([1.5 * 10**9, 5 * 10**9]).
    fs : TYPE, optional
        DESCRIPTION. The default is 10**10.
    #nb_bits : TYPE, optional
        DESCRIPTION. The default is 6.
    code : TYPE, optional
        DESCRIPTION. The default is '101011'.
    mode_res : TYPE, optional
        DESCRIPTION. The default is 'peak'.
    mode_plot : TYPE, optional
        DESCRIPTION. The default is 'linear'.
    Q1 : TYPE, optional
        DESCRIPTION. The default is 10.
    Q2 : TYPE, optional
        DESCRIPTION. The default is 50.
    alpha1 : TYPE, optional
        DESCRIPTION. The default is ALPHA_1.
    alpha2 : TYPE, optional
        DESCRIPTION. The default is ALPHA_2.
    worN : TYPE, optional
        DESCRIPTION. The default is 512.
    show : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    
    #width = bandwidth[1] - bandwidth[0]
    #sub_width = width / len(code)

    #start_points = np.arange(bandwidth[0], bandwidth[1], sub_width, dtype=float)
    idx = np.where(np.array([int(digit) for digit in code]) == 1)[0]
    #subset = start_points[idx]

    #list_subset = list(subset)
    n = len(code)
    #random response to set some parameters: length of the output and vector of frequencies
        #It seems as it is not necessary, freq can be generated. See the code following freq, h = ...
    '''
    b, a = random_resonance(fs=fs, f1=freq_limits[0], f2 = freq_limits[1], Q1=Q1, Q2=Q2, 
                            show_params=False, mode=mode_res)
    #freq, h = signal.freqz(b, a, fs=fs, worN=worN)
    '''
    
    if isinstance(worN, int):
        N = worN
    else:
        N = worN.shape[0]
    
    if len(idx) == 0: #no tag. code = '0000000...'
        h = np.zeros((N,)) + 1j * np.zeros((N,))
        h_noisy = noising(h, alpha1=alpha1, alpha2=alpha2, step=step_noise)
        
        #y = abs(h)
        #y_noisy = abs(h_noisy)
    else:
        h = np.zeros((N,)) + 1j * np.zeros((N,))
        h_noisy = noising(h, alpha1=0, alpha2=0, step=step_noise)

        #freq, h = signal.freqz(b, a, fs=fs, worN=worN)
        h_noisy = noising(h, alpha1=0, alpha2=0, step=step_noise)

        #y = abs(h)
        #y_noisy = abs(h_noisy)
        
        f0_pos = np.zeros(n)

        for i in idx:
            b, a, f0 = random_resonance(fs=fs, f1=freq_limits[i] + delta, f2 = freq_limits[i+1] - delta, Q1=Q1, Q2=Q2, 
                                    show_params=False, mode=mode_res)
            _, h0 = signal.freqz(b, a, fs=fs, worN=worN)
            #abs_val = np.absolute(h0)
            #abs_val_norm = (abs_val - abs_val.min()) / (abs_val.max() - abs_val.min())
            
            
            #f0_pos[i] = f0
            
            amp = random.uniform(amp1, amp2) #making the amplitud of the peaks random
            f0_pos[i] = max(abs(amp*h0)) #saving maximum value instead of peak
            #print(amp)
            h += amp * h0
            h_noisy += noising(amp * h0, alpha1=0, alpha2=0, step=step_noise)
            #y = y + abs(h)
            #y_noisy = y_noisy + abs(h_noisy)
            
        #(abs_val_norm/abs_val) *
    
    #h_noisy += noising(amp * h0, alpha1=0, alpha2=0, step=step_noise)
    y = abs(h)
    y_noisy = noising_abs(abs(h_noisy), alpha1=alpha1, alpha2=alpha2, step=step_noise)
    
    #using random system (a, b) to compute the frequency vector
    b, a, _ = random_resonance(fs=fs, f1=freq_limits[0]+delta, f2 = freq_limits[1]-delta, Q1=Q1, Q2=Q2, 
                            show_params=False, mode=mode_res)
    freq, _ = signal.freqz(b, a, fs=fs, worN=worN)


    if show:
        fig, axes = plt.subplots(2, 2, figsize=(16,8))
        xlim = [freq_limits[0] / 10**9, freq_limits[-1] / 10**9]
        
        if mode_plot == 'linear':
            axes[0, 0].plot(freq / 10**9, y, color='blue')
            axes[0, 1].plot(freq / 10**9, y_noisy, color='blue')
            
        else:
            axes[0, 0].plot(freq / 10**9, 20 * np.log10(y), color='blue')
            axes[0, 1].plot(freq / 10**9, 20 * np.log10(y_noisy), color='blue')
        
        axes[1, 0].plot(freq / 10**9, np.angle(h), color='green')
        axes[1, 1].plot(freq / 10**9, np.angle(h_noisy), color='green')
        
        axes[0, 0].set_xlim(xlim)
        axes[0, 1].set_xlim(xlim)
        
        axes[1, 0].set_xlim(xlim)
        axes[1, 1].set_xlim(xlim)
        
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        
        axes[1, 0].set_ylabel('Angle (rad)', color='green')
        axes[0, 0].set_ylabel('Amplitude', color='blue')

        vline_points = freq_limits[1:-1]
        vline_points = np.array(vline_points[1:])
        vline_points = vline_points / 10**9
    
        for x in vline_points:
            axes[0, 0].axvline(x, color='black')
            axes[0, 1].axvline(x, color='black')
            axes[1, 0].axvline(x, color='black')
            axes[1, 1].axvline(x, color='black')
            
    idx = freq >= freq_limits[0]

    #return h[idx], h_noisy[idx], freq[idx]
    return h[idx], y_noisy[idx], freq[idx], f0_pos

def generate_labels(start=0, step=1, code_length=5, half=False):
    '''
    Generate interatively binary codes from "start" to "2^code_length". "step"
    is used to set the step between to consecutive codes. 

    Parameters
    ----------
    start : TYPE, int
        DESCRIPTION. The default is 0. 
    step : TYPE, optional
        DESCRIPTION. The default is 1.
    code_length : TYPE, optional
        DESCRIPTION. The default is 5.
    half : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    all_codes = [bin(i)[2:].zfill(code_length) for i in range(0, 2**code_length, 1)]
    
    if half:
        # Choosing half of the available code so that each column in the labels matrix has the same number of '1' and '0'
        idx = []
        i = 0
        while i < 2 ** code_length:
            if i % 2 == 0:
                idx.append(i)
                i += 3
            else:
                idx.append(i)
                i += 1
        code_list = np.array(all_codes)[idx]
        
    else:
        code_list = [bin(i)[2:].zfill(code_length) for i in range(start, 2**code_length, step)]
    
    #removing training code (code_list) from all_codes. The remaining codes are
    #testing codes
     
    for code in code_list:
        all_codes.remove(code)
        
    return code_list, all_codes

def generate_set(tags_list, meas_per_tag = 20,
                 freq_limits = [2.8*10**9, 3.7*10**9, 4.6*10**9, 5.5*10**9, 6.5*10**9, 7.3*10**9],
                 step=2*10**6,
                 Q1=30, Q2=50,
                 alpha1=ALPHA_1, alpha2=ALPHA_2, delta=0,
                 worN=512, step_noise=1):
    X = []
    labels = []
    f0_pos = []
    fs = freq_limits[-1] * 2
    
    for code in tags_list:
        for i in range(meas_per_tag):
            
            _, x, _, f0s = random_tag(code=code,
                                step=step,
                                freq_limits = freq_limits,
                                fs=fs, Q1=Q1, Q2=Q2, delta=delta,
                                alpha1=alpha1, alpha2=alpha2, worN=worN, step_noise=step_noise)
            
            X.append(x)
            f0_pos.append(f0s)
            '''
            X.append(random_tag(code=code,
                                step=step,
                                freq_limits = freq_limits,
                                fs=fs, Q1=Q1, Q2=Q2, delta=delta,
                                alpha1=alpha1, alpha2=alpha2, worN=worN, step_noise=step_noise)[1])
            '''
            
        labels.append([code] * meas_per_tag)

    X = np.array(X)
    X = X.reshape(X.shape + (1,))
    labels = np.array(labels).ravel()
    freq = random_tag(code=tags_list[0],
                      freq_limits = freq_limits,
                      step=step,
                      fs=fs, Q1=Q1, Q2=Q2, delta=delta,
                      alpha1=alpha1, alpha2=alpha2, worN=worN, step_noise=step_noise)[2]
    
    f0_pos = np.array(f0_pos)
    
    y = [np.array([int(i) for i in label]) for label in labels]
    y = np.array(y)

    return X, y, freq, f0_pos