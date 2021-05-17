from scipy.signal import lfilter, freqz
import sounddevice as sd
import numpy as np
import pandas as pd
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os
import librosa, librosa.display
import scipy

def mp3_to_spec(in_path, out_path, save=True):
    """
    Converts the .mp3 file located at in_path into the .jpg spectrogram located at out_path.
    
    Parameter save determines if the spectrogram will be saved or returned.
    """
    sample_path = in_path
    # Load audio file
    audiosample, samp_rate = librosa.load(sample_path, sr=None, mono=True)
    # Perfrom stft on the file
    s = librosa.stft(audiosample)
    # Save spectrogram
    if save == True:
        # Plot spectrogram
        fig, ax = plt.subplots(figsize=(10,5))
        plt.axis('off')
        img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(s), ref=np.max), y_axis='log', x_axis='s', ax=ax, sr=samp_rate)
        # Save spectrogram as a graph without axes or white borders
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(out_path)
        plt.close(fig)
        print("Turned {} into {}".format(in_path, out_path))
    else:
        return s
    
def np_to_spec(np_sample):
    """
    Converts numpy array into spectrogram, used for testing.
    """
    s = librosa.stft(np_sample)
    return s

def butterworth_filter(sample, sr, lowcut, highcut):
    """
    Converts np array of an audio sample into a butterworth filtered sample.
    
    sample: np array of audio sample, created by librosa.load
    sr: sampling rate of audio sample
    lowcut: everything above lowcut will be included in the output
    highcut: everything above highcut will be excluded in the output, has to be lower than sr
    
    returns: Butterworth filtered version of sample
    """
    if not isinstance(sample, np.ndarray):
        raise TypeError("Only able to filter numpy array samples")
    # Calculate butterworth filter
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    # Apply butterworth filter
    y = scipy.signal.lfilter(b, a, sample)
    return y
    
def spectral_subtraction(sample, noise):
    """
    Converts np array of audio sample into a spectral subtracted sample.
    
    sample: np array of audio sample, created by calling librosa.load on mp3 file
    noise: np array of noise sample
    return_samp: decides whether the filtered sample will be returned or the spectrogram
    
    returns: Sample that has had the noise subtracted from it.
    """
    # Loading audio stft
    s= librosa.stft(sample)    # Short-time Fourier transform
    ss= np.abs(s)         # get magnitude
    angle= np.angle(s)    # get phase
    b=np.exp(1.0j* angle) # use this phase information when Inverse Transform
    # Loading noise stft
    ns= librosa.stft(noise) 
    nss= np.abs(ns)
    mns= np.mean(nss, axis=1) # get mean
    # Applying spectral subtraction
    sa= ss - mns.reshape((mns.shape[0],1))  # reshape for broadcast to subtract
    sa0= sa * b  # apply phase information
    y= librosa.istft(sa0) # back to time domain signal
    return y

def add_noise(vocsample, vsr, noise, nsr, sample_duration=2):
    """
    Places the vocalization in the middle of noise to make it a constant size (if possible).
    
    vocsample: np array from individual vocalization
    vsr: vocsample sampling rate
    noise: np array from noise file
    nsr: noise sampling rate
    sample_duration: desired length of sample in seconds.
    
    Returns list of samples, either placed in noise or full 2s segments.
    """
    # Load noise sample and cut it to sample duration
    noise_sample_length = nsr * sample_duration
    noise_start = np.random.randint(0,len(noise) - noise_sample_length)
    noise_sample = noise[noise_start: noise_start + noise_sample_length]
    mean = sample_duration / 2
    if len(vocsample) < noise_sample_length:
        # Place vocalization in middle of noise
        begin = (mean*nsr) - 1/2 * len(vocsample)
        end = (mean*nsr) + 1/2 * len(vocsample)
        noise_sample[int(begin):int(end)] = vocsample
        return [noise_sample]
    else:
        samples = []
        begin = 0
        step = noise_sample_length
        while begin + step <= len(vocsample):
            samples.append(vocsample[begin: begin+step])
            begin += step
            
        finalsamp = vocsample[begin:]
        noise_sample[0 : int(len(finalsamp))] = finalsamp
        samples.append(noise_sample)
        return samples

def filter_audio(vocsample, samp_rate, noisefile_path, audiofilter):
    """
    Converts an np array of a vocalization sample into a filtered vocalization sample.

    vocsample: np array of vocalization created by librosa.load
    audiofilter: None (no filter), ss (spectral subtraction), or bw (butterworth filtering)

    returns: np array of vocalization sample filtered with desirable filter.
    """
    # Filter if neccessary
    if audiofilter == "ss":
        vocsample = spectral_subtraction(vocsample, noisefile_path)
    elif audiofilter == "bw":
        vocsample = butterworth_filter(vocsample, samp_rate, 128, 125000)
    
    return vocsample

def save_spectrogram(voc_stft, vsr, out_path):
    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(10,5))
    plt.axis('off')
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(voc_stft), ref=np.max), y_axis='log', x_axis='s', ax=ax, sr=vsr)
    # Save spectrogram as a graph without axes or white borders
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(out_path)
    plt.close(fig)

def feature_extraction(input_folder_path, output_folder_path, noise_path, audiofilter=None):
    # Iterate through class folders
    noise, nsr = librosa.load(noise_path, sr=None, mono=True)
    for voctypes in sorted(os.listdir(input_folder_path)):
        type_path = "{}/{}".format(input_folder_path, voctypes)
        for voc in os.listdir(type_path):
            # Load vocalzation
            voc_path = type_path + "/" + voc
            vocsample, vsr = librosa.load(voc_path, sr=None, mono=True)
            # Filter audio
            vocsample = filter_audio(vocsample, vsr, noise_path, audiofilter)
            # Place audio in center of noise
            voc_noise_samples = add_noise(vocsample, vsr, noise, nsr, sample_duration=2)
            # Turn noise added audio samples to spectrograms
            for i in range(len(voc_noise_samples)):
                voc_noise_samples[i] = librosa.stft(voc_noise_samples[i])
                spec_folder = "{}/{}".format(output_folder_path, voctypes)
                spec_path = spec_folder + "/" + voc[:-4] + "_" + str(i) + ".jpg"
                print(spec_path)
                save_spectrogram(voc_noise_samples[i], vsr, spec_path)
            

feature_extraction("../data/audio", "../data/transforms", "../data/audio/noise/noise_0.wav")