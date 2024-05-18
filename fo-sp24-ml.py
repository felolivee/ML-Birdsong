from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import librosa.display
import matplotlib.pyplot as plt
import itertools
from pydub.utils import db_to_float
import antropy as ent
import numpy as np
import os
import csv
import pandas as pd
import os
import scipy.io
from PIL import Image
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# first three methods are for splitting audio files into chunks
# if you alaready the onsets and offsets of the syllables from matlab, skip to split_matlab in the second section

############################################################################################################################################################################

#SEGMENTATION WITHOUT .NOTMAT FILE:

#do not need to call, it is built into the silence_pick method
def detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges

#do not need to call
def detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step=1):
    """
    Returns a list of all nonsilent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_silent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
    len_seg = len(audio_segment)

    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges

#call this method to determine the min_silence_len and silence_thresh that best fits your songfile
def silence_pick(sound_file, min_silence_len, silence_thresh,):

    # detect_silence returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    silent_ranges = detect_silence(sound_file, min_silence_len, silence_thresh) 

    # Convert the silent ranges to segment start and end times
    segment_start_times = [silent_range[0] / 1000 for silent_range in silent_ranges]
    segment_end_times = [silent_range[1] / 1000 for silent_range in silent_ranges]

    #plots silence ranges on the audio waveform
    #use this plot to hepl you determine the min_silence_len and silence_thresh
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(sound_file) / 1000, len(sound_file.get_array_of_samples())), sound_file.get_array_of_samples())
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Audio Waveform")

    for start_time, end_time in zip(segment_start_times, segment_end_times):
        plt.axvline(x=start_time, color='black', linestyle='--')
        plt.axvline(x=end_time, color='black', linestyle='--')
        plt.fill_betweenx([-1, 1], start_time, end_time, color='black', alpha=0.9)

    plt.show()
    return sound_file, min_silence_len, silence_thresh

#do not need to call, built into the slice_audio method
#can change output file name formatting here if needed
def slice(sound_file, min_silence_len, silence_thresh, name, show = False):
    audio_chunks = split_on_silence(sound_file, min_silence_len, silence_thresh)
    for i, chunk in enumerate(audio_chunks):
        out_file = "chunked_2/{}_syllable{}.wav".format(name, i)
        chunk.export(out_file, format="wav")
        if show == True:
            view_segments(out_file)

#call this method to slice the audio file into syllables with the min_silence_len and silence_thresh you have chosen
def slice_audio(dir, min_silence_len, silence_thresh):
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            sound_file = AudioSegment.from_wav(dir + '/' + file)
            fileName = file[:-4]
            silence_pick(sound_file, min_silence_len, silence_thresh)
            slice(sound_file, min_silence_len, silence_thresh, fileName)

############################################################################################################################################################################

#SEGMENTATION USING .NOTMAT FILE:

#call
def split_matlab(mat_dir,wav_dir):
    """
    - Splits the audio files into syllables using the onsets and offsets from the .mat files. 
    - You can change the output file name formatting here if needed.
    
    @param mat_dir: the directory containing the .mat files
    @param wav_dir: the directory containing the .wav files
    
    """
    for file in os.listdir(mat_dir):
        i = 0
        if file.endswith('.mat'):
            fileName = file[:-13]
            mat = scipy.io.loadmat(mat_dir + '/' + file)
            for j in range(len(mat["onsets"])):
                t1 = mat["onsets"][j][0]
                t2 = mat["offsets"][j][0]
                label = mat["labels"][0][j]
                fromWavPath = wav_dir + '/' + fileName + '.wav'
                outputCSV = 'labels_11.csv'
                if os.path.exists(fromWavPath):
                    with open(outputCSV, 'a', newline='') as file:
                        row = ['{}_syllable{}'.format(fileName, j), label]
                        writer = csv.writer(file)
                        writer.writerow(row)
                    i += 1
                else:
                    print("Wav does not exist: ", fileName)
                    break

#no need to call (experimented with windowing and fft to determine the frequency with the highest amplitude)
def windowing(sound_file):
    y, sr = librosa.load(sound_file)
    samples = librosa.get_duration(y=y, sr=sr)*sr
    frame_size = samples
    hann_win = np.hanning(frame_size)
    y_windowed = y * hann_win

    ft = np.fft.fft(y)
    magnitude_spectrum = np.abs(ft)
    print(np.argmax(magnitude_spectrum))
    plt.figure(figsize=(18, 5))
    frequency = np.linspace(0, sr, magnitude_spectrum.shape[0])
    print(frequency[np.argmax(magnitude_spectrum)])
    num_frequency_bins = magnitude_spectrum.shape[0]
    plt.plot(frequency[0:(10000 *num_frequency_bins)// sr], magnitude_spectrum[:(10000*num_frequency_bins)// sr])
    plt.xlabel('Frequency (Hz)')
    plt.title('Magnitude Spectrum')
    plt.show()

    
    ft1 = np.fft.fft(y_windowed)
    magnitude_spectrum1 = np.abs(ft1)
    plt.figure(figsize=(18, 5))
    frequency1 = np.linspace(0, sr, magnitude_spectrum1.shape[0])
    num_frequency_bins1 = magnitude_spectrum1.shape[0]
    pot_f0 = magnitude_spectrum1[magnitude_spectrum1 < 1000]
    max_amplitude_freq = frequency1[np.argmax(pot_f0)]

    print("Frequency with highest amplitude:", max_amplitude_freq)
    plt.plot(frequency1[:num_frequency_bins1], magnitude_spectrum1[:num_frequency_bins1])
    plt.xlabel('Frequency (Hz)')
    plt.title('Magnitude Spectrum')
    plt.show()

#no need to call (visualization of a syllable in waveform)
def view_segments(out_file):
    y, sr = librosa.load(out_file)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color = 'blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('wh13wh95 syllable')
    plt.tight_layout()
    plt.show()

############################################################################################################################################################################

#1) FEATURE EXTRACTION USING WAVEFORM

#call
def extract_from_spectrogram(dir, out, viewNsave = False):
    """
    - extracts acoustic features from the spectrogram of the syllables
    - features include wiener entropy, spectral entropy, gravity center, spectral width, max frequency, mfccs, and duration
    @param dir: the directory containing the chunked syllables
    @param out: the output file to save the features in csv format to
    @param viewNsave: if True, the spectrograms are displayed and images are saved to a folder
    """
    n_fft = 0
    row = []
    isMat = (dir == 'chunked_mat')
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            fileName = file[:-4]
            audio_path = os.path.join(dir, file)
            y, sr = librosa.load(audio_path)
            spectral_entropy = ent.spectral_entropy(x = y, sf = sr, method = 'fft')
            if len(y) == 0:
                print("FILE IS {}".format(fileName)) #identifies broken files that must be removed from the directory
            if isMat and len(y) > 0:
                samples = int(librosa.get_duration(y=y, sr=sr)*sr) #number of samples in the audio file
                ft = np.fft.fft(y) #fourier transform of the audio file
                frequency = np.linspace(0, sr, samples) #frequency range
                interested_frequencies = frequency[0:(10000 * samples)// sr] #interested frequency range to find the frequency with the highest amplitude (range is ajustable)
                magnitude_spectrum = np.abs(ft) ** 2 #power spectrum of the audio file. note: **2 was not there in the original code
                indexed_magnitudes = magnitude_spectrum[:len(interested_frequencies)] #magnitude spectrum just of the interested frequency range
                num_frequency_bins = indexed_magnitudes.shape[0] #number of frequency bins
                max_freq = interested_frequencies[np.argmax(indexed_magnitudes)] #frequency with the highest amplitude

                #optimal n_fft for the spectrogram
                if num_frequency_bins > 1024:
                    n_fft = 2048
                elif num_frequency_bins > 512:
                    n_fft = 1024
                elif num_frequency_bins > 256:
                    n_fft = 512
                                    
                #Compute the spectrogram
                spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

                wiener_entropy = librosa.feature.spectral_flatness(y=y, n_fft= n_fft)

                #could not find a way to compute f0
                #f0, e, f = librosa.pyin(y=y, fmin=10, fmax=6000, sr=sr, frame_length= int(samples), fill_na= None)

                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft= n_fft)

                gravity_center = librosa.feature.spectral_centroid(y = y, sr = sr,  n_fft= n_fft)

                spectral_width = librosa.feature.spectral_bandwidth(y = y, sr = sr,  n_fft= n_fft)

                duration = num_frequency_bins

                row.append([fileName, wiener_entropy.mean(), spectral_entropy, gravity_center.mean(), spectral_width.mean(), max_freq, mfccs.mean(), duration])
                if viewNsave == True:
                    # Convert to decibels
                    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                    view_spectrograms(spectrogram_db, fileName)
    write_features(row, out) 

# no need to call, built into extract_from_spectrogram
def write_features(data, out):
    file_exists = os.path.isfile(out)
    with open(out, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['File Name', 'Wiener Entropy', 'Spectral Entropy', 'Gravity Center', 'Spectral Width', 'Max Frequency', 'MFCCs', 'Duration', 'Label'])
        for row in data:
            writer.writerow(row)

#no need to call, just a visualization of the spectrogram in decibles (also saves the images to a folder)            
def view_spectrograms(spectrogram_db, file_name):
    # Plot the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(file_name)
            plt.show()
            # Save the spectrogram image to a folder
            plt.savefig('spectrograms/{}.png'.format(file_name))
            plt.close()


#2) FEATURE EXTRACTION USING CNN

#call
def remove_white_spec(dir):
    """
    - removes white images from the spectrograms folder

    @param dir: the directory containing the spectrogram images
    """
    for file in os.listdir(dir):
        if file.endswith(".png"):
            image_path = os.path.join(dir, file)
            image = Image.open(image_path)
            center_pixel = image.getpixel((image.width // 2, image.height // 2))
            if center_pixel == (255, 255, 255, 255):
                print("Removing white image: ", file)
                os.remove(image_path)

#call
def filter_labels(labels, pictures):
    """
    - filters the features csv file to only include the syllables that have non-white spectrograms
    - saves the filtered labels to a new csv file (name can be changed)

    @param labels: the features csv file
    @param pictures: the directory containing the spectrogram images
    """
    filtered_label = []
    with open(labels, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            file_name = row[0]
            file_N = file_name + '.png'
            if file_N in os.listdir(pictures):
                filtered_label.append(row)
            else:
                pass
    with open('labels_filtered16.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in filtered_label:
            writer.writerow(row)

#no need to call
def mel_spec(audio_file):
    y, sr = librosa.load(audio_file)

    # Step 1: Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)

    # Step 2: Apply Mel filter bank
    n_fft = 2048  # Length of the FFT window
    n_mels = 128  # Number of Mel bands
    mel = librosa.filters.mel(sr = sr, n_fft = n_fft, n_mels = n_mels)

    mel_spec = np.dot(mel, np.abs(D)**2)

    # Step 3: Compute MFCCs using Discrete Cosine Transform (DCT)
    n_mfcc = 15  # Number of MFCC coefficients
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    # Visualize the Mel spectrogram and MFCCs
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.power_to_db(mel_spec), sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    plt.tight_layout()
    plt.show()

#call
def CNN(spectrgrams_dir, output_dir):
    """
    - Extracts features from the spectrograms using a CNN model
    - Saves the features to a CSV file (name can be changed)
    @param spectrgrams_dir: the directory containing the spectrogram images
    @param output_dir: the output file to save the features in csv format to

    """
    # Define the CNN model to reduce the dimensionality of the spectrogram images
    model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 400, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu')
    ])

    model1.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # Compile the model

    folder_path = "spectrograms" # Path to the folder containing the spectrogram images

    data = [] # List to store file names and extracted features


    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"): 
            img_path = os.path.join(folder_path, filename) # Path to the image file
            img = load_and_preprocess_image(img_path) # Load and preprocess the image       
            features = model1.predict(img) # Extract features using the CNN model
            data.append([filename] + features.flatten().tolist()) # Append the file name and features to the list

    # Convert data to pandas DataFrame
    fet = []
    for i in range(len(data[0]) - 1):
        fet.append("feature{}".format(i + 1))
    columns = ["filename"] + fet
    df = pd.DataFrame(data, columns=columns) 

    # Output DataFrame to CSV
    output_csv_path = "spectrogram_features.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"Features extracted and saved to {output_csv_path}.")

#no need to call, built into CNN method
def load_and_preprocess_image(image_path, target_size=(1000 , 400)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

############################################################################################################################################################################

#RESULTS:
# 1. Use Organge to load the csv files and perform the classification
# 2. Use metrics in Orange to find different metrics such as precision, recall, f1 score, etc.
# 3. Use the following method to calculate the accuracy. requires adjustment to fit your needs 

#custom way to find results, adjust to your needs
def calc_results(file):
    c1 = {}
    c2 = {}
    c3 = {}
    c4 = {}
    c5 = {}
    c6 = {}
    c7 = {}
    c8 = {}
    c9 = {}
    letters = {}
    ct = 0
    with open(file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[2] == 'C1':
                if row[1] in c1.keys():
                    c1[row[1]] +=1
                else:
                    c1[row[1]] = 1
            elif row[2] == 'C2':
                if row[1] in c2.keys():
                    c2[row[1]] +=1
                else:
                    c2[row[1]] = 1
            elif row[2] == 'C3':
                if row[1] in c3.keys():
                    c3[row[1]] +=1
                else:
                    c3[row[1]] = 1
            elif row[2] == 'C4':
                if row[1] in c4.keys():
                    c4[row[1]] +=1
                else:
                    c4[row[1]] = 1
            elif row[2] == 'C5':
                if row[1] in c5.keys():
                    c5[row[1]] +=1
                else:
                    c5[row[1]] = 1
            elif row[2] == 'C6':
                if row[1] in c6.keys():
                    c6[row[1]] +=1
                else:
                    c6[row[1]] = 1
            elif row[2] == 'C7':
                if row[1] in c7.keys():
                    c7[row[1]] +=1
                else:
                    c7[row[1]] = 1
            elif row[2] == 'C8':
                if row[1] in c8.keys():
                    c8[row[1]] +=1
                else:
                    c8[row[1]] = 1
            elif row[2] == 'C9':
                if row[1] in c9.keys():
                    c9[row[1]] +=1
                else:
                    c9[row[1]] = 1
            if row[1] in letters.keys():
                letters[row[1]] += 1
            elif row[1] != 'Labels':
                letters[row[1]] = 1
            # if row[1] == 'C1' and row[2] == 'C1':
            #     ct += 1
            # elif row[1] == 'C2' and row[2] == 'C2':
            #     ct += 1
            # elif row[1] == 'C3' and row[2] == 'C3':
            #     ct += 1
            # elif row[1] == 'C4' and row[2] == 'C4':
            #     ct += 1
            # elif row[1] == 'C5' and row[2] == 'C5':
            #     ct += 1
            # elif row[1] == 'C6' and row[2] == 'C6':
            #     ct += 1
            # elif row[1] == 'C7' and row[2] == 'C7':
            #     ct += 1
            # elif row[1] == 'C8' and row[2] == 'C8':
            #     ct += 1
            # elif row[1] == 'C9' and row[2] == 'C9':
            #     ct += 1
            # elif row[1] == 'C10' and row[2] == 'C10':
            #     ct += 1

    for key, val in letters.items():
        c1[key] = round((c1.get(key, 0)/val) * 100,2)
        c2[key] = round((c2.get(key, 0)/val) * 100,2)
        c3[key] = round((c3.get(key, 0)/val) * 100,2)
        c4[key] = round((c4.get(key, 0)/val) * 100,2)
        c5[key] = round((c5.get(key, 0)/val) * 100,2)
        c6[key] = round((c6.get(key, 0)/val) * 100,2)
        c7[key] = round((c7.get(key, 0)/val) * 100,2)
        c8[key] = round((c8.get(key, 0)/val) * 100,2)
        c9[key] = round((c9.get(key, 0)/val) * 100,2)    

    top_c1 = sorted(c1.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c2 = sorted(c2.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c3 = sorted(c3.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c4 = sorted(c4.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c5 = sorted(c5.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c6 = sorted(c6.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c7 = sorted(c7.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c8 = sorted(c8.items(), key=lambda x: x[1], reverse=True)[:3]
    top_c9 = sorted(c9.items(), key=lambda x: x[1], reverse=True)[:3]

    print("Top 3 elements in c1:", top_c1) 
    print("Top 3 elements in c2:", top_c2) 
    print("Top 3 elements in c3:", top_c3) 
    print("Top 3 elements in c4:", top_c4) 
    print("Top 3 elements in c5:", top_c5) 
    print("Top 3 elements in c6:", top_c6) 
    print("Top 3 elements in c7:", top_c7)  
    print("Top 3 elements in c8:", top_c8) 
    print("Top 3 elements in c9:", top_c9) 

#plots pca, adjust to your parameters
def plotPCA(features):
    import matplotlib.pyplot as plt

    # Load the features from the CSV file
    features = pd.read_csv(features)

    # Extract the columns for pc1, pc2, and pc3
    pc1 = features['PC1']
    pc2 = features['PC2']
    pc3 = features['PC3']
    cluster = features['Cluster']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the color of data points based on the cluster
    colors = {'C1': 'red', 'C2': 'green', 'C3': 'black', 'C4': 'yellow', 'C5': 'purple', 'C6': 'blue', 'C7': 'orange', 'C8': 'pink'}
    for cluster_name, color in colors.items():
        indices = cluster[cluster == cluster_name].index[:1000]  
        ax.scatter(pc1[indices], pc2[indices], pc3[indices], c=color, alpha=0.5)  
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.show()


