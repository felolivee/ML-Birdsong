from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import librosa.display
import matplotlib.pyplot as plt
import itertools
from pydub.utils import db_to_float
from scipy.signal import find_peaks
from scipy.signal import spectrogram
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
from keras.models import Model
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
            view_segment(out_file)

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
def split_matlab(mat_dir, wav_dir, saveWav_dir):
    """
    - Splits the audio files into syllables using the onsets and offsets from the .mat files. 
    - You can change the output file name formatting here if needed.
    
    @param mat_dir: the directory containing the .mat files
    @param wav_dir: the directory containing the .wav files
    @saveWav_dir: directory to save your chunked syllables to
    """
    row = []
    syllable_label = []
    for file in os.listdir(mat_dir):
        i = 0
        if file.endswith('.mat'):
            fileName = file[:-13]
            mat = scipy.io.loadmat(mat_dir + '/' + file)
            for j in range(len(mat["onsets"])):
                t1 = mat["onsets"][j][0]
                t2 = mat["offsets"][j][0]
                label = mat["labels"][0][j]
                fromWavPath = '{}/{}.wav'.format(wav_dir,fileName)
                newAudio = AudioSegment.from_wav(fromWavPath)
                newAudio = newAudio[t1:t2]
                save_to_dir = '{}/{}_syllable{}.wav'.format(saveWav_dir, fileName, j)
                newAudio.export(save_to_dir, format="wav")
                outputCSV_path = '{}/labeled_syllables.csv'.format(saveWav_dir)
                if os.path.exists(fromWavPath): #if wav file exists
                    with open(outputCSV_path, 'a', newline='') as file:
                        row = ['{}_syllable{}'.format(fileName, j), label]
                        syllable_label.append(('{}_syllable{}'.format(fileName, j), label))
                        writer = csv.writer(file)
                        writer.writerow(row)
                    i += 1
                else:
                    print("Wav does not exist: ", fileName)
                    break
    syllable_label.sort()
    return outputCSV_path, syllable_label



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
def view_segment(out_file):
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
def extract_from_waveform(dir, out, syllable_label, saveSpec, mid=False):
    """
    - extracts acoustic features from the spectrogram of the syllables
    - features include wiener entropy, spectral entropy, gravity center, spectral width, max frequency, mfccs, and duration
    @param dir: the directory containing the chunked syllables
    @param out: the output file to save the features in csv format to
    @param labeled_list: list of syllables and their label
    @param saveSpec: if True, the spectrograms images are saved to a folder
    """
    row = []
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            fileName = file[:-4]
            audio_path = os.path.join(dir, file)
            y, sr = librosa.load(audio_path)
            N = len(y) # number of samples
            if N == 0:
                print("{} has 0 samples. File is likely broken and needs manual removal from directory".format(fileName)) #identifies broken files that must be removed from the directory
            if N > 0:
                if (mid == True):
                    thirds = len(y) //3
                    sr_mid = sr
                    mid_y = y[thirds:2*thirds]
                    n_mid = len(mid_y)

                    ft_mid = np.fft.fft(mid_y, n = n_mid)
                    ft_mid = np.abs(ft_mid)
                    normalized_ft_mid = ft_mid / n_mid

                    freq_bins_mid = np.fft.fftfreq(n_mid, 1/sr_mid) 
                    positive_freq_bins_mid = freq_bins_mid[:n_mid// 2]
                    positive_ft_mid = normalized_ft_mid[:n_mid // 2]

                    selected_freq_mid = positive_freq_bins_mid [positive_freq_bins_mid <= 10000]
                    selected_ft_mid = positive_ft_mid [:len(selected_freq_mid)]

                    idx_highest_3_mag = np.argsort(selected_ft_mid)[-3:]
                    freq_3_highest_mag = selected_freq_mid[idx_highest_3_mag]

                    selected_freq = selected_freq_mid
                    selected_ft = selected_ft_mid
                else:
                    # CALCULATE FT: 
                    ft = np.fft.fft(y, n = N) # fft -> frequency vs complex_numbers array 
                    ft = np.abs(ft) # fft -> frequency vs magnitude/amplitude array 
                    normalized_ft = ft / N # ensures that the amplitude from FFT is scalled same as time-domain signal's amplitude

                    f_max = sr / 2 # f_max is the highest frequency that can be analyzed (Nyquist frequency)
                    
                    # COMPUTE FREQUENCY BINS FOR FT SIZE:
                    freq_bins = np.fft.fftfreq(N, 1/sr) 

                    # Select the positve frequencies (first half) since the signal is symmetric 
                    positive_freq_bins = freq_bins[:N // 2]

                    # Postive part of the fft amplitudes
                    positive_ft = normalized_ft[:N // 2]

                    # SELECT DESIRED FREQUENCY RANGE 0 - 10,000 AND FORRESPONDING FT VALUES:
                    selected_freq = positive_freq_bins [positive_freq_bins <= 10000]
                    selected_ft = positive_ft [:len(selected_freq)]

                    # FIND FREQUENCY WITH THE HIGHEST MAGNITUDE/AMPLITUDE (PEAK)
                    idx_highest_mag = np.argmax(selected_ft)
                    freq_highest_mag = selected_freq[idx_highest_mag]
    
                wiener_entropy = librosa.feature.spectral_flatness(y=y, n_fft= N)

                duration = N / sr # duration (s) = samples / sampling_rate

                #could not find a way to compute f0
                f0, _, _ = librosa.pyin(y=y, fmin=10, fmax=6000, sr=sr)
                
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft= N)

                spectral_entropy = ent.spectral_entropy(x = y, sf = sr, method = 'fft')

                gravity_center = librosa.feature.spectral_centroid(y = y, sr = sr,  n_fft= N)

                spectral_width = librosa.feature.spectral_bandwidth(y = y, sr = sr,  n_fft= N)
                for syllable_name, label in syllable_label:
                    #print("fileName: {} vs line[0]: {}".format(fileName, syllable_name))
                    if fileName == syllable_name:
                        corr_label = label
                        if mid == False:
                            row.append([fileName, wiener_entropy.mean(), spectral_entropy, gravity_center.mean(), spectral_width.mean(), freq_highest_mag, mfccs.mean(), duration, corr_label])
                        else:
                            if len(freq_3_highest_mag) == 3 : 
                                row.append([fileName, wiener_entropy.mean(), spectral_entropy, gravity_center.mean(), spectral_width.mean(), freq_3_highest_mag[0], freq_3_highest_mag[1], freq_3_highest_mag[2], mfccs.mean(), duration, corr_label])
                            else :
                                print("{} will not be added to CSV labeled file becuase length of the 3 frequencies with the highest amplitude = {} instead of 3".format(fileName, len(freq_3_highest_mag)))

                if saveSpec != '':
                    spectrogramy = librosa.feature.melspectrogram(y=y, sr=sr, n_fft = N) #Mel spectrogram is a representation of the power spectrum of a sound signal
                    spectrogram_db = librosa.power_to_db(spectrogramy, ref=np.max) #This line converts the power spectrogram (amplitude squared) to decibel (dB) units
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel')
                    plt.axis('off')
                    plt.savefig('{}/{}.png'.format(saveSpec, fileName), bbox_inches='tight', pad_inches=0)
                    plt.close()

    out_Path = '{}/acoustic_features_labled.csv'.format(out)
    write_features(row, out_Path, mid)
    return y, sr, selected_freq, selected_ft, f0, out_Path

# no need to call, built into extract_from_spectrogram
def write_features(data, out_Path, mid):
    file_exists = os.path.isfile(out_Path)
    with open(out_Path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            if mid == False:
                writer.writerow(['File Name', 'Wiener Entropy', 'Spectral Entropy', 'Gravity Center', 'Spectral Width', 'Max Frequency', 'MFCCs', 'Duration', 'Label'])
            else:
                writer.writerow(['File Name', 'Wiener Entropy', 'Spectral Entropy', 'Gravity Center', 'Spectral Width', 'Max Frequency 1', 'Max Frequency 2', 'Max Frequency 3', 'MFCCs', 'Duration', 'Label'])
        for row in data:
            writer.writerow(row)

#no need to call, plots f0
def plot_f0(sr, selected_ft, f0):
    times = librosa.times_like(f0, sr=sr)
    D = librosa.amplitude_to_db(selected_ft, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

#no need to call, plots all peaks
def plot_peaks(selected_ft):
    peaks, _ = find_peaks(selected_ft, height=0)
    plt.plot(selected_ft)
    plt.plot(peaks, selected_ft[peaks], "x")
    plt.show()

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
def filter_labels(labeledCSV_path, filteredCSV_folder, pictures):
    """
    - filters the features csv file to only include the syllables that have non-white spectrograms
    - saves the filtered labels to a new csv file (name can be changed)

    @param labels: the features csv file
    @param pictures: the directory containing the spectrogram images
    """
    filtered_label = []
    with open(labeledCSV_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            file_name = row[0]
            file_N = file_name + '.png'
            if file_N in os.listdir(pictures):
                filtered_label.append(row)
            else:
                pass
    filteredCSV_path2 = "{}/filtered_labeled_syl.csv".format(filteredCSV_folder)
    with open(filteredCSV_path2, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in filtered_label:
            writer.writerow(row)
    return filteredCSV_path2

#call
def CNN(spectrograms_dir, labeled_filteredCSV_path, cnn_features_labeledCSV_folder, vis_featMaps_path, vis_filters):
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
   
    if vis_filters != 0:
        #get filters of first layer
        filters, biases = model1.layers[0].get_weights()
        visualize_filters(filters, vis_filters)

    if vis_featMaps_path != '':
        
        img = load_and_preprocess_image(vis_featMaps_path)
        _ = model1.predict(img)
        
        layer_outputs = []
        layer_names = []
        for layer in model1.layers[:5]: #only the first 5 layers give us 2D feature maps
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
        activation_model = Model(inputs=model1.inputs, outputs=layer_outputs)
        feature_maps = activation_model.predict(img)
        visualize_featureMaps(feature_maps, layer_names)

    data = []

    

    cnn_features_labeledCSV_path = "{}/spectrogram_features.csv".format(cnn_features_labeledCSV_folder)

    # Iterate through each file in the folder
    for file in os.listdir(spectrograms_dir):
        if file.endswith(".png"): 
            img_path = os.path.join(spectrograms_dir, file) # Path to the image file
            img = load_and_preprocess_image(img_path) # Load and preprocess the image       
            features = model1.predict(img) # Extract features using the CNN model
            with open(labeled_filteredCSV_path, 'r') as csvF:
                    fileName = file[:-4]
                    reader = csv.reader(csvF)
                    for line in reader:
                        if fileName == line[0]:
                            label = line[1]
                            data.append([fileName] + features.flatten().tolist() + [label]) # Append the file name and features to the list

    # Convert data to pandas DataFrame
    fet = []
    for i in range(len(data[0]) - 2):
        fet.append("feature{}".format(i + 1))
    columns = ["File Name"] + fet + ["Label"]
    df = pd.DataFrame(data, columns=columns) 

    # Output DataFrame to CSV
    df.to_csv(cnn_features_labeledCSV_path, index=False)

#no need to call, built into CNN method
def load_and_preprocess_image(image_path, target_size=(1000 , 400)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img) # each color channel is zero-centered with respect to the ImageNet dataset without scaling
    return img

def visualize_filters(filters, vis_filters):
    # Normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # Plot the first 6 filters
    n_filters, ix = vis_filters, 1
    for i in range(n_filters):
        # Get the filter
        f = filters[:, :, :, i]
        for j in range(3):
            # Plot each channel separately
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='viridis')
            ix += 1
    plt.show()

def visualize_featureMaps(feature_maps, layer_names):
    for i, layer_name in enumerate(layer_names):
        feature_map = feature_maps[i]

        row_size = feature_map.shape[1]
        col_size = feature_map.shape[2]
        n_features = feature_map.shape[3] #number of feature maps (i.e. 32 for the first layer)
        
        # Plot each feature map individually
        plt.figure(figsize=(20, 8))
        plt.suptitle(layer_name, fontsize=16)
        
        for j in range(n_features):
            plt.subplot(8, n_features, j+1)
            x = feature_map[0, :, :, j]

            # Normalization and scaling
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128

            x = np.clip(x, 0, 255).astype('uint8')
            plt.imshow(x, cmap='viridis')
            plt.axis('off')
            plt.title(f'Feature Map {j+1}')
        
        plt.tight_layout()
        plt.show()
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


