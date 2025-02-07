import torch
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.fft import fft



model = torch.load("cnnmodel_fine.pth")
model.load_state_dict(torch.load("cnnmodel_fine.pth"))
model.eval()

csv_file_path = '/content/drive/MyDrive/prometheus/eegtemp512.csv'
df = pd.read_csv(csv_file_path, skiprows=2)
volt = df['Voltage(uV)']
raw_eeg = volt.values

segments = []
def segment_data(vec, segments):
    num = len(vec) // 3
    for i in range(2):
        segments.append(vec[i*num:(i+1)*num])
    segments.append(vec[2*num:])
    return segments

segments = segment_data(raw_eeg, segments)
print([len(seg) for seg in segments])


### Use when you have two csv files, which is eeg of 60seconds.
#segments = segment_data(raw_eeg1, segments)

def preprocess_eeg(data, original_sampling_rate, target_sampling_rate, lowcut, highcut):
    nyquist = 0.5 * original_sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    num_samples = int(len(filtered_data) * (target_sampling_rate / original_sampling_rate))
    downsampled_data = resample(filtered_data, num_samples)
    return downsampled_data

def create_eeg_dataset(raw_data, segment_size=200):
    original_sampling_rate = 1000
    target_sampling_rate = 200
    lowcut = 0.5
    highcut = 75

    preprocessed_data = preprocess_eeg(raw_data.flatten(),
                                     original_sampling_rate,
                                     target_sampling_rate,
                                     lowcut,
                                     highcut)

    usable_length = (len(preprocessed_data) // segment_size) * segment_size
    trimmed_data = preprocessed_data[:usable_length]

    num_segments = len(trimmed_data) // segment_size
    dataset = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment = trimmed_data[start_idx:end_idx]

        fourier_segment = np.abs(fft(segment))[:segment_size]

        formatted_segment = fourier_segment.reshape(1, segment_size)
        dataset.append(formatted_segment)

    dataset = np.array(dataset)
    return dataset


rawdataset = []
for i in range(len(segments)):
  rawdataset.append(create_eeg_dataset(segments[i]))
  print(rawdataset[i].shape)


out = []
for i in range(len(rawdataset)):
  out.append(model(torch.from_numpy(rawdataset[i]).float().to('cuda')))

sco = [tensor[:, 0].mean().item() for tensor in out]
print(sco)

max_idx = sco.index(max(sco))

character = ['강대호', '딱지남', '박민수', '성기훈', '박정배', '타노스']

print(f"당신이 선호하는 오징어게임 캐릭터는 {character[max_idx]}입니다.")

