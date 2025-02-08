import pandas as pd

# Load the CSV file
file_path = "D:\codingan pyton/selamat.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
df.info(), df.head()
# Compute descriptive statistics
df.describe().T
import matplotlib.pyplot as plt
import numpy as np

# Convert dataframe to numpy array for easier manipulation
data = df.to_numpy()

# Line plot of both rows (assuming they are separate signals)
plt.figure(figsize=(12, 5))
plt.plot(data[0], label="Signal 1", alpha=0.7)
plt.plot(data[1], label="Signal 2", alpha=0.7)
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Line Plot of Signal Amplitudes")
plt.legend()
plt.show()
from scipy.signal import spectrogram

# Generate spectrogram for the first signal
fs = 1  # Assuming a sampling frequency of 1 Hz (adjust if needed)
f, t, Sxx = spectrogram(data[0], fs)

# Plot spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Spectrogram of Signal 1")
plt.colorbar(label="Power (dB)")
plt.show()
# Plot histogram of amplitude values
plt.figure(figsize=(10, 5))
plt.hist(data[0], bins=50, alpha=0.7, label="Signal 1", density=True)
plt.hist(data[1], bins=50, alpha=0.7, label="Signal 2", density=True)
plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.title("Histogram of Signal Amplitudes")
plt.legend()
plt.show()
