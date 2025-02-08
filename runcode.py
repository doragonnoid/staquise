import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

file_path = "selamat.csv"
df = pd.read_csv(file_path, header=None)

df.info(), df.head(3)
import matplotlib.pyplot as plt

# Plot line chart for the two rows of data
plt.figure(figsize=(12, 5))
plt.plot(df.columns.astype(float), df.iloc[0], label="Signal 1", alpha=0.7)
plt.plot(df.columns.astype(float), df.iloc[1], label="Signal 2", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Time Series Representation of Speech Signal")
plt.legend()
plt.show()
# Rename columns to ensure they are properly formatted as floats
df.columns = pd.RangeIndex(start=0, stop=df.shape[1], step=1)

# Plot line chart again after renaming columns
plt.figure(figsize=(12, 5))
plt.plot(df.columns, df.iloc[0], label="Signal 1", alpha=0.7)
plt.plot(df.columns, df.iloc[1], label="Signal 2", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Time Series Representation of Speech Signal")
plt.legend()
plt.show()
import numpy as np
import scipy.signal

# Generate spectrogram for the first signal
fs = 1  # Assuming 1 unit per sample since no sampling rate info is given
f, t, Sxx = scipy.signal.spectrogram(df.iloc[0].values, fs)

# Plot spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Speech Signal')
plt.colorbar(label="Intensity")
plt.show()
# Calculate descriptive statistics
stats = df.describe().T  # Transpose for better readability

# Select key statistics
stats_summary = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats_summary.head(10)  # Show first 10 columns for reference
