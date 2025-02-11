import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import spectrogram, welch
from scipy.fftpack import fft
from sklearn.decomposition import PCA
import librosa
import librosa.display

# Load CSV file without header
df = pd.read_csv("selamat.csv", header=None)

# Convert dataframe to numpy array
data = df.to_numpy()

# Streamlit App Title
st.title("Signal Data Analysis")

# Display Data Info
st.subheader("Dataset Overview")
st.write("Shape of the dataset:", df.shape)
st.write("First few rows of data:")
st.write(df.head())

# Compute and display descriptive statistics
st.subheader("Descriptive Statistics")
st.write(pd.DataFrame(data).describe().T)

# Heatmap of correlation
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(data).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Line Plot of Signal Data
st.subheader("Line Plot of Signal Amplitudes")
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(len(data)):
    ax.plot(data[i], label=f"Signal {i+1}", alpha=0.7)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_title("Line Plot of Signal Amplitudes")
ax.legend()
st.pyplot(fig)

# Spectrogram for each signal
fs = 1  # Assuming a sampling frequency of 1 Hz
for i in range(len(data)):
    f, t, Sxx = spectrogram(data[i], fs)
    st.subheader(f"Spectrogram of Signal {i+1}")
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Spectrogram of Signal {i+1}")
    fig.colorbar(cax, label="Power (dB)")
    st.pyplot(fig)

# FFT Analysis
st.subheader("FFT (Fast Fourier Transform) Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(data)):
    fft_vals = np.abs(fft(data[i]))[:len(data[i])//2]
    fft_freqs = np.fft.fftfreq(len(data[i]), d=1/fs)[:len(data[i])//2]
    ax.plot(fft_freqs, fft_vals, label=f"Signal {i+1}")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("FFT Analysis of Signals")
ax.legend()
st.pyplot(fig)

# Histogram Plot
st.subheader("Histogram of Signal Amplitudes")
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(data)):
    ax.hist(data[i], bins=50, alpha=0.7, label=f"Signal {i+1}", density=True)
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title("Histogram of Signal Amplitudes")
ax.legend()
st.pyplot(fig)

# PCA for Dimensionality Reduction
st.subheader("PCA for Dimensionality Reduction")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data.T)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("PCA Projection of Signal Data")
st.pyplot(fig)

# Mel-Frequency Cepstral Coefficients (MFCCs) for Speech Analysis
st.subheader("MFCC Analysis for Speech Data")
fig, ax = plt.subplots(figsize=(10, 6))
mfccs = librosa.feature.mfcc(y=data[0], sr=fs, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time', cmap='coolwarm', ax=ax)
ax.set_ylabel("MFCC Coefficients")
ax.set_xlabel("Time (samples)")
ax.set_title("MFCC of Signal 1")
st.pyplot(fig)

st.write("Analysis Completed! 🚀")
