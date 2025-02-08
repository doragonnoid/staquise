import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

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

# Line Plot of Signal Data
st.subheader("Line Plot of Signal Amplitudes")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data[0], label="Signal 1", alpha=0.7)
ax.plot(data[1], label="Signal 2", alpha=0.7)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_title("Line Plot of Signal Amplitudes")
ax.legend()
st.pyplot(fig)

# Generate spectrogram for the first signal
fs = 1  # Assuming a sampling frequency of 1 Hz
f, t, Sxx = spectrogram(data[0], fs)

# Spectrogram Plot
st.subheader("Spectrogram of Signal 1")
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title("Spectrogram of Signal 1")
fig.colorbar(cax, label="Power (dB)")
st.pyplot(fig)

# Histogram Plot
st.subheader("Histogram of Signal Amplitudes")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data[0], bins=50, alpha=0.7, label="Signal 1", density=True)
ax.hist(data[1], bins=50, alpha=0.7, label="Signal 2", density=True)
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title("Histogram of Signal Amplitudes")
ax.legend()
st.pyplot(fig)

st.write("Analysis Completed! 🚀")

# Create requirements.txt file
requirements = """
streamlit
pandas
matplotlib
numpy
scipy
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

st.write("Generated requirements.txt for deployment.")
