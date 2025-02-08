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

# Dropdown selection for signal visualization
signal_options = {"Data Row 1": data[0], "Data Row 2": data[1], "Data Row 3": data[2], "Data Semua": data.flatten()}
selected_signal = st.selectbox("Select Signal Data to Visualize:", list(signal_options.keys()))
selected_data = signal_options[selected_signal]

# Line Plot of Signal Data
st.subheader("Line Plot of Selected Signal Amplitudes")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(selected_data, label=selected_signal, alpha=0.7)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Line Plot of {selected_signal}")
ax.legend()
st.pyplot(fig)

# Generate spectrogram for the selected signal
fs = 1  # Assuming a sampling frequency of 1 Hz
f, t, Sxx = spectrogram(selected_data, fs)

# Spectrogram Plot
st.subheader(f"Spectrogram of {selected_signal}")
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title(f"Spectrogram of {selected_signal}")
fig.colorbar(cax, label="Power (dB)")
st.pyplot(fig)

# Histogram Plot
st.subheader(f"Histogram of {selected_signal} Amplitudes")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(selected_data, bins=50, alpha=0.7, label=selected_signal, density=True)
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title(f"Histogram of {selected_signal} Amplitudes")
ax.legend()
st.pyplot(fig)

# Population vs Sample Analysis with Visualization
st.subheader("Population vs Sample Analysis")

def calculate_population_sample_stats(data):
    population_mean = np.mean(data)
    sample_size = int(len(data) * 0.7)
    sample = np.random.choice(data, size=sample_size, replace=False)
    sample_mean = np.mean(sample)
    sample_percentage = (sample_size / len(data)) * 100
    population_percentage = 100 - sample_percentage
    return population_mean, sample_mean, sample, population_percentage, sample_percentage

pop_mean, samp_mean, sample_data, pop_perc, samp_perc = calculate_population_sample_stats(selected_data)

st.write(f"{selected_signal} - Population Mean: {pop_mean}, Sample Mean: {samp_mean}")
st.write(f"{selected_signal} - Population: {pop_perc:.2f}%, Sample: {samp_perc:.2f}%")

# Plot Population vs Sample
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(selected_data, bins=50, alpha=0.5, label=f"Population {selected_signal}", density=True, color='blue')
ax.hist(sample_data, bins=50, alpha=0.7, label=f"Sample {selected_signal}", density=True, color='red')
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title(f"Population vs Sample - {selected_signal}")
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
