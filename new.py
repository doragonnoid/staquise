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

pop_mean_1, samp_mean_1, sample_1, pop_perc_1, samp_perc_1 = calculate_population_sample_stats(data[0])
pop_mean_2, samp_mean_2, sample_2, pop_perc_2, samp_perc_2 = calculate_population_sample_stats(data[1])

st.write(f"Signal 1 - Population Mean: {pop_mean_1}, Sample Mean: {samp_mean_1}")
st.write(f"Signal 1 - Population: {pop_perc_1:.2f}%, Sample: {samp_perc_1:.2f}%")
st.write(f"Signal 2 - Population Mean: {pop_mean_2}, Sample Mean: {samp_mean_2}")
st.write(f"Signal 2 - Population: {pop_perc_2:.2f}%, Sample: {samp_perc_2:.2f}%")

# Plot Population vs Sample
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data[0], bins=50, alpha=0.5, label="Population Signal 1", density=True, color='blue')
ax.hist(sample_1, bins=50, alpha=0.7, label="Sample Signal 1", density=True, color='red')
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title("Population vs Sample - Signal 1")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data[1], bins=50, alpha=0.5, label="Population Signal 2", density=True, color='green')
ax.hist(sample_2, bins=50, alpha=0.7, label="Sample Signal 2", density=True, color='orange')
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title("Population vs Sample - Signal 2")
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

describe this code 
1. Loading and Displaying Data
Loads a CSV file without a header into a Pandas DataFrame.
Converts the DataFrame into a NumPy array for numerical processing.
Displays:
The shape of the dataset.
The first few rows.
Descriptive statistics (mean, std, min, max, etc.).
2. Signal Data Visualization
Line Plot of Signal Amplitudes:

Plots the first two signals (rows) from the dataset.
Shows amplitude variation over time.
Spectrogram of Signal 1:

Computes a spectrogram using scipy.signal.spectrogram().
Uses a sampling frequency (fs) of 1 Hz.
Displays a color-coded frequency-time representation of the first signal.
Histogram of Signal Amplitudes:

Shows amplitude distributions for the first two signals.
Uses 50 bins and normalizes the histogram.
3. Additional Functionalities
Generates a requirements.txt file for deployment.
Lists dependencies:
streamlit, pandas, matplotlib, numpy, and scipy.
Displays a completion message: "Analysis Completed! 🚀".
Possible Issues / Considerations
Assumes at least two signals (rows) in the dataset. If the CSV has fewer, it will cause errors.
Sampling frequency (fs = 1) is arbitrary—it should match the actual data.
Uses only the first two rows for analysis—other signals (if present) are ignored.
