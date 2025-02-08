import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Pastikan file berada di direktori yang benar
file_path = os.path.join(os.getcwd(), "selamat.csv")

# Periksa apakah file ada
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' tidak ditemukan.")
else:
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Tampilkan informasi dasar
    print(df.info())
    print(df.head())

    # Statistik deskriptif
    print(df.describe().T)

    # Konversi DataFrame ke NumPy
    data = df.to_numpy()

    # Pastikan setidaknya ada dua baris untuk visualisasi
    if data.shape[0] < 2:
        print("Error: Data tidak memiliki cukup baris untuk visualisasi.")
    else:
        # Line plot dari dua sinyal pertama
        plt.figure(figsize=(12, 5))
        plt.plot(data[0], label="Signal 1", alpha=0.7)
        plt.plot(data[1], label="Signal 2", alpha=0.7)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.title("Line Plot of Signal Amplitudes")
        plt.legend()
        plt.show()

        # Spectrogram untuk sinyal pertama
        fs = 100  # Sesuaikan dengan sampling rate yang sesuai
        f, t, Sxx = spectrogram(data[0], fs)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram of Signal 1")
        plt.colorbar(label="Power (dB)")
        plt.show()

        # Histogram amplitudo sinyal
        plt.figure(figsize=(10, 5))
        plt.hist(data[0], bins=50, alpha=0.7, label="Signal 1", density=True)
        plt.hist(data[1], bins=50, alpha=0.7, label="Signal 2", density=True)
        plt.xlabel("Amplitude")
        plt.ylabel("Density")
        plt.title("Histogram of Signal Amplitudes")
        plt.legend()
        plt.show()
