import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import os

def analyze_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    print("Data loaded successfully.\n")
    print(f"First few rows of the dataset:")
    print(df.head())
    print("\n")

    # Basic information about the dataset
    try:
        data_info = df.info()
        print("Data info:\n", data_info, "\n")
        
        statistical_summary = df.describe().T
        print("Statistical summary:\n", statistical_summary)
    except Exception as e:
        print(f"Error during data analysis: {e}")

    # Convert dataframe to numpy array for easier manipulation
    data = df.to_numpy()
    
    print("\nData shape:", data.shape, "\n")

    # Line plot of both rows (assuming they are separate signals)
    plt.figure(figsize=(12, 5))
    plt.plot(data[0], label="Signal 1", alpha=0.7)
    plt.plot(data[1], label="Signal 2", alpha=0.7)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Line Plot of Signal Amplitudes")
    plt.legend()
    plt.show()

    # Generate spectrogram for the first signal
    fs = 1  # Sampling frequency (adjust as needed)

    try:
        f, t, Sxx = spectrogram(data[0], fs)
        print("\nSpectrogram generated successfully.\n")
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram of Signal 1")
        plt.colorbar(label="Power (dB)")
        plt.show()
    except Exception as e:
        print(f"Error generating spectrogram: {e}")

    # Plot histogram of amplitude values
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(data[0], bins=50, alpha=0.7, label="Signal 1", density=True)
        plt.hist(data[1], bins=50, alpha=0.7, label="Signal 2", density=True)
        plt.xlabel("Amplitude")
        plt.ylabel("Density")
        plt.title("Histogram of Signal Amplitudes")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error generating histogram: {e}")

def main():
    # Define output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to your CSV file
    file_path = "D:\codingan pyton\\selamat.csv"
    
    try:
        print("Starting data analysis...")
        analyze_data(file_path)
        
        # Save figures in the output directory
        plt.savefig(os.path.join(output_dir, "line_plot.png"))
        plt.close()
        
        plt.savefig(os.path.join(output_dir, "spectrogram_plot.png"))
        plt.close()
        
        plt.savefig(os.path.join(output_dir, "histogram_plot.png"))
        plt.close()

    except Exception as e:
        print(f"An error occurred during data analysis: {e}")

if __name__ == "__main__":
    main()
