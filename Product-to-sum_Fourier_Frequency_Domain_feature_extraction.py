import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

output_folder = "Frequency_Domain_feature_extraction_plots"

os.makedirs(output_folder, exist_ok=True)

def pts_fourier_features(ppg, fs, n_harmonics=8, plot=False):
    
    N = len(ppg)
    t = np.arange(N) / fs
    features = {}
    
    for k in range(1, n_harmonics+1):
        
        f = k * (fs / N)         
        
        cos_term = np.cos(2*np.pi*f*t)
        sin_term = np.sin(2*np.pi*f*t)

        a_k = (2/N) * np.sum(ppg * cos_term)   
        b_k = (2/N) * np.sum(ppg * sin_term)   
        magnitude = np.sqrt(a_k**2 + b_k**2)
        
        features[f"H{k}_a_cos"] = a_k
        features[f"H{k}_b_sin"] = b_k
        features[f"H{k}_magnitude"] = magnitude
    
    if plot:
        plt.figure(figsize=(10,5))
        mags = [features[f"H{k}_magnitude"] for k in range(1, n_harmonics+1)]
        freqs = [k*(fs/N) for k in range(1, n_harmonics+1)]
        plt.stem(freqs, mags, basefmt=" ")
        plt.title("Product-to-Sum Fourier Magnitudes of PPG", fontsize=12, weight='bold')
        plt.xlabel("Frequency (Hz)", fontsize=10, weight='bold')
        plt.ylabel("Magnitude", fontsize=10, weight='bold')
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{segment_name}_Magnitudes_Of_PPG.png"), dpi=300)
        plt.show()
    
    return features


if __name__ == "__main__":
    input_folder = "preprocessed_signals"
    output_folder = "Frequency_Domain_feature_extraction_plots"
    os.makedirs(output_folder, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith("_QREGF.npy")])
    print(f"Found {len(npy_files)} preprocessed signals.")

    for npy_file in npy_files:
        segment_name = npy_file.replace("_QREGF.npy", "")
        input_file = os.path.join(input_folder, npy_file)

        data = np.load(input_file)
        time_sec = data[:,0]
        filtered_ppg = data[:,1]

        fs = 1 / np.median(np.diff(time_sec))
        print(f"\nProcessing: {segment_name}, fs ≈ {fs:.2f} Hz, duration={time_sec[-1]:.2f}s")

        features = pts_fourier_features(filtered_ppg, fs, n_harmonics=100, plot=True)

        df = pd.DataFrame([features])
        out_file = os.path.join(output_folder, f"{segment_name}_PTS_Fourier_Frequency_Domain_features.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved features → {out_file}")
        print("✅ PTS Fourier Frequency Domain Features:")
        print(df.T.head(20))
