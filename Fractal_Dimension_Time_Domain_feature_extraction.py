import os
from turtle import right
import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

output_folder = "Time_Domain_feature_extraction_plots"

os.makedirs(output_folder, exist_ok=True)


def katz_fd(x):
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    n = len(x)
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


def higuchi_fd(x, kmax=10):
    N = len(x)
    L = []
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            idxs = np.arange(1, int(np.floor((N-m)/k)), dtype=np.int32)
            if len(idxs) == 0:
                continue
            Lmk = np.sum(np.abs(x[m + idxs*k] - x[m + k*(idxs-1)]))
            Lmk = (Lmk * (N-1)) / (((N-m)//k) * k)
            Lk.append(Lmk)
        if len(Lk) > 0:
            L.append(np.mean(Lk))
    L = np.array(L)
    lnL = np.log(L)
    lnk = np.log(1.0 / np.arange(1, len(L)+1))
    fd, _ = np.polyfit(lnk, lnL, 1)
    return fd


def petrosian_fd(x):
    diff = np.diff(x)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(x)
    return np.log10(n) / (np.log10(n) + np.log10(n/(n + 0.4*N_delta)))




def extract_fd_features(ppg, fs, kmax=10):
    peaks, _ = signal.find_peaks(ppg, distance=int(0.4*fs))  # systolic peaks
    features = []
    beats = []
    for i in range(len(peaks)-1):
        beat = ppg[peaks[i]:peaks[i+1]]
        if len(beat) < 5:
            continue
        try:
            kfd = katz_fd(beat)
            hfd = higuchi_fd(beat, kmax=kmax)
            pfd = petrosian_fd(beat)
        except Exception:
            continue
        features.append({
            "Beat_Index": i,
            "Katz_FD": kfd,
            "Higuchi_FD": hfd,
            "Petrosian_FD": pfd
        })
        beats.append(beat)
    return pd.DataFrame(features), peaks, beats




if __name__ == "__main__":
    input_folder = "preprocessed_signals"
    output_folder = "Time_Domain_feature_extraction_plots"
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
        print(f"\nProcessing: {segment_name}")
        print(f"Length: {len(filtered_ppg)} samples, Duration: {time_sec[-1]:.2f} sec, fs ≈ {fs:.2f} Hz")

        
        df_features, peaks, beats = extract_fd_features(filtered_ppg, fs=fs, kmax=20)
        print(df_features.head())

        
        out_file = os.path.join(output_folder, f"{segment_name}_FD_features.csv")
        df_features.to_csv(out_file, index=False)
        print(f"Saved features → {out_file}")

        
        plt.figure(figsize=(12, 4))
        plt.plot(time_sec, filtered_ppg, label="Filtered PPG", color="blue")
        plt.plot(time_sec[peaks], filtered_ppg[peaks], "ro", label="Detected Peaks")
        plt.xlabel("Time (sec)", fontsize=10, weight='bold')
        plt.ylabel("Amplitude", fontsize=10, weight='bold')
        plt.title(f"PPG Signal with Detected Peaks ({segment_name})", fontsize=12, weight='bold')
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output_folder, f"{segment_name}_PPG_with_Peaks.png"), dpi=300)
        plt.close()
        
        if beats:
            max_len = max(len(b) for b in beats)
            aligned_beats = [np.pad(b, (0, max_len-len(b)), mode="constant", constant_values=np.nan) for b in beats]

            plt.figure(figsize=(9, 5))
            for i, b in enumerate(aligned_beats):
                if i < 5:
                    row = df_features.iloc[i]
                    label = (f"Beat {row['Beat_Index']}: "
                             f"KFD={row['Katz_FD']:.3f}, "
                             f"HFD={row['Higuchi_FD']:.3f}, "
                             f"PFD={row['Petrosian_FD']:.3f}")
                else:
                    label = None
                plt.plot(b, alpha=0.4, linewidth=3, label=label)
            plt.xlabel("Samples (aligned at systolic peak)", fontsize=10, weight='bold')
            plt.ylabel("Amplitude", fontsize=10, weight='bold')
            plt.title(f"Overlay of {len(beats)} PPG Beats with FD values ({segment_name})", fontsize=12, weight='bold')
            plt.xticks(fontsize=10, weight='bold')
            plt.yticks(fontsize=10, weight='bold')
            plt.legend(loc="upper right", fontsize=8, prop={'weight': 'bold'})
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(output_folder, f"{segment_name}_PPG_Beats_FD_Overlay.png"), dpi=300)
            plt.close()
