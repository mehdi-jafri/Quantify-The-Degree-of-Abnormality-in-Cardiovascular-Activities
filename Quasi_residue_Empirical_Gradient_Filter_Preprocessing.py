import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import wfdb
import os
from glob import glob


input_folder = "downloaded/P100/p10014354/81739927"
output_folder = "preprocessed_signals"
plot_folder = "preprocessed_plots"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return sig.filtfilt(b, a, signal)

def moving_average(signal, window_samples):
    if window_samples <= 1:
        return signal.copy()
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(signal, kernel, mode='same')

def qregf_baseline_subtraction(signal, fs,
                               bp=(0.5,6.0),
                               baseline_window_s=1.2,
                               residue_window_s=0.25,
                               grad_smooth_s=0.15,
                               alpha=12.0):
    filtered_bp = bandpass_filter(signal, bp[0], bp[1], fs)
    baseline_window = max(1, int(round(baseline_window_s * fs)))
    baseline = moving_average(filtered_bp, baseline_window)
    residue = filtered_bp - baseline
    residue_smooth = moving_average(residue, int(residue_window_s * fs))
    grad = np.abs(np.gradient(residue_smooth)) * fs
    grad_env = moving_average(grad, int(grad_smooth_s * fs))
    grad_env_norm = grad_env - grad_env.min()
    if grad_env_norm.max() > 0:
        grad_env_norm /= grad_env_norm.max()
    weight = 1.0 / (1.0 + alpha * grad_env_norm)
    weight = moving_average(weight, int(grad_smooth_s * fs))
    filtered_residue = residue_smooth * weight
    filtered_signal = filtered_residue + baseline
    return filtered_signal, baseline, residue_smooth


hea_files = sorted(glob(os.path.join(input_folder, "*.hea")))

for hea_file in hea_files:
    segment_name = os.path.splitext(os.path.basename(hea_file))[0]
    rec_path = os.path.join(input_folder, segment_name)
    try:
        signals, fields = wfdb.rdsamp(rec_path)
        fs = fields['fs']
        possible_names = ["PPG", "PLETH", "PULSE", "SPO2", "OXY"]
        ppg_candidates = [i for i, sig_name in enumerate(fields['sig_name'])
                          if any(name in sig_name.upper() for name in possible_names)]
        if not ppg_candidates:
            print(f"❌ No PPG-like channel found in {segment_name}. Skipping.")
            continue
        ppg_channel_idx = ppg_candidates[0]
        ppg_signal = np.nan_to_num(signals[:, ppg_channel_idx])
        t = np.arange(len(ppg_signal)) / fs
        filtered_ppg, baseline, residue = qregf_baseline_subtraction(ppg_signal, fs)
        # Save plot
        plt.figure(figsize=(12,6))
        plt.subplot(3,1,1)
        plt.plot(t, ppg_signal, linewidth=2)
        plt.title("(a) Original PPG signal", fontsize=12, weight='bold')
        plt.ylabel("Amplitude", fontsize=10, weight='bold')
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(t, baseline, linewidth=2)
        plt.title("(b) Estimated movement/baseline", fontsize=12, weight='bold')
        plt.ylabel("Amplitude", fontsize=10, weight='bold')
        plt.grid(True)
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        plt.subplot(3,1,3)
        plt.plot(t, filtered_ppg, linewidth=2)
        plt.title("(c) Cleaned signal (original minus baseline)", fontsize=12, weight='bold')
        plt.xlabel("Time (s)", fontsize=10, weight='bold')
        plt.ylabel("Amplitude", fontsize=10, weight='bold')
        plt.grid(True)
        plt.xticks(fontsize=10, weight='bold')
        plt.yticks(fontsize=10, weight='bold')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(plot_folder, f"Preprocessed_{segment_name}.png"), dpi=300)
        plt.close()
        # Save preprocessed signal
        output_file = os.path.join(output_folder, f"{segment_name}_QREGF.npy")
        data_to_save = np.column_stack((t, filtered_ppg))
        np.save(output_file, data_to_save)
        print(f"✅ Preprocessed signal saved at: {output_file}")
    except Exception as e:
        print(f"❌ Error processing {segment_name}: {e}")
