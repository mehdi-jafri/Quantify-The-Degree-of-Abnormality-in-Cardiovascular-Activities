import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


win_size = 256        # samples per window
step = 128            # step between windows (50% overlap)
latent_dim = 12       # latent dimension for VAE
hidden_dims = [128,64] # hidden layers
epochs = 30           # training epochs
default_fs = 62.47    # default sampling frequency if not in time column
min_window = 16       # minimum window size for short segments


def sliding_windows(signal: np.ndarray, win_size: int, step: int):
    if signal.ndim == 1:
        L = len(signal)
        windows = [signal[i:i+win_size] for i in range(0, L - win_size + 1, step)]
        return np.stack(windows) if windows else np.empty((0, win_size), dtype=signal.dtype)
    elif signal.ndim == 2:
        all_w = []
        for s in signal:
            w = sliding_windows(s, win_size, step)
            if w.size:
                all_w.append(w)
        return np.vstack(all_w) if all_w else np.empty((0, win_size))
    else:
        raise ValueError("signal must be 1D or 2D array")

def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    N = x.size
    Lk = np.zeros(kmax)
    for k in range(1, kmax+1):
        Lm = np.zeros(k)
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                Lm[m] = 0
                continue
            diffs = np.abs(np.diff(x[idxs]))
            norm = (N-1)/((len(idxs)-1)*k)
            Lm[m] = (np.sum(diffs) * norm)/k
        Lk[k-1] = Lm.mean()
    nonzero = Lk>0
    if nonzero.sum()<2:
        return 1.0
    lnL = np.log(Lk[nonzero])
    lnk = np.log(1.0/np.arange(1,kmax+1)[nonzero])
    slope,_ = np.polyfit(lnk, lnL,1)
    return float(slope)

def product_to_sum_fourier_features(x: np.ndarray, fs: float=1.0, n_peaks: int=8) -> np.ndarray:
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))
    mag = np.abs(X)
    idx_sorted = np.argsort(mag)[::-1]
    top_idx = idx_sorted[:n_peaks]
    if len(top_idx) < 2:
        top_idx = np.arange(min(n_peaks, mag.size))
    pair_feats = []
    max_bin = mag.size - 1
    for i in range(len(top_idx)):
        for j in range(i+1, len(top_idx)):
            a,b = top_idx[i], top_idx[j]
            sum_idx = min(a+b,max_bin)
            diff_idx = abs(a-b)
            val = 0.5*(mag[sum_idx]+mag[diff_idx])
            pair_feats.append(val)
    top_mags = mag[top_idx]
    ratios = [top_mags[0]/(top_mags[i]+1e-8) for i in range(1,len(top_mags))]
    delays = [1,2,4,8]
    td_feats = []
    for d in delays:
        if d>=N:
            td_feats.extend([0.0,0.0])
            continue
        prod = x[:-d]*x[d:]
        td_feats.append(prod.mean())
        td_feats.append(prod.std())
    feats = np.concatenate([[higuchi_fd(x)], [x.mean(), x.std(), np.median(x), np.max(x)-np.min(x)], top_mags, ratios, pair_feats, td_feats])
    return feats.astype(float)

def extract_features(ppg_windows: np.ndarray, fs: float = 100.0, n_fft_peaks: int = 8, kmax: int = 10) -> np.ndarray:
    all_feats = []
    for w in ppg_windows:
        w_z = (w - w.mean())/(w.std()+1e-8)
        feats = product_to_sum_fourier_features(w_z, fs,n_fft_peaks)
        feats = np.concatenate([[higuchi_fd(w_z,kmax)], [w.mean(),w.std(),np.median(w), np.max(w)-np.min(w)], feats])
        all_feats.append(feats)
    return np.vstack(all_feats)


class BatchNormVAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims=[128,64]
        enc=[]
        prev=input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(prev,h))
            enc.append(nn.BatchNorm1d(h))
            enc.append(nn.ReLU(inplace=True))
            prev=h
        self.encoder=nn.Sequential(*enc)
        self.fc_mu=nn.Linear(prev,latent_dim)
        self.fc_logvar=nn.Linear(prev,latent_dim)
        dec=[]
        prev=latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(prev,h))
            dec.append(nn.BatchNorm1d(h))
            dec.append(nn.ReLU(inplace=True))
            prev=h
        dec.append(nn.Linear(prev,input_dim))
        self.decoder=nn.Sequential(*dec)

    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,x):
        h=self.encoder(x)
        mu=self.fc_mu(h)
        logvar=self.fc_logvar(h)
        z=self.reparameterize(mu,logvar)
        x_recon=self.decoder(z)
        return x_recon, mu, logvar

def loss_function(recon_x,x,mu,logvar,beta=1.0):
    recon_loss=nn.functional.mse_loss(recon_x,x,reduction='mean')
    kld=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon_loss+beta*kld,recon_loss.detach(),kld.detach()

def train_vae(model,dataloader,epochs=50,lr=1e-3,device="cpu",beta=1.0):
    model.to(device)
    opt=optim.Adam(model.parameters(),lr=lr)
    for ep in range(1,epochs+1):
        model.train()
        tot_loss,tot_recon,tot_kld,n=0,0,0,0
        for batch in dataloader:
            xb=batch[0].to(device)
            opt.zero_grad()
            recon,mu,logvar=model(xb)
            loss,recon_l,kld_l=loss_function(recon,xb,mu,logvar,beta)
            loss.backward()
            opt.step()
            bs=xb.shape[0]
            tot_loss+=loss.item()*bs
            tot_recon+=recon_l.item()*bs
            tot_kld+=kld_l.item()*bs
            n+=bs
        print(f"Epoch {ep}/{epochs} | Loss: {tot_loss/n:.6f} Recon: {tot_recon/n:.6f} KLD: {tot_kld/n:.6f}")
    return model


def adaptive_threshold(errors: np.ndarray, k: float=3.0) -> float:
    mean,std = float(np.mean(errors)),float(np.std(errors))
    return mean + k*std

def anomaly_score(errors: np.ndarray, threshold: float) -> np.ndarray:
    std = np.std(errors)+1e-8
    z = (errors-threshold)/std
    scores = 1/(1+np.exp(-z))
    return scores

def detect_anomalies(model: nn.Module, features: np.ndarray, device="cpu", k=3.0):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features,dtype=torch.float32).to(device)
        recon,mu,logvar = model(x)
        errs = torch.mean((recon-x)**2, dim=1).cpu().numpy()
    thr = adaptive_threshold(errs,k=k)
    scores = anomaly_score(errs,thr)
    is_anom = scores>0.5
    return errs, thr, scores, is_anom


data_dir = "preprocessed_signals"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

all_files = sorted(glob(os.path.join(data_dir,"*_QREGF.npy")))
segments={}
for f in all_files:
    seg_name=os.path.basename(f).replace("_QREGF.npy","")
    sig=np.load(f)
    segments[seg_name]=sig
    print(f"Loaded {seg_name}, shape {sig.shape}")


all_features=[]
segment_indices={}
start_idx=0
for seg_name,sig in segments.items():
    if sig.ndim == 2 and sig.shape[1] >= 2:
        time_sec = sig[:,0]
        fs = 1 / np.median(np.diff(time_sec))
        windows = sliding_windows(sig[:,1], win_size, step)
    else:
        fs = default_fs
        windows = sliding_windows(sig, win_size, step)
    
    
    if windows.shape[0] == 0:
        
        if len(sig) >= min_window:
            adj_win = len(sig)//2
            adj_step = max(1, adj_win//2)
            print(f"Adjusting window for short segment {seg_name}: win_size={adj_win}, step={adj_step}")
            windows = sliding_windows(sig[:,1] if sig.ndim==2 else sig, adj_win, adj_step)
        if windows.shape[0] == 0:
            print(f"Skipping segment {seg_name}, too short even after adjustment (length={len(sig)})")
            continue
    
    feats = extract_features(windows, fs=fs, n_fft_peaks=8, kmax=10)
    all_features.append(feats)
    end_idx = start_idx + len(feats)
    segment_indices[seg_name] = (start_idx, end_idx)
    start_idx = end_idx

if len(all_features) == 0:
    raise ValueError("No valid segments found for feature extraction!")

all_features=np.vstack(all_features)
print("Feature matrix shape:",all_features.shape)


mu = all_features.mean(axis=0)
sigma = all_features.std(axis=0)+1e-8
features_z=(all_features-mu)/sigma


train_ds = TensorDataset(torch.tensor(features_z,dtype=torch.float32))
train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
input_dim = features_z.shape[1]
model = BatchNormVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
model = train_vae(model,train_loader,epochs=epochs,lr=1e-3,device="cpu",beta=1.0)


errors, thr, scores, is_anom = detect_anomalies(model, features_z, device="cpu", k=3.0)
results={}
for seg_name,(s,e) in segment_indices.items():
    seg_errors=errors[s:e]
    seg_scores=scores[s:e]
    seg_anom=is_anom[s:e]
    results[seg_name]={
        "n_windows": e-s,
        "n_anomalies": int(seg_anom.sum()),
        "mean_error": float(seg_errors.mean()),
        "max_score": float(seg_scores.max())
    }
    
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(seg_errors,label="Reconstruction Error")
    axes[0].axhline(thr,color="r",linestyle="--",label=f"Threshold={thr:.4f}")
    axes[0].set_ylabel("Error", fontsize=10, weight='bold')
    axes[0].set_title(f"Segment: {seg_name} | Anomalies: {seg_anom.sum()}/{len(seg_errors)}", fontsize=12, weight='bold')
    

    axes[0].legend()
    axes[1].plot(seg_scores,label="Anomaly Score (0–1)",)
    axes[1].axhline(0.5,color="r",linestyle="--",label="Score Threshold=0.5")
    axes[1].set_xlabel("Window index", fontsize=10, weight='bold')
    axes[1].set_ylabel("Score", fontsize=10, weight='bold')
    axes[1].legend(prop={'weight': 'bold'})
    
    for ax in axes:
        ax.tick_params(axis='both', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plots_dir,f"{seg_name}_anomaly.png"))
    plt.close(fig)

print(f"All plots saved in '{plots_dir}/' folder.")


print("\n=== Anomaly Detection Summary ===")
for k,v in results.items():
    print(f"{k}: {v['n_anomalies']}/{v['n_windows']} anomalous windows | mean error={v['mean_error']:.4f} | max score={v['max_score']:.3f}")
