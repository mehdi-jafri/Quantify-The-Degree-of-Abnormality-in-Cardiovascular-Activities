import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties



df = pd.read_csv("Performance\\MAE.txt", sep="\t", header=None)


df.columns = ["Samples", "OQ-BNA", "DenseNet ABiLSTM", "TQCPat", "PPG-NET"]


x = range(len(df["Samples"]))
bar_width = 0.2
plt.figure(figsize=(12, 6))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.bar([i - 1.5*bar_width for i in x], df["OQ-BNA"], width=bar_width, label="OQ-BNA", color=colors[0])
plt.bar([i - 0.5*bar_width for i in x], df["DenseNet ABiLSTM"], width=bar_width, label="DenseNet ABiLSTM", color=colors[1])
plt.bar([i + 0.5*bar_width for i in x], df["TQCPat"], width=bar_width, label="TQCPat", color=colors[2])
plt.bar([i + 1.5*bar_width for i in x], df["PPG-NET"], width=bar_width, label="PPG-NET", color=colors[3])


plt.xticks(x, df["Samples"], rotation=45, fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel("Samples", fontweight="bold")
plt.ylabel("MAE of outlier detection (%)", fontweight="bold")
plt.title("Comparison of Models", fontweight="bold")


bold_font = FontProperties(weight="bold")
plt.legend(prop=bold_font)
plt.savefig("MAE_comparison.png", dpi=300)
plt.tight_layout()
plt.show()
