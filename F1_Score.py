import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


df = pd.read_csv("Performance\\F1.txt", sep="\t", header=None)

df.columns = ["Samples", "OQ-BNA", "DenseNet ABiLSTM", "TQCPat", "PPG-NET"]

x = range(len(df["Samples"]))
bar_width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
models = ["OQ-BNA", "DenseNet ABiLSTM", "TQCPat", "PPG-NET"]
bold_font = FontProperties(weight="bold")


plt.figure(figsize=(12, 6))

for i, model in enumerate(models):
    plt.bar(
        [j + (i-1.5)*bar_width for j in x],
        df[model],
        width=bar_width,
        label=model,
        color=colors[i]
    )

plt.xticks(x, df["Samples"], rotation=45, fontweight="bold")
plt.yticks(fontweight="bold")
plt.xlabel("Samples", fontweight="bold")
plt.ylabel("F1-score", fontweight="bold")
plt.title("F1-score Comparison", fontweight="bold")
plt.legend(prop=bold_font)
plt.tight_layout()
plt.savefig("F1_comparison.png", dpi=300)
plt.show()
