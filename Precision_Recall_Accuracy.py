import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


df = pd.read_csv("Performance\\PRA.txt", sep="\t", header=None)


df.columns = [
    "Samples",
    "Precision_OQ-BNA", "Precision_DenseNet ABiLSTM", "Precision_TQCPat", "Precision_PPG-NET",
    "Recall_OQ-BNA", "Recall_DenseNet ABiLSTM", "Recall_TQCPat", "Recall_PPG-NET",
    "Accuracy_OQ-BNA", "Accuracy_DenseNet ABiLSTM", "Accuracy_TQCPat", "Accuracy_PPG-NET"
]


x = range(len(df["Samples"]))
bar_width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
models = ["OQ-BNA", "DenseNet ABiLSTM", "TQCPat", "PPG-NET"]
bold_font = FontProperties(weight="bold")


def plot_metric(metric):
    plt.figure(figsize=(12, 6))
    
    for i, model in enumerate(models):
        plt.bar(
            [j + (i-1.5)*bar_width for j in x], 
            df[f"{metric}_{model}"], 
            width=bar_width, 
            label=model, 
            color=colors[i]
        )
    
    
    plt.xticks(x, df["Samples"], rotation=45, fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.xlabel("Samples", fontweight="bold")
    plt.ylabel(metric, fontweight="bold")
    plt.title(f"{metric} Comparison", fontweight="bold")
    plt.legend(prop=bold_font)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300)
    plt.show()


plot_metric("Precision")
plot_metric("Recall")
plot_metric("Accuracy")
