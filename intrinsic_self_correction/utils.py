import json
import os.path
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def plot_results(non_toxic_probs, toxic_probs, output_dir, model_basename, toxicity_classifier,strong_or_weak, fname):
    non_toxic_probs = np.array(non_toxic_probs) if non_toxic_probs is not None else None
    toxic_probs = np.array(toxic_probs) if toxic_probs is not None else None
    
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(4, 3))
    x = np.arange(non_toxic_probs.shape[1])
    if non_toxic_probs is not None:
        non_toxic_data = non_toxic_probs
        means_non_toxic = non_toxic_data.mean(axis=0).reshape(-1)
        stds_non_toxic = non_toxic_data.std(axis=0).reshape(-1)
        plt.plot(x, means_non_toxic, label='Non-toxic Prompting', color='#377eb8', linewidth=2)
        plt.fill_between(x, means_non_toxic - stds_non_toxic, means_non_toxic + stds_non_toxic, color='#377eb8', alpha=0.2)
    if toxic_probs is not None:
        toxic_data = toxic_probs
        means_toxic = toxic_data.mean(axis=0).reshape(-1)
        stds_toxic = toxic_data.std(axis=0).reshape(-1)
        plt.plot(x, means_toxic, label='Toxic Prompting', color='#e41a1c', linewidth=2)
        plt.fill_between(x, means_toxic - stds_toxic, means_toxic + stds_toxic, color='#e41a1c', alpha=0.2)

    plt.xticks(x)
    plt.xlim([0, 4])
    plt.ylim([0, 1])
    plt.xlabel("Round", fontsize=12, labelpad=4)
    plt.ylabel("Toxicity", fontsize=12, labelpad=4)
    plt.legend(loc="best", fontsize=11)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)    
    plt.title(f"{model_basename} – {toxicity_classifier} - {strong_or_weak}", 
            fontsize=14, weight = 'bold', pad=8)

    ax = plt.gca()
    plt.xticks(fontsize=11, fontweight="normal")
    plt.yticks(fontsize=11, fontweight="normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(output_dir, fname),
            dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_json(data, output_dir, fname):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, fname), "w") as f:
        json.dump(data.tolist() if isinstance(data, np.ndarray) else data, f)
    