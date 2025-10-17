import json
import os.path
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from transformers import RobertaTokenizer, RobertaForSequenceClassification

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
    plt.figure(figsize=(7, 5))
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
    plt.ylabel("Non-toxic Probability", fontsize=12, labelpad=4)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.6)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)    
    plt.title(f"{model_basename} – {toxicity_classifier} - {strong_or_weak}", 
            fontsize=14, weight="bold", pad=6)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()

def save_json(data, output_dir, fname):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, fname), "w") as f:
        json.dump(data.tolist() if isinstance(data, np.ndarray) else data, f)
    


def parse_hidden_states(num_data, num_rounds, output_dir = "hidden_states"):
    length = []
    result = []
    last_hidden = []
    last_hidden_mean = []
    toxicity = []

    for d in range(num_data):
        length_temp =[]
        result_temp =[]
        last_hidden_temp =[]
        toxicity_temp =[]
        for ri in range(num_rounds):
            fname = os.path.join(output_dir,f"%s_%s.pkl"%(ri, str(d).zfill(5)))
            item = pickle.load(open(fname, "rb"))
            #length_temp.append(item['length'])
            result_temp.append(item['result'])
            last_hidden_temp.append(item['last_hidden'])
            toxicity_temp.append(item['toxicity_evaluations'][1])
        length.append(length_temp)
        result.append(result_temp)
        last_hidden.append(last_hidden_temp)
        toxicity.append(toxicity_temp)
        
    last_hidden = np.array(last_hidden)
    toxicity = np.array(toxicity)
    last_hidden_mean = np.mean(last_hidden, axis=0)

    return np.array(length), result, last_hidden, last_hidden_mean, toxicity


def my_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_toxicity(sentence, scoring_tokenizer, scoring_model):

    batch = scoring_tokenizer.encode(sentence, return_tensors="pt", max_length=512).to(scoring_model.device)
    output = scoring_model(batch)
    x = output.logits[0].cpu().detach().numpy()
    return float(my_softmax(x)[1]) #Toxic Score

def gen_score_function():
    scoring_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
    scoring_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to(torch.device("cuda"))
    scoring_cache = dict()
    def my_get_toxicity(sentence):
        if sentence in scoring_cache:
            return scoring_cache[sentence]
        else:
            score = get_toxicity(sentence, scoring_tokenizer, scoring_model)
            scoring_cache[sentence] = score
        return score
    return my_get_toxicity
