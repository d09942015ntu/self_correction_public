import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_splits(toxic_url, non_toxic_url):
    tox_df = pd.read_json(toxic_url, lines=True)
    non_tox_df = pd.read_json(non_toxic_url, lines=True)

    tox_df["toxicity"] = tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)
    non_tox_df["toxicity"] = non_tox_df["prompt"].apply(lambda d: d.get("toxicity") if isinstance(d, dict) else None)

    # Splits into 4k train and 1k test for both toxic and non-toxic subsets
    tox_df["tox_bin"] = pd.qcut(tox_df["toxicity"], q=10, duplicates="drop")
    toxic_train, toxic_test = train_test_split(tox_df, test_size=1000, stratify=tox_df["tox_bin"], random_state=87)
    toxic_train = toxic_train.drop(columns=["tox_bin"])
    toxic_test = toxic_test.drop(columns=["tox_bin"])

    # print(toxic_train.columns) -> Index(['filename', 'begin', 'end', 'challenging', 'prompt', 'toxicity'], dtype='object')
    toxic_train.to_json("./data/toxic_train_4k.jsonl", orient="records", lines=True, force_ascii=False)
    toxic_test.to_json("./data/toxic_test_1k.jsonl", orient="records", lines=True, force_ascii=False)

    plt.figure(figsize=(9, 5.5))
    plt.hist(toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
    plt.hist(toxic_test["toxicity"],  bins=10, alpha=0.55, label="Test")
    plt.title("Toxicity Distribution of toxic train and test splits")
    plt.xlabel("Toxicity")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tox_splits_distribution.png", dpi=160)
    plt.close()

    print("\nToxic splits toxicity stats:")
    print(f"Train split. mean = {toxic_train['toxicity'].mean()}, std = {toxic_train['toxicity'].std()}")
    print(f"Test  split. mean = {toxic_test['toxicity'].mean()}, std = {toxic_test['toxicity'].std()}")

    non_tox_df["tox_bin"] = pd.qcut(non_tox_df["toxicity"], q=10, duplicates="drop")
    non_toxic_train, non_toxic_test = train_test_split(non_tox_df, test_size=1000, stratify=non_tox_df["tox_bin"], random_state=87)
    non_toxic_train = non_toxic_train.drop(columns=["tox_bin"])
    non_toxic_test = non_toxic_test.drop(columns=["tox_bin"])

    # print(non_toxic_train.columns) -> Index(['filename', 'begin', 'end', 'challenging', 'prompt', 'toxicity'], dtype='object')
    non_toxic_train.to_json("./data/non_toxic_train_4k.jsonl", orient="records", lines=True, force_ascii=False)
    non_toxic_test.to_json("./data/non_toxic_test_1k.jsonl", orient="records", lines=True, force_ascii=False)
    
    plt.figure(figsize=(9, 5.5))
    plt.hist(non_toxic_train["toxicity"], bins=10, alpha=0.55, label="Train")
    plt.hist(non_toxic_test["toxicity"],  bins=10, alpha=0.55, label="Test")
    plt.title("Toxicity Distribution of non-toxic train and test splits")
    plt.xlabel("Toxicity")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("non_tox_splits_distribution.png", dpi=160)
    plt.close()

    # Toxic splits toxicity stats:
    # Train split. mean = 0.68214068351, std = 0.11449748153377481
    # Test  split. mean = 0.68254970954, std = 0.11437375989966483

    # Non-toxic splits toxicity stats:
    # Train split. mean = 0.12387753295302499, std = 0.13354831709837772
    # Test  split. mean = 0.12386198251899999, std = 0.13397116776069914
    print("\nNon-toxic splits toxicity stats:")
    print(f"Train split. mean = {non_toxic_train['toxicity'].mean()}, std = {non_toxic_train['toxicity'].std()}")
    print(f"Test  split. mean = {non_toxic_test['toxicity'].mean()}, std = {non_toxic_test['toxicity'].std()}")

create_splits(
  "https://raw.githubusercontent.com/LizLizLi/DeStein/refs/heads/master/data/RealToxicityPrompts/5k/output_toxic_5k.jsonl",
  "https://raw.githubusercontent.com/LizLizLi/DeStein/refs/heads/master/data/RealToxicityPrompts/5k/output_no_toxic_5k.jsonl"
)
