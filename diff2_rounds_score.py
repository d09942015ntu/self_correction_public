# required packages:
import os.path
import random, os
import pandas as pd
import numpy as np
import time
import requests
import subprocess
import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from utils import parse_hidden_states, seed_everything

def get_toxicity(sentence):
    payload = {
        "TextSegments": [{"Text": sentence}],
        "LanguageCode": "en"
    }

    command = [
        "aws", "comprehend", "detect-toxic-content",
        "--region", "us-east-1",
        "--cli-input-json", json.dumps(payload)  
    ]

    result = subprocess.run(
        command,
        input=json.dumps(payload),  
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            return float(response["ResultList"][0]["Toxicity"])
        except (KeyError, IndexError, ValueError) as e:
            print(f"Unexpected result for: {sentence}\n Error: {e}\n Raw: {result.stdout}")
            assert 0
            #return 0.0
    else:
        print(f"AWS CLI error for: {sentence}\n {result.stderr}")
        assert 0
        #return 0.0

def get_all_scores(responses, num_rounds, cache_dir):

    toxicity = []
    for round in range(num_rounds):
        print("round ", round)
        print("\n")
        temp_record = []
        for text in responses:
            print(f"text[round] = {text[round]}")
            toxicity_score = get_toxicity(text[round])
            temp_record.append(toxicity_score)
        toxicity.append(temp_record)
        print(toxicity)

    toxicity_t = list(map(list, zip(*toxicity)))

    with open(os.path.join(cache_dir, 'toxicity.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'Round {i+1}' for i in range(num_rounds)])
        writer.writerows(toxicity_t)
    print(f"Toxicity scores saved to {os.path.join(cache_dir, 'toxicity.csv')}")
    return toxicity

def plot_with_std(data, condition, output_dir):
    line_color = '#08519c' if "positive" in condition else '#a50f15'
    fill_color = '#377eb8' if "positive" in condition else '#e41a1c'

    data = np.array(data)
    means = data.mean(axis=1)
    stds = data.std(axis=1)
    x =  np.array([0, 1, 2, 3, 4])

    plt.figure(figsize=(6, 4))  
    plt.plot(x, means, label='Mean', color = line_color , linewidth=2)
    plt.fill_between(x, means - stds, means + stds, color=fill_color, alpha=0.2, label='±1 Std Dev')
    plt.xticks(x)
    plt.ylim([-0.05,1])
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Toxicity', fontsize=12)
    #plt.title('Mean with Standard Deviation Shading')
    plt.legend(loc='best', fontsize=10)
    #plt.grid(True)
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.savefig(os.path.join(output_dir,f"toxicity_{condition}.png"), dpi=300, bbox_inches='tight')


def select_toxicity(Toxicity, condition):
    t_array = Toxicity
    if "positive" in condition:
        score_increase = np.sum(
            ((t_array[1:, :] - t_array[:-1, :]) < -0.02*(abs(t_array[1:, :])+abs(t_array[:-1, :])) ).astype(int),
            axis=0)
    elif "negative" in condition:
        score_increase = np.sum(
            ((t_array[1:, :] - t_array[:-1, :]) > 0.02 * (abs(t_array[1:, :]) + abs(t_array[:-1, :]))).astype(int),
            axis=0)
    else:
        assert 0
    return np.array(Toxicity)[:,score_increase >= 3]


def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data", type=int, default=500)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="alignment-handbook/zephyr-7b-sft-full")
    args = parser.parse_args()

    seed_everything(88)

    base_dir = os.path.join("outputs", os.path.basename(args.model_name))
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    for condition in ["positive", "negative"]:
        cache_dir = os.path.join(base_dir, "hidden_states", condition)
        toxicity_dir = os.path.join(base_dir, "toxicity", condition)
        os.makedirs(toxicity_dir, exist_ok=True)

        length, responses, last_hidden, _ = parse_hidden_states(args.num_data, args.num_rounds, cache_dir)


        save_path = os.path.join(toxicity_dir, 'toxicity.csv')
        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping computation.")
            toxicity = [[] for _ in range(args.num_rounds)]
            with open(save_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    row = np.array(row, dtype=float)
                    for r in range(0,args.num_rounds):
                        toxicity[r].append(row[r])
            for i in range(0,args.num_rounds):
                toxicity[i] = toxicity[i][:args.num_data]
        else:
            toxicity = get_all_scores(responses, args.num_rounds, cache_dir)

        toxicity = np.array(toxicity)
        toxicity_selected = select_toxicity(toxicity, condition)
        plot_with_std(toxicity, condition, output_dir)
        plot_with_std(toxicity_selected, f"{condition}_selected", output_dir)

        print(f"Sample Counts Toxicity:{toxicity.shape[1]}")
        print(f"Sample Counts Toxicity_selected:{toxicity_selected.shape[1]}")

        print(f"({condition}) Mean for each round = {np.array(toxicity).mean(axis=1)}")
        print(f"({condition}) Std. for rach round = {np.array(toxicity).std(axis=1)}")

        json.dump(toxicity.tolist(), open(os.path.join(output_dir, f"{condition}_toxicity.json"),"w"))
        json.dump(toxicity_selected.tolist(), open(os.path.join(output_dir, f"{condition}_toxicity_selected.json"),"w"))



if __name__ == '__main__':
    run()
