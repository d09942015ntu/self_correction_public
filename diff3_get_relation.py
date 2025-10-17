import os
import heapq
import pandas as pd
import numpy as np
import torch
import time
import csv
import requests
import json
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from argparse import ArgumentParser
from utils import parse_hidden_states



def inner_product_unembed_diff(nopca_difference, unembedding):
    inner = []
    for data in nopca_difference:
        temp_inners = []
        for diffround in  data:
            difference_tensor = torch.tensor(diffround, dtype=unembedding.dtype, device=unembedding.device)
            temp_inner = unembedding @ difference_tensor
            temp_inners.append(temp_inner.detach().cpu().numpy())
        inner.append(temp_inners)
    return np.stack(inner) 

def average_diff(inner):
    inner_ave = []
    for round in range(inner.shape[1]):
        temp_inner = torch.zeros(32000)
        for data in range(inner.shape[0]):
            temp_inner += inner[data][round]
        inner_ave.append(temp_inner/inner.shape[0])
    return np.stack(inner_ave) 

def calculate_innerscore(list, condition, toporsmall, num = 100):
    score = 0
    output = []
    for ri in range(len(list)):
        for data in list[ri]:
            inner = data[0]
            toxic = data[1]
            score = score + inner*toxic
        output.append(score)
        score = 0
    print(f"({toporsmall}_{num}) inner_score of {condition} prompt is {output}")
    return output

def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data", type=int, default=500)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="alignment-handbook/zephyr-7b-sft-full")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_dir = os.path.join("outputs", os.path.basename(args.model_name))
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    unembedding = model.lm_head.weight

    k = 100

    for condition in ["positive", "negative"]:
        cache_dir = os.path.join(base_dir, "hidden_states", condition)
        toxicity_dir = os.path.join(base_dir, "toxicity", condition)

        _, _, last_hidden, _ = parse_hidden_states(args.num_data, args.num_rounds, cache_dir)
        diff = last_hidden[:,1:,:] - last_hidden[:,:-1,:]

        inner_products = inner_product_unembed_diff(diff, unembedding)
        ave_inner_p = average_diff(inner_products)
        print(f"ave_inner_p.shape = {ave_inner_p.shape}")
        vocab = tokenizer.get_vocab()

        toxic_dict = {}
        with open(os.path.join(base_dir, 'tokens.csv'), mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                token = row[0]
                score = row[1]
                toxic_dict[vocab[token]] = [token,float(score)]
        #print(toxic_dict)

        keys = list(toxic_dict.keys())
        relation = []
        for r in range(args.num_rounds-1):
            temp_rela = []
            for ids in keys:
                temp_rela.append([float(ave_inner_p[r,ids]),toxic_dict[ids][1]])
            relation.append(temp_rela)

        top_k = []
        for ri in range(len(relation)):
            top_k.append(heapq.nlargest(k, relation[ri], key = lambda x: x[1]))

        small_k = []
        for ri in range(len(relation)):
            small_k.append(heapq.nsmallest(k, relation[ri], key=lambda x: x[1]))

        with open(os.path.join(toxicity_dir, 'relation.csv'), 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['InnerProduct', 'Toxicity'])
            for round in range(args.num_rounds-1):
                writer.writerow(['Round', f"{round+1} and {round+2}"])
                for inner, toxic in relation[round]:
                    writer.writerow([inner, toxic])
        print("storage as relation.csv")

        calculate_innerscore(top_k, condition,"top")
        calculate_innerscore(small_k, condition,"small")
  
if __name__ == '__main__':
    run()
