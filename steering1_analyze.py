import argparse
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity



def analyze(args):

    rng0 = np.random.RandomState(0)
    rng1 = np.random.RandomState(1)
    dataset_tags = "".join(args.data_tags)
    dataset_ratio = int(args.data_ratio*100)
    steering_vec_dir = os.path.join(args.output_dir,os.path.basename(args.model_dir),f"steering_d{dataset_tags}_t{dataset_ratio}","1_vector")
    steering_vec_sep_raw = json.load(open(os.path.join(steering_vec_dir,"steer_sep.json"),"r"))
    steering_vec = json.load(open(os.path.join(steering_vec_dir,"steer.json"),"r"))
    steering_vec = dict((x,np.average(y,axis=0)) for x,y in steering_vec.items())
    steering_vec_sep = defaultdict(list)
    results_txt = ""
    cosine_sim_result = []
    for layer in steering_vec.keys():
        for steering_vec_i in steering_vec_sep_raw:
            steering_vec_sep[layer].append(np.average(steering_vec_i[layer],axis=0))
        vec_all = [steering_vec[layer]]+steering_vec_sep[layer]
        results = cosine_similarity(vec_all,vec_all)
        result_avg = np.average(results)
        cosine_sim_result.append(result_avg)

        vec_all = [steering_vec[layer]]+steering_vec_sep[layer]

        vec_rand = [rng0.permutation(steering_vec[layer]), rng1.permutation(steering_vec[layer])]
        results_rand = cosine_similarity(vec_all,vec_rand)

        results_txt +=(f"layer {layer},\ncosine sim avg:{result_avg},\ncosine sim:\n {results}\n\ncosine sim rand:\n {results_rand}\n\n")

    with open(os.path.join(steering_vec_dir,"analyze.txt"),"w") as f:
        f.write(results_txt)
        print(results_txt)
    plt.clf()
    plt.plot(cosine_sim_result)
    plt.ylim(0,1)
    plt.title("Cosine Similarity")
    plt.savefig(os.path.join(steering_vec_dir,"analyze.png"))



# Models: 
    # Qwen3: "Qwen/Qwen3-4B-Instruct-2507" 
    # Qwen2.5: "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" 
    # Qwen2: "Qwen/Qwen2-7B-Instruct"
    # Mistral: "mistralai/Mistral-7B-Instruct-v0.3"
    # Gemma-3: "google/gemma-3-4b-it"
    # Deepseek: "deepseek-ai/deepseek-llm-7b-chat"
    # Zephyr: "HuggingFaceH4/zephyr-7b-beta"
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_tags", type=str, default=["1"], nargs="+")
    parser.add_argument("--data_ratio", type=float, default=1)
    parser.add_argument("--model_dir", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--limit", type=int, default=20000000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    analyze(args)



if __name__ == '__main__':
    run()
