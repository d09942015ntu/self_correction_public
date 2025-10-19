import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import parse_hidden_states

from sklearn.metrics.pairwise import cosine_similarity





def get_inner_prod(args):

    rng = np.random.RandomState(0)

    steering_vec = json.load(open(args.steering_vec, "r"))


    output_dir = os.path.dirname(args.steering_vec).replace("1_vector", "4_innerprod")
    os.makedirs(output_dir, exist_ok=True)


    steering_vec = dict([(int(x), np.array(y)) for x, y in  steering_vec.items()])
    random_vec = dict([(int(x), rng.permutation(y.T).T) for x, y in  steering_vec.items()])

    vec_types = {
        "steering_vec": steering_vec,
        "random_vec": random_vec,
    }

    result_all = defaultdict(list)
    for vec_type, vec_vals in vec_types.items():
        for condition in ["positive", "negative"]:
            length, result, last_hidden, last_hidden_mean, toxicity = (
                parse_hidden_states(args.limit, args.num_rounds, output_dir=os.path.join(args.diff_vec_dir,condition)))

            diff_vec = last_hidden[:, 1:, :] - last_hidden[:, :-1, :]
            for round in range(args.num_rounds-1):
                diff_vec_r = diff_vec[:,round,:]

                for layer in vec_vals.keys():
                    cosine_sim_r = cosine_similarity(diff_vec_r, vec_vals[layer])
                    cval = np.average(cosine_sim_r)
                    if "random" in vec_type :
                        result_all[f"random"].append((layer, cval))
                    else:
                        result_all[f"{condition}_round{round+1}"].append((layer,cval))

    plt.clf()
    plt.figure(figsize=(4, 3))
    for ikey,ivals in result_all.items():
        ivals_avg = defaultdict(list)
        for ival in ivals:
            ivals_avg[ival[0]].append(ival[1])
        if "random" in ikey:
            plt.plot(list(ivals_avg.keys()),[np.average(x) for x in ivals_avg.values()], label="random", color="gray")
        else:
            label = ikey.replace("positive", "non-toxic").replace("negative", "toxic")
            plt.plot(list(ivals_avg.keys()),[np.average(x) for x in ivals_avg.values()], label=label)
        plt.plot(list(ivals_avg.keys()), [0 for _ in ivals_avg.values()], color="black")
    plt.xlabel("layer", fontsize=11, labelpad=4)
    plt.ylabel("cosine similarity", fontsize=11, labelpad=0)
    plt.title("Qwen2.5-3B-Instruct",fontsize=12, weight = 'bold', pad=6)
    plt.legend(loc="best",fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"Qwen2.5-3B_innerprod_result.png"))



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_vec", type=str, default="outputs/Qwen2.5-3B-Instruct/steering_d1_t100/1_vector/steer.json")
    parser.add_argument("--diff_vec_dir", type=str, default="outputs/Qwen2.5-3B-Instruct/RoBERTa_strong_prompt_text_detox_results")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--num_rounds",type=int,default=3)

    args = parser.parse_args()




    get_inner_prod(args)



if __name__ == '__main__':
    run()
