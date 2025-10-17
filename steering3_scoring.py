import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from utils import gen_score_function


def get_score(args, selected_layer):
    my_score_func = gen_score_function()
    for s_dir in [-1.0, 1.0]:
        s_dir_str = 'negative' if s_dir < 0 else 'positive'
        input_json_files = sorted(glob.glob(os.path.join(args.input_dir,"**","*.json"), recursive=True))[:args.limit]
        total = min(args.limit, len(input_json_files))
        for i, input_json_file in enumerate(input_json_files):
            if i%100 ==0:
                print(f"{s_dir_str}, file:{i + 1}/{total}")
            output_json_file = input_json_file.replace("2_inference", "3_scoring")
            if os.path.exists(output_json_file):
                json_out = json.load(open(output_json_file,"r"))
            else:
                os.makedirs(os.path.dirname(output_json_file),exist_ok=True)
                json_out = {}
            json_in = json.load(open(input_json_file, "r"))

            for layer in selected_layer:
                o_key = str((layer, s_dir))
                if o_key not in json_in.keys():
                    continue
                if o_key in json_out.keys():
                    continue
                sentence = json_in[o_key]
                score = my_score_func(sentence)
                json_out[o_key] = {"sentence":sentence, "score":score}
            json.dump(json_out, open(output_json_file, "w"), indent=2)
    return total


def plot_score(args, selected_layer, output_dir, output_tag, limit=100):
    plot_title_all={
        "_hook_seq_toxic": "Steering Vector, Starting from Toxic",
        "_hook_seq_non_toxic": "Steering Vector, Starting from Non-Toxic",
        "_hook_avg_toxic": "Averaged Steering Vector, Starting from Toxic",
        "_hook_avg_non_toxic": "Averaged Steering Vector, Starting from Non-Toxic",
        "_hook_random_toxic": "Random Steering Vector, Starting from Toxic",
        "_hook_random_non_toxic":  "Random Steering Vector, Starting from Non-Toxic",
    }
    json_result_all = {}
    for s_dir in [-1.0, 1.0]:
        s_dir_str = 'negative' if s_dir < 0 else 'positive'
        json_key = f"{s_dir_str} steering"
        json_result_all[json_key] = defaultdict(list)
        input_json_files = sorted(glob.glob(os.path.join(args.input_dir,"**","*.json"), recursive=True))[:args.limit]
        total = min(limit, len(input_json_files))
        for i, input_json_file in enumerate(input_json_files):
            if i%100 == 0:
                print(f"{s_dir_str}, file:{i+1}/{total}")
            if i > total:
                break
            output_json_file = input_json_file.replace("2_inference", "3_scoring")
            assert os.path.exists(output_json_file)
            json_out = json.load(open(output_json_file,"r"))

            for layer in selected_layer:
                o_key = str((layer, s_dir))
                if o_key in json_out.keys():
                    score = json_out[o_key]['score']
                    json_result_all[json_key][layer].append(score)

    json_result_avg = {}
    for ikey in json_result_all.keys():
        json_result_avg[ikey] = {}
        for jkey in json_result_all[ikey]:
            json_result_avg[ikey][jkey] = np.average(json_result_all[ikey][jkey])

    plt.clf()

    baseline_exist = False
    for ikey in json_result_avg.keys():
        data = sorted(json_result_avg[ikey].items(), key=lambda x:x[0])
        print(f"{ikey}:{json_result_avg[ikey]}")
        data_x = [x[0] for x in data if x[0] >=0]
        data_y = [x[1] for x in data if x[0] >=0]

        plt.plot(data_x, data_y, label=ikey)

        if not baseline_exist:
            plt.plot([0, data[-1][0]], [data[0][1], data[0][1]], label=f"no steering",
                     color="grey")
            baseline_exist=True


    plt.legend(loc='upper right')
    plt.xlabel("layer")
    plt.ylabel("toxicity score")
    plt.title(f"{plot_title_all.get(output_tag,'')}, {limit} Samples")
    plt.savefig(os.path.join(output_dir, f"{output_tag}_{limit}.png"))
    json.dump(
        {
           "avg":json_result_avg,
           "all":json_result_all,
        },
        open(os.path.join(output_dir, f"{output_tag}_{limit}.json"),"w"), indent=2)





def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/gemma-3-4b-it/steering_d1_t100/2_inference/")
    parser.add_argument("--limit", type=int, default=100000)

    args = parser.parse_args()

    output_dir = args.input_dir.replace("2_inference", "3_scoring")
    output_dir_base = os.path.join(output_dir.split("3_scoring")[0],"3_scoring","pngs")
    output_tag = output_dir.split("3_scoring")[-1].replace("/","_")
    os.makedirs(output_dir_base, exist_ok=True)


    selected_layer = [-1] + list(range(35))

    print(args)
    total_files = get_score(args, selected_layer)
    for l in [int(total_files/8), int(total_files/4), int(total_files/2), int(total_files)]:
        plot_score(args, selected_layer, output_dir_base, output_tag, limit=l)



if __name__ == '__main__':
    run()
