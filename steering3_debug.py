import argparse
import json
import os
import glob
import numpy as np
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from utils import get_toxicity, gen_score_function





def get_score(args, selected_layer, output_dir, scoring_tokenizer, scoring_model):
    for condition in ["positive","negative"] :
        #if condition == "positive":
        #    s_dir = -1
        #else:
        #    s_dir = 1
        for s_dir in [-1.0, -0.5, 0.5, 1.0]:
            input_sub_dir = os.path.join(args.input_dir, condition)
            output_sub_dir = os.path.join(output_dir, condition)
            os.makedirs(output_sub_dir, exist_ok=True)
            input_json_files = sorted(glob.glob(os.path.join(input_sub_dir,"*.json")))[:args.limit]
            total = min(args.limit, len(input_json_files))
            for i, input_json_file in enumerate(input_json_files):
                #print(f"condition:{condition}, file:{i+1}/{total}")
                output_json_file = input_json_file.replace("2_inference", "3_scoring")
                if os.path.exists(output_json_file):
                    json_out = json.load(open(output_json_file,"r"))
                else:
                    json_out = {}
                json_in = json.load(open(input_json_file, "r"))

                for layer in selected_layer:
                    o_key = str((layer, s_dir))
                    if o_key not in json_in.keys():
                        continue
                    if o_key in json_out.keys():
                        continue
                    sentence = json_in[o_key]
                    score = get_toxicity(sentence, scoring_tokenizer, scoring_model)
                    #print(f"layer:{layer}, dir:{s_dir}, score:{score}")
                    json_out[o_key] = {"sentence":sentence, "score":score}
                json.dump(json_out, open(output_json_file, "w"), indent=2)


def plot_score(args, selected_layer, output_dir, smooth=True, limit=100):
    json_result_all = {}
    #    "positive_-1":defaultdict(list),
    #    "positive_-0.5":defaultdict(list),
    #    "positive_0.5":defaultdict(list),
    #    "positive_1":defaultdict(list),
    #    "negative_-1":defaultdict(list),
    #    "negative_-0.5":defaultdict(list),
    #    "negative_0.5":defaultdict(list),
    #    "negative_1":defaultdict(list)
    #}
    for condition in ["positive", "negative"]:
        #if condition == "positive":
        #    s_dir = -1
        #else:
        #    s_dir = 1

        json_result_all[f"{condition}"] = defaultdict(list)
        input_sub_dir = os.path.join(args.input_dir, condition)
        output_sub_dir = os.path.join(output_dir, condition)
        os.makedirs(output_sub_dir, exist_ok=True)
        input_json_files = sorted(glob.glob(os.path.join(input_sub_dir,"*.json")))[:args.limit]
        total = min(limit, len(input_json_files))
        for i, input_json_file in enumerate(input_json_files):
            file_id = os.path.basename(input_json_file).replace(".json","")
            #json_result_all[condition][file_id] = []
            for s_dir in [-1.0, -0.5, 0.5, 1.0]:
                #print(f"condition:{condition}, file:{i+1}/{total}")
                if i > total:
                    break
                output_json_file = input_json_file.replace("2_inference", "3_scoring")
                assert os.path.exists(output_json_file)
                json_out = json.load(open(output_json_file,"r"))
                layer = -1
                o_key = str((layer, s_dir))
                if o_key in json_out.keys():
                    score = json_out[o_key]['score']
                    json_result_all[condition][file_id].append(score)
            if np.average(json_result_all[condition][file_id]) != json_result_all[condition][file_id][0]:
                print(f"assertion failed:{condition}, file_id:{file_id}")

            #print(1)





def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/Qwen2.5-3B-Instruct/steering_d1_t100/2_inference/")
    parser.add_argument("--limit", type=int, default=300)

    args = parser.parse_args()

    output_dir = args.input_dir.replace("2_inference", "3_scoring")
    os.makedirs(output_dir, exist_ok=True)


    selected_layer = [-1] #+ list(range(35))

    scoring_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
    scoring_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
    #batch = tokenizer.encode("You are amazing!", return_tensors="pt")
    #output = scoring_model(batch)

    get_score(args, selected_layer, output_dir, scoring_tokenizer, scoring_model)
    for l in [(args.limit)]:
        for scale in [1.0,0.5]:
            plot_score(args, selected_layer, output_dir,  smooth=False, limit=l)



if __name__ == '__main__':
    run()
