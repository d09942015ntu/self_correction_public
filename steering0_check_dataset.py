import argparse
import json
import os

from utils import gen_score_function


my_score_func = gen_score_function()


def get_embed_sim(args):

    toxic_lines = open(args.data_toxic, "r").readlines()
    nontoxic_lines = open(args.data_notoxic,"r").readlines()

    counts = 0
    s_toxic = 0
    s_notoxic = 0

    for t_item, nt_item in zip(toxic_lines, nontoxic_lines):
        print(f"forward: {counts}/{args.limit}")

        if counts >= args.limit:
            break

        item_toxic = json.loads(t_item)
        item_notoxic = json.loads(nt_item)
        prompt_toxic = item_toxic['prompt']['text']
        prompt_notoxic = item_notoxic['prompt']['text']
        s_toxic += my_score_func(prompt_toxic)
        s_notoxic += my_score_func(prompt_notoxic)

        counts += 1

    print(f"s_toxic:{s_toxic/args.limit}, s_notoxic:{s_notoxic/args.limit}")





def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_toxic", type=str, default="data/1_RealToxicityPrompts/5k/output_toxic_5k.jsonl")
    parser.add_argument("--data_notoxic", type=str, default="data/1_RealToxicityPrompts/5k/output_no_toxic_5k.jsonl")
    parser.add_argument("--model_dir", type=str, default="models/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()


    output_dir = os.path.join(args.output_dir,os.path.basename(args.model_dir),"steering","0_vector")
    os.makedirs(output_dir,exist_ok=True)


    get_embed_sim(args) #, hidden_states, output_dir)



if __name__ == '__main__':
    run()
