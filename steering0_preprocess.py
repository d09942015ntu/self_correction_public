import os
import pandas as pd
from tqdm import tqdm

from utils import gen_score_function

my_score_func = gen_score_function()



def post_process(t_prompt_score, nt_prompt_score):
    t_prompt_score = sorted(t_prompt_score, key=lambda x: x[1], reverse=True)
    nt_prompt_score = sorted(nt_prompt_score, key=lambda x: x[1])
    t_prompt_score = t_prompt_score[:int(len(t_prompt_score) )]
    nt_prompt_score = nt_prompt_score[:int(len(nt_prompt_score) )]
    return t_prompt_score, nt_prompt_score



def run_1dataset():
    run_and_save_json(1, run_realtoxicity)


def run_and_save_json(data_id, run_function):
    limit=2000000
    t_prompt_score, nt_prompt_score = run_function(limit)
    t_prompt_score=[{"prompt":x[0], "toxicity":x[1]} for x in t_prompt_score]
    nt_prompt_score=[{"prompt":x[0], "toxicity":x[1]} for x in nt_prompt_score]

    output_dir=f"data_processed/{data_id}"
    os.makedirs(output_dir, exist_ok=True)


    t_df = pd.DataFrame(data=t_prompt_score, columns=["prompt", "toxicity"])
    nt_df = pd.DataFrame(data=nt_prompt_score, columns=["prompt", "toxicity"])

    t_df.to_json(f"{output_dir}/t_prompt_score.jsonl", orient="records", lines=True)
    nt_df.to_json(f"{output_dir}/nt_prompt_score.jsonl", orient="records", lines=True)



def run_toxigen(limit=2000000):
    data_all = pd.read_parquet('data/2_toxigen-data/annotated/train-00000-of-00001.parquet')
    t_prompt_score = []
    nt_prompt_score = []
    for row in tqdm(data_all.iterrows(), total=len(data_all)):
        if row[0] >= limit:
            break
        prompt = row[1]['text']
        score = row[1]['toxicity_human']
        if score >= 3:
            t_prompt_score.append((prompt, my_score_func(prompt)))
        elif score <=2:
            nt_prompt_score.append((prompt, my_score_func(prompt)))

    return post_process(t_prompt_score, nt_prompt_score)


def run_toxicchat(limit=2000000):
    data_all = pd.read_csv('data/3_toxic-chat/data/0124/toxic-chat_annotation_train.csv')
    t_prompt_score = []
    nt_prompt_score = []
    for row in tqdm(data_all.iterrows(), total=len(data_all)):
        if row[0] >= limit:
            break
        prompt = row[1]['user_input']
        if len(prompt) > 1000:
            continue
        toxic = row[1]['toxicity']
        if toxic == 1:
            t_prompt_score.append((prompt, my_score_func(prompt)))
        else:
            nt_prompt_score.append((prompt, my_score_func(prompt)))

    return post_process(t_prompt_score, nt_prompt_score)



def run_realtoxicity( limit=2000000):

    data_toxic=pd.read_json("data/1_real-toxicity-prompts/toxic_train_4k.jsonl", lines=True)
    data_notoxic=pd.read_json("data/1_real-toxicity-prompts/non_toxic_train_4k.jsonl",lines=True)

    t_prompt_score = []
    nt_prompt_score = []
    counts = 0
    for t_row, nt_row in tqdm(zip(data_toxic.iterrows(), data_notoxic.iterrows()),total=len(data_toxic)):
        if counts > limit:
            break
        toxic_prompt = t_row[1]['prompt']['text']
        notoxic_prompt = nt_row[1]['prompt']['text']

        toxic_score = t_row[1]['prompt']['toxicity']
        notoxic_score = nt_row[1]['prompt']['toxicity']

        t_prompt_score.append((toxic_prompt, toxic_score))
        nt_prompt_score.append((notoxic_prompt, notoxic_score))

        counts += 1
    return post_process(t_prompt_score, nt_prompt_score)


def run():
    run_1dataset()

if __name__ == '__main__':
    run()
