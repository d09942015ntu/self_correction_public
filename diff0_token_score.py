# required packages:
import torch
import requests
import json
import os
import re
import csv
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser

def is_english_word(token):
    if token[0] == "▁" and len(token) > 2:
        return re.fullmatch(r"[a-zA-Z]+", token[1:]) is not None
    else:
        return re.fullmatch(r"[a-zA-Z]+", token) is not None
    
def get_toxicity(word):
    all_results = []
    command = [
        "aws", "comprehend", "detect-toxic-content",
        "--language-code", "en",
        "--text-segments", json.dumps([{"Text": f"This is {word}"}]),
        "--output", "json"
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)

            if "ResultList" in response and response["ResultList"]:
                scores = response["ResultList"][0].get("Labels", [])
                all_results.append(response["ResultList"][0].get("Toxicity", 0.0))
                for label_info in scores:
                    all_results.append(label_info['Score'])
                return all_results

        except json.JSONDecodeError:
            print(f"[JSON decode error for '{word}']:\n{result.stdout}")
            return None
    else:
        print(f"[AWS Error for '{word}']:\n{result.stderr}")
        return None

def run(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    unembedding = model.lm_head.weight
    vocab = tokenizer.get_vocab()
    #tokens = list(vocab.keys())
    tokens = sorted(vocab.keys(), key=lambda x: vocab[x])
    print(len(tokens))

    Labels = ['Token','Toxicity','PROFANITY','HATE_SPEECH','INSULT','GRAPHIC','HARASSMENT_OR_ABUSE','SEXUAL','VIOLENCE_OR_THREAT']
    with open(os.path.join('outputs', os.path.basename(args.model_name), 'tokens1.csv'),
              'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(Labels)
        for token in tokens:
            if is_english_word(token):
                toxicity_result = get_toxicity(token)
                if toxicity_result is not None:
                    Toxicity, PROFANITY, HATE_SPEECH, INSULT, GRAPHIC, HARASSMENT_OR_ABUSE, SEXUAL, VIOLENCE_OR_THREAT = toxicity_result
                    value = [token, Toxicity, PROFANITY, HATE_SPEECH, INSULT, GRAPHIC, HARASSMENT_OR_ABUSE, SEXUAL, VIOLENCE_OR_THREAT]
                    writer.writerow(value)
    print("storage as tokens1.csv")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name",type=str,default="alignment-handbook/zephyr-7b-sft-full")
    args = parser.parse_args()
    run(args)