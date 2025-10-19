import argparse
import json
import os
import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor
from collections import defaultdict


def gen_hook_func_hidden_states(layer_idx, hidden_states):
    def get_hidden_states(module, input, output):
        if "Qwen2" in str(module) or "Qwen3" in str(module) or "zephyr" in str(module) or "Mistral" in str(module):
            hidden_states[layer_idx].append(output[0]) # (seq_len, hidden_size)
        elif "Gemma3" in str(module):
            hidden_states[layer_idx].append(output[0][0])
        else:
            assert 0
    return get_hidden_states


def init_model_hook_hidden_states(model):
    hidden_states = defaultdict(list)
    if "Qwen2" in str(model) or "Qwen3" in str(model) or "Llama" in str(model) or "Mistral" in str(model):
        for k, model_layer_k in enumerate(list(model.model.layers)):
            model_layer_k.register_forward_hook(gen_hook_func_hidden_states(layer_idx=k, hidden_states=hidden_states))
    elif "Gemma3" in str(model):
        for k, model_layer_k in enumerate(list(model.language_model.layers)):
            model_layer_k.register_forward_hook(gen_hook_func_hidden_states(layer_idx=k, hidden_states=hidden_states))
        # hook to be implemented for different models
        assert 0, f"Hook is not implemented for model {str(model)}"
    return hidden_states


def init_model(args, hook_hidden=True):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=False, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, ignore_mismatched_sizes=True,
                                                 local_files_only=False, token=access_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    if hook_hidden:
        hidden_states = init_model_hook_hidden_states(model)
    else:
        hidden_states = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, hidden_states


def dataset_preprocessing(args):
    t_prompts = []
    t_scores = []
    nt_prompts = []
    nt_scores = []
    for data_tag in args.data_tags:
        data_toxic=f"data_processed/{data_tag}/t_prompt_score.jsonl"
        data_notoxic=f"data_processed/{data_tag}/nt_prompt_score.jsonl"
        df_t = pd.read_json(data_toxic, lines=True)
        df_nt = pd.read_json(data_notoxic, lines=True)
        for (i_t, row_t), (i_nt, row_nt) in zip(df_t.iterrows(),df_nt.iterrows()):
            if i_t/len(df_t) < args.data_ratio:
                t_prompts.append(row_t['prompt'])
                t_scores.append(row_t['toxicity'])
            if i_nt / len(df_nt) < args.data_ratio:
                nt_prompts.append(row_nt['prompt'])
                nt_scores.append(row_nt['toxicity'])
    print(f"t_toxicity:{np.average(t_scores)}, nt_toxicity:{np.average(nt_scores)}")
    return t_prompts, nt_prompts


def get_embed_steering(args, model, tokenizer, hidden_states):
    dataset_tags = "".join(args.data_tags)
    dataset_ratio = int(args.data_ratio*100)
    output_dir = os.path.join(args.output_dir,os.path.basename(args.model_dir),f"steering_d{dataset_tags}_t{dataset_ratio}","1_vector")
    os.makedirs(output_dir,exist_ok=True)

    if "Qwen2" in str(model) or "Qwen3" in str(model) or "Llama" in str(model) or  "Mistral" in str(model):
        selected_layers = list(range(model.model.config.num_hidden_layers))
    elif "Gemma3" in str(model):
        selected_layers = list(range(len(model.language_model.layers)))
    else:
        assert 0

    t_prompts, nt_prompts = dataset_preprocessing(args)

    sep = 5

    embed_toxic = [defaultdict(int) for _ in range(sep)]
    embed_notoxic = [defaultdict(int) for _ in range(sep)]

    counts = 0

    max_len = 0

    # Count Max Len
    for prompt_toxic, prompt_notoxic in zip(t_prompts, nt_prompts):
        if counts >= args.limit:
            break
        ids_toxic = tokenizer.encode(prompt_toxic)
        ids_notoxic = tokenizer.encode(prompt_notoxic)
        max_len = max(max_len, len(ids_toxic),len(ids_notoxic))

    # Start Iteration
    total_line = min([args.limit, len(t_prompts), len(nt_prompts)])
    for prompt_toxic, prompt_notoxic in zip(t_prompts, nt_prompts):
        print(f"forward: {counts}/{total_line}")
        if counts >= args.limit:
            break

        ids_toxic = tokenizer.encode(prompt_toxic)
        ids_notoxic = tokenizer.encode(prompt_notoxic)
        ids_toxic = [[tokenizer.pad_token_id]*(max_len - len(ids_toxic)) + ids_toxic]
        ids_notoxic = [[tokenizer.pad_token_id]*(max_len - len(ids_notoxic)) + ids_notoxic]

        model.forward(torch.Tensor(ids_toxic).to(int).to(model.device))

        for layer, ascore in hidden_states.items():
            if layer in selected_layers:
                embed_toxic[counts%sep][layer] += (ascore[0].detach().cpu().numpy())
            ascore.clear()

        model.forward(torch.Tensor(ids_notoxic).to(int).to(model.device))

        for layer, ascore in hidden_states.items():
            if layer in selected_layers:
                embed_notoxic[counts%sep][layer] += (ascore[0].detach().cpu().numpy())
            ascore.clear()

        counts += 1

    steering_vec = defaultdict(int)
    steering_vec_sep = [defaultdict(int) for _ in range(sep)]

    for i in range(sep):
        for layer in embed_toxic[i].keys():
            embed_toxic[i][layer] = embed_toxic[i][layer]/counts
            embed_notoxic[i][layer] = embed_notoxic[i][layer]/counts
            steering_vec_layer = (embed_notoxic[i][layer] - embed_toxic[i][layer])
            steering_vec[layer] += (1/sep)*steering_vec_layer
            steering_vec_sep[i][layer] = steering_vec_layer.tolist()

    steering_vec = dict((x,y.tolist()) for x,y in steering_vec.items())


    json.dump(steering_vec, open(os.path.join(output_dir, f"steer.json"), "w"), indent=2)
    json.dump(steering_vec_sep, open(os.path.join(output_dir, f"steer_sep.json"), "w"), indent=2)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Models: 
    # Qwen3: "Qwen/Qwen3-4B-Instruct-2507" 
    # Qwen2.5: "Qwen/Qwen2.5-3B-Instruct" 
    # Mistral: "mistralai/Mistral-7B-Instruct-v0.3"
    # Gemma-3: "google/gemma-3-4b-it"
    # Zephyr: "HuggingFaceH4/zephyr-7b-beta"
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_tags", type=str, default=["1"], nargs="+")
    parser.add_argument("--data_ratio", type=float, default=1)
    parser.add_argument("--model_dir", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--limit", type=int, default=20000000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    seed_everything(0)


    model, tokenizer, hidden_states = init_model(args)

    get_embed_steering(args, model, tokenizer, hidden_states)


access_token = '<ACCESS_TOKEN>'
if __name__ == '__main__':
    run()
