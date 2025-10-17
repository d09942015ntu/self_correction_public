# required packages:
import os.path
import pandas as pd
import pickle
import numpy as np
import torch
import time
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
import glob
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import joblib
from utils import parse_hidden_states

def pca_process(pca,last_hidden,initial = 0):
    flat_hidden = last_hidden.reshape((last_hidden.shape[0]*last_hidden.shape[1],last_hidden.shape[2]),)
    if initial == 0:
        pca_results = pca.fit_transform(flat_hidden)
    else: 
        pca_results = pca.transform(flat_hidden)
    pca_results  = pca_results.reshape((last_hidden.shape[0],last_hidden.shape[1],pca_results.shape[-1]),)
    return pca_results

def plot_3dim_vector(input_points, condition, output_dir, k):
    for L in range(len(input_points)):
        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, facecolors='none', edgecolors='grey', s=50, linewidths=1.5, label='Origin')
        vectors = np.array(input_points[L])
        norm_vectors  = normalize(vectors)
        # print(f"input_points[L] = {input_points[L]}")
        ax.view_init(elev=30, azim=80) #elev=30, azim=45
        origin = np.zeros((len(norm_vectors), 3))

        if condition[:8] == "positive":
            if k == 1: 
                vector_color = 'seagreen' 
                line_width=1.2 
                Alpha = 0.6
                if L == 0:
                    arrow_length = 0.01
                else: 
                    arrow_length = 0.1
            elif k == 2:
                line_width = 0.4
                vector_color = '#A7C7E7'
                arrow_length = 0.1
                Alpha = 0.7
            else:
                line_width = 0.4
                vector_color = '#1F3A93'
                arrow_length = 0.1
                Alpha = 0.5
        else: 
            if k == 1: 
                vector_color = 'seagreen'
                line_width=1.2 
                Alpha = 0.6
                if L == 0:
                    arrow_length = 0.01
                else: 
                    arrow_length = 0.1
            elif k == 2:
                line_width = 0.4
                vector_color = '#F28B82'
                arrow_length = 0.1
                Alpha = 0.7
            else:
                line_width = 0.4
                vector_color = '#800020'
                arrow_length = 0.1
                Alpha = 0.4
        ax.quiver(
            origin[:, 0], origin[:, 1], origin[:, 2],
            norm_vectors[:, 0], norm_vectors[:, 1], norm_vectors[:, 2],
            color=vector_color, alpha=Alpha, linewidth=line_width, arrow_length_ratio = arrow_length
        )
        tick_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_zticks(tick_vals)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))
        margin_ratio = 1.1  
        for dim, set_lim in zip(range(3), [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            max_abs = np.max(np.abs(norm_vectors[:, dim]))
            lim = max_abs * margin_ratio
            set_lim(-lim, lim)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"test_3d_plot_{condition}{L}.png"), bbox_inches='tight', dpi=300)#, bbox_inches='tight'
        plt.close()

def plot_with_SVD(differences, condition, output_dir, k=2):
    diff_svd = []
    num_rounds = differences.shape[1]+1
    for rounds in differences:
        U, S, VT = np.linalg.svd(rounds)
        print(f"sig value = {np.diag(S)}")
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        VT_k = VT[:k, :]

        rounds_k = U_k @ S_k @ VT_k
        diff_svd.append(rounds_k)
    return plot_3dim_vector(diff_svd, condition, output_dir,k)


def run():
    parser = ArgumentParser()
    parser.add_argument("--num_data",type=int,default=500)
    parser.add_argument("--num_rounds",type=int,default=5)
    parser.add_argument("--model_name", type=str, default="alignment-handbook/zephyr-7b-sft-full")
    args = parser.parse_args()


    base_dir = os.path.join("outputs", os.path.basename(args.model_name))
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    last_hidden_all = []
    for condition in ["positive", "negative"]:
        cache_dir = os.path.join(base_dir, "hidden_states", condition)

        if os.path.exists("shared_pca.joblib"):
            pca = joblib.load("shared_pca.joblib")
            print("Use existed PCA.")
        else:
            pca = PCA(n_components=3)
            joblib.dump(pca, "shared_pca.joblib")
            print("PCA saved.")

        # Positive
        _, _, last_hidden, _ = parse_hidden_states(args.num_data, args.num_rounds, cache_dir)
        last_hidden_all.append(last_hidden)

    last_hidden_all = np.concatenate(last_hidden_all)
    pca_results_all = pca_process(pca,last_hidden_all,0)

    for i, condition in enumerate(["positive", "negative"]):
        pca_result = pca_results_all[i*(args.num_data):(i+1)*(args.num_data)]
        diff1 = pca_result[:,1:,:] - pca_result[:,:-1,:]
        diff1 = diff1.transpose((1,0,2))

        plot_with_SVD(diff1 , f"{condition}(rank = 2)", output_dir, k=2)
        plot_with_SVD(diff1 , f"{condition}(rank = 1)", output_dir, k=1)
        plot_3dim_vector(diff1 ,f"{condition}(rank = 3)", output_dir, k=3)

        #length2, result2, last_hidden2, last_hidden_mean2 = parse_hidden_states(args.num_data, args.num_rounds, args.output_dir2)
        #pca_results2 = pca_process(pca,last_hidden2,1)
        #diff2 = pca_results2[:,1:,:] - pca_results2[:,:-1,:]
        #diff2 = diff2.transpose((1,0,2))

        #plot_with_SVD(diff2 , "Negative(rank = 2)", args.output_dir2, k=2)
        #plot_with_SVD(diff2 , "Negative(rank = 1)", args.output_dir2, k=1)
        #plot_3dim_vector(diff2 ,"Negative(rank = 3)", args.output_dir2, k=3)
        #pass

if __name__ == '__main__':
    run()
    pass
