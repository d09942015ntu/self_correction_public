# Explanation_ISC
We provide an explanation of how a large language model (LLM) responds to instructions, with a particular focus on its intrinsic self-correction behavior. In the experimental section, we use the Zephyr-7B-SFT model on the RealToxicityPrompts dataset, and this repository contains the code used to conduct our experiments.

## Requirement
You need to install Python packages torch, transformers, scikit-learn, random, numpy, matplotlib, json, tqdm and so on. 

## Experiments

### Difference Vector 
- diff0_hidden_states.py: Get the score of toxicity of each tokens.
- diff1_hidden_states.py: Extracts the input length, generated response, and hidden states from the LLM.
- diff2_rounds_score.py: Sends generated responses to the AWS Perspective API to obtain toxicity scores, corresponding to Table 1 and Figure 1.
- diff3_get_relation.py: Computes the inner product between the hidden state differences and the unembedding matrix across rounds, corresponding to Table 2.
- diff4_pca_results.py: Visualizes the differences in hidden states across rounds, corresponding to Figure 2.
  
### Steering Vector

- steering0_preprocess.py: pre-process the dataset, sort the dataset in increasing/decreasing toxicity 
- steering1_build.py: build the steering vector
- steering2_run.py: using steering vector to generate positive/negative completions
- steering3_scoring.py: show the score of positive/negative completions
- steering4_innerprod.py: calculate the inner product between steering vector and difference vector

How to run the result of steering vector

```sh
bash steering0_preprocess.sh
bash steering1_build.sh
bash steering2_run.sh
bash steering3_scoring.sh
bash steering4_innerprod.sh
```

### Others
- utils.py: Contains utility functions such as parse_hidden_states.

## TODO

- Find a toxicity (or non-toxicity) steering vector. we can select the optimal steering vector based on 
  - the cosine similarity between the steering vector and the prompting ”difference vectors”, 
  - its performance on detoxification.
- After such a vector is computed, you need to verify 
  - decrease/increase the performance of detoxification, 
  - it suppresses non-toxic tokens or promote toxic tokens (Note: For visualization, I thinkwe can use a heat map.)
  - the cosine similarity between the steering vector and the promptingdifference vector” is relatively high.
- We need to estimate the ”λ” in our formulation, the strength that the steering vector separates toxic and non-toxic tokens.
- Test robustness by trying a more diverse set of self-correction prompts.
- Provide more analysis on the prompting ”difference vectors” and the hidden states. How?
- Test other tasks and other models (and other features). 

