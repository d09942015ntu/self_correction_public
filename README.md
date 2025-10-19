# Explanation_ISC
We provide an explanation of how a large language model (LLM) responds to instructions, with a particular focus on its intrinsic self-correction behavior.

## Requirement
You need to install Python packages torch, transformers, scikit-learn, random, numpy, matplotlib, json, tqdm and so on. 

## Experiments

### Difference Vector 
- diff0_create_split.py: Create train/test split for the dataset.
- diff1_detoxify_text.py: Generate difference vectors, scored by detoxify package.
- diff1_roberta_text.py: Generate difference vectors, scored by roberta toxicity classifier.
  
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

