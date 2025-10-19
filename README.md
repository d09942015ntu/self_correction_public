# Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability
This repository contains the official implementation for the paper **Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability**. 

⚠️Warning: Some data, prompts, and model outputs may contain toxic or offensive language.
## Requirements
Install the following Python packages to run the code: torch, transformers, scikit-learn, numpy, matplotlib, tqdm, and related dependencies.

## Experiments

### Prompt-Induced Shifts
- diff0_create_split.py: Creates train/test splits.
- diff1_detoxify_text.py: Conducts text detoxification and toxification and sample prompt-induced shifts; toxicity scored by Detoxify.
- diff1_roberta_text.py: Conducts text (de)toxification and sample prompt-induced shifts; toxicity scored by RoBERTa-toxicity-classifier.
  
### Steering Vector
- steering0_preprocess.py: Pre-process the dataset and sort the dataset in increasing/decreasing toxicity.
- steering1_build.py: Build the steering vectors.
- (Optional) steering2_run.py: Using steering vectors to generate positive/negative completions.
- (Optional) steering3_scoring.py: Plots the toxicity scores of positive/negative completions.
- steering4_cossim.py: Plots the consine similarity between steering vectors and prompt-induced shifts.

To run the experiments on steering vectors:
```sh
bash steering0_preprocess.sh
bash steering1_build.sh
# Optional:
# bash steering2_run.sh
# bash steering3_scoring.sh
bash steering4_innerprod.sh
```

### Others
- utils.py: Utility functions.

