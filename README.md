# SlimDoc
Code for the paper "SlimDoc: Lightweight Distillation of Document Transformer Models", which explores distillation of transformer based encoder models for document understanding.
Further it presents an unsupervised transitive distillation from an LLM into a teacher DU model and finally into a compact student model.

The repository supports the following models out of the box:
- LiLT with 12 layers ([SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base))
- LayoutLMv3 base with 12 layers ([microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base))

## Data
Our data is hosted on Google Drive [here](https://drive.google.com/file/d/1FCYc-Ur55d6HysoIYG3FnWcZPZbky9aL/view?usp=sharing).

It includes: 
- the base datasets 
  - DocVQA
  - InfographicsVQA
  - WikiTableQuestions
  - SROIE
  - FUNSD
- our custom subsets for extractive VQA of DocVQA, InfographicsVQA and WikiTableQuestions
- the [LAPDoc](https://github.com/marcel-lamott/LAPDoc) prompts and results for the unsupervised transitive distillation
- the files for small and tiny vocabularies

## Setup
1. Execute ```pip install -e .``` in the main directory and install at least the following packages:
   - tqdm
   - transformers
   - torch
   - Levenshtein
   - wandb
   - pandas
   - jsonlines
   - pdf2image
   - datasets

2. Download the data from the URL above and copy both contained folders into slimdoc/data

## Usage

### Train
Please refer to the documentations of the command line interfaces of the following scripts (by invoking them with ```-h``` parameter):
- ```train/train.py``` to fine-tune or distill a single model
- ```train/runner.py``` to fine-tune or distill a series of models (e.g. all 4-layer students)

### Evaluate
To evaluate a single model invoke ```eval/eval.py [RUN_NAME]```.

For evaluation of models trained on DocVQA/InfographicsVQA/WikiTableQuestions you also need to install https://github.com/due-benchmark/evaluator.

<!---
## Citation
If you use this code or find it helpful for your research, please consider citing our paper. This helps us continue supporting and maintaining the project.
```
@inproceedings{yourcitation2025,
  title     = {Your Paper Title},
  author    = {Author One and Author Two and Author Three},
  booktitle = {Proceedings of the XYZ Conference},
  year      = {2025},
  url       = {https://arxiv.org/abs/your_arxiv_id}  % or DOI if published
}
```
-->