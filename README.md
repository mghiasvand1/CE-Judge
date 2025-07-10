<p>This repository contains the code implementation for the paper titled "<a href="https://arxiv.org/pdf/2507.06774">Checklist Engineering Empowers Multilingual LLM Judges</a>".</p>

### Abstract
Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators—a trend known as the LLM-asa-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.

### Usage
Here is a sample code snippet for using our framework in inference and evaluation modes.
```
python main.py inference --dataset mmeval --lang es --type pairwise --api_key <your_novita_api_key>
python main.py evaluate --label_file labels/mmeval_es_reasoning.txt --type pairwise
```

### Citation
If you find our paper useful for your work or research, please kindly cite it:
```
@article{mohammadkhani2025checklist,
  title={Checklist Engineering Empowers Multilingual LLM Judges},
  author={Mohammadkhani, Mohammad Ghiasvand and Beigy, Hamid},
  journal={arXiv preprint arXiv:2507.06774},
  year={2025}
}
```
