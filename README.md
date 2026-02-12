# TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models (ICLR 2026)

The official repository for "TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models".

<p align="center">

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Stephengzk/TPRU-25k">TPRU 25k</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/Stephengzk/TPRU-test">TPRU-test</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://openreview.net/forum?id=crOvAD9MPA&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)">Paper (ICLR 2026)</a>&nbsp&nbsp
</p>

<p align="center">
       🤗 <a href="https://huggingface.co/Stephengzk/TPRU-3B">TPRU-3B</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Stephengzk/TPRU-7B">TPRU-7B<a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Stephengzk/TPRU-32B">TPRU-32B</a>&nbsp&nbsp |  &nbsp&nbsp📑 <a href="https://openreview.net/forum?id=crOvAD9MPA&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)">Arxiv Paper</a>&nbsp&nbsp
</p>

## News

- [2026/02/12] **We released [RL Dataset (Vision-R1-rl)](https://huggingface.co/datasets/Osilly/Vision-R1-rl), [Vision-R1-CI-7B](https://huggingface.co/Osilly/Vision-R1-CI-7B), [Vision-R1-32B](https://huggingface.co/Osilly/Vision-R1-32B), [Vision-R1-72B](https://huggingface.co/Osilly/Vision-R1-72B)** !
- [2026/01/26] **TPRU has been accepted by ICLR 2026!** 
- [2025/09/26] We introduce **TPRU**, a large-scale dataset designed to cultivate temporal reasoning in MLLMs via three complementary tasks: *Temporal Reordering*, *Next-Frame Prediction*, and *Previous-Frame Review*.



## Introduction

Multimodal Large Language Models (MLLMs), particularly smaller variants, often exhibit a critical deficiency in understanding temporal and procedural visual data. This gap hinders their application in real-world embodied AI tasks like robotic manipulation and navigation.

To address this, we introduce **TPRU** (**T**emporal-**P**rocedural **R**easoning and **U**nderstanding), a dataset and training paradigm designed to bridge this gap. TPRU sources 24,750 high-quality training samples from diverse embodied scenarios (Robotic Manipulation, LEGO Assembly, GUI Navigation, etc.). By leveraging reinforcement learning (GRPO) with our specific temporal tasks, our **TPRU-7B** model achieves state-of-the-art results, significantly performing larger proprietary models like **GPT-4o** on procedural understanding benchmarks.

Figure 1: Overview of the TPRU dataset and task formulation. Unlike prior datasets, TPRU enforces active cross-modal validation through negative samples and structured temporal tasks.

## Dataset: TPRU

The TPRU dataset is systematically designed to enhance procedural logic through three core tasks:

1. **Temporal Reordering:** Reconstructing the correct sequence of shuffled frames.


2. **Next-Frame Prediction:** Predicting the immediate future state given a sequence.


3. **Previous-Frame Review:** deducing the prerequisite state given an outcome.



| Dataset Split        | Samples | Source Scenarios                            |
| -------------------- | ------- | ------------------------------------------- |
| **TPRU-25K (Train)** | 24,750  | Robotic Manipulation, LEGO, GUI, Navigation |

 |
| **TPRU-Test (Eval)** | 461 | Manually curated & verified challenging instances 

 |

## Performance

Our RL-finetuned **TPRU-7B** demonstrates massive improvements in temporal reasoning, outperforming significantly larger models.

### TPRU-Test Results

On our manually curated test set, TPRU-7B achieves **75.70%** accuracy, surpassing GPT-4o (67.68%) and Gemini-1.5-Flash (65.35%).

### Generalization on Public Benchmarks

TPRU-7B also generalizes effectively to established benchmarks, showing significant gains on **MuirBench** and **LEGO-Puzzles** without degrading general capabilities.

| Model                | TPRU-Test  | MuirBench (Overall) | LEGO-Puzzles (Overall) |
| -------------------- | ---------- | ------------------- | ---------------------- |
| Qwen2.5-VL-7B (Base) | 50.33%     | 58.35%              | 36.5%                  |
| **TPRU-7B (Ours)**   | **75.70%** | **65.04%**          | **42.8%**              |
| GPT-4o               | 67.68%     | 68.00%              | 57.7%                  |

## Installation

```bash
git clone https://github.com/Stephen-gzk/TPRU.git

cd TPRU

conda create -n tpru python=3.10

conda activate tpru

pip install torch==2.6.0

pip install requirements.txt

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

## Training & Evaluation

We utilize the **Easy-R1** framework with Group-wise Preference Optimization (GRPO) for training.

### Training

To reproduce the TPRU-7B model using the TPRU-25K dataset:

```bash
# Example script for GRPO training
bash scripts/train_tpru_7b_grpo.sh

```

### Evaluation

To evaluate on TPRU-Test and other benchmarks using VLMEvalKit:

```bash
# Evaluate on TPRU-Test
bash scripts/eval_tpru_test.sh --model_path /path/to/tpru-7b

```

## Citation

If you find this repo or the TPRU dataset useful for your research, please consider citing our ICLR 2026 paper:

```bibtex
@inproceedings{gao2026tpru,
  title={TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models},
  author={Gao, Zhenkun and Wang, Xuhong and Tan, Xin and Xie, Yuan},
  booktitle={Published as a conference paper at ICLR 2026},
  year={2026}
}

```

## Acknowledgements

We thank the developers of [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Easy-R1](https://github.com/hiyouga/EasyR1), and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for their open-source contributions.

