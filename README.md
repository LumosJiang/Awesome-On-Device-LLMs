<div align="center">

# Awesome On-Device LLMs

**A curated list of papers on on-device large language models, focusing on model compression and system optimization techniques.**

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Awesome-On-Device-LLMs&left_color=green&right_color=red)

</div>

## 📋 Contents

* [🔢 Model Quantization](#-model-quantization)
  * [Post-Training Quantization](#post-training-quantization)
  * [Quantization-Aware Training](#quantization-aware-training)
  * [Ultra-Low Bit Quantization](#ultra-low-bit-quantization)
* [✂️ Model Pruning](#-model-pruning)
  * [Structured Pruning](#structured-pruning)
  * [Unstructured Pruning](#unstructured-pruning)
* [🎓 Knowledge Distillation](#-knowledge-distillation)
  * [Rationale-based Distillation](#rationale-based-distillation)
  * [Uncertainty-aware KD](#uncertainty-aware-kd)
  * [Multi-teacher Distillation](#multi-teacher-distillation)
  * [Dynamic and Adaptive Strategies](#dynamic-and-adaptive-strategies)
  * [Task-specific and Foundations](#task-specific-and-foundations)
* [🔀 Low-Rank Factorization](#-low-rank-factorization)
  * [Training-Time Low-Rank (PEFT)](#training-time-low-rank-peft)
  * [Post-Training Low-Rank](#post-training-low-rank)
  * [Architectural Low-Rank and Linear Attention](#architectural-low-rank-and-linear-attention)
* [🔗 Hybrid Compression](#-hybrid-compression)
  * [Quantization + Sparsity](#quantization--sparsity)
  * [Quantization + Low-Rank](#quantization--low-rank)
  * [Pruning + Low-Rank](#pruning--low-rank)
  * [Quantization + Distillation](#quantization--distillation)
  * [Distillation + Pruning](#distillation--pruning)
  * [Distillation + Low-Rank](#distillation--low-rank)

---

## 🔢 Model Quantization

### Post-Training Quantization

+ **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Aug 2022, NeurIPS'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.07339)
  [![Code](https://img.shields.io/github/stars/TimDettmers/bitsandbytes.svg?style=social&label=Star)](https://github.com/TimDettmers/bitsandbytes)

+ **ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers** (Jun 2022, NeurIPS'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.01861)

+ **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Oct 2022, ICLR'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.17323)
  [![Code](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social&label=Star)](https://github.com/IST-DASLab/gptq)

+ **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (Nov 2022, ICML'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.10438)
  [![Code](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social&label=Star)](https://github.com/mit-han-lab/smoothquant)

+ **AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration** (Jun 2023, MLSys'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00978)
  [![Code](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social&label=Star)](https://github.com/mit-han-lab/llm-awq)

+ **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** (Aug 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.13137)

+ **FPTQuant: Function-Preserving Transforms for LLM Quantization** (Jun 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.04985)

+ **FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design** (Aug 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.04405)

+ **KVQuant: Towards 10M-Token Context** (Jan 2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.18079)

+ **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead** (Jun 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.03482)

### Quantization-Aware Training

+ **PACT: Parameterized Clipping Activation for Quantized Neural Networks** (May 2018, ICLR'18)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1805.06085)

+ **HAWQV3: Dyadic Neural Network Quantization** (Nov 2020, arXiv'21)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.10680)

+ **Low-Rank Quantization-Aware Training for LLMs** (Jun 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.06385)

+ **AutoMPQ: Automatic Mixed-Precision Neural Network Search via Few-Shot Quantization Adapter** (2024, TETCI'24)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://ieeexplore.ieee.org/document/10523945)

+ **BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation** (Aug 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2024.acl-long.259/)

+ **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models** (Jul 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.11062)

+ **Precision Neural Network Quantization via Learnable Adaptive Modules** (Apr 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.17263)

+ **Stabilizing Quantization-Aware Training by Implicit-Regularization on Hessian Matrix** (Mar 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.11159)

### Ultra-Low Bit Quantization

+ **SqueezeLLM: Dense-and-Sparse Quantization** (Jun 2023, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07629)

+ **Extreme Compression of Large Language Models via Additive Quantization** (Jan 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.06118)

+ **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks** (Feb 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.04396)

+ **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** (Feb 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.17764)

+ **LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid** (Jul 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.10032)

+ **Treasures in Discarded Weights for LLM Quantization** (Apr 2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/34376)

+ **Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models** (Jul 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2025.acl-long.1382/)

---

## ✂️ Model Pruning

### Structured Pruning

+ **LLM-Pruner: On the Structural Pruning of Large Language Models** (2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/file/44956951349095f74492a5471128a7e0-Paper-Conference.pdf)

+ **FLAP: Fluctuation-based Adaptive Structured Pruning for Large Language Models** (2024, AAAI'24)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://doi.org/10.1609/aaai.v38i10.28960)

+ **SlimGPT: Layer-wise Structured Pruning for Large Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c1c44e46358e0fb94dc94ec495a7fb1a-Paper-Conference.pdf)

+ **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf)

+ **APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference** (Jul 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/zhao24g.html)

+ **LaCo: Large Language Model Pruning via Layer Collapse** (Nov 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/EMNLP-2E8B57.svg)](https://aclanthology.org/2024.findings-emnlp.372/)

+ **DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/file/84a7fc24ed52e8eff514c33e8ac76ea3-Paper-Conference.pdf)

+ **SlimLLM: Accurate Structured Pruning for Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://icml.cc/virtual/2025/poster/46559)

+ **Olica: Efficient Structured Pruning of Large Language Models without Retraining** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=hhhcwCgyM1)

+ **GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching** (Jun 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.20480)

+ **Let LLM Tell What to Prune and How Much to Prune** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=zFR5aWGaUs)

+ **Instruction-Following Pruning for Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=juARG7yu4P)

+ **Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=WOt1owGfuN)

+ **Runtime Adaptive Pruning for LLM Inference** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.17138)

### Unstructured Pruning

+ **SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot** (Jul 2023, ICML'23)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v202/frantar23a.html)

+ **A Simple and Effective Pruning Approach for Large Language Models** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/14c856c7a41297804de4c4890e846b25-Paper-Conference.pdf)

+ **Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/0147d967a5db3b8dde08d2a327b24568-Paper-Conference.pdf)

+ **One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models** (2024, ICASSP'24)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://doi.org/10.1109/ICASSP48485.2024.10445737)

+ **SparseLLM: Towards Global Pruning of Pre-trained Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/522134ee1c52c7a2b929bc87cfe1781c-Abstract-Conference.html)

+ **DLP: Dynamic Layerwise Pruning in Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=11id5ppGZ8)

+ **Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining** (Aug 2025, IEEE AICCSA'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.15828)

+ **Improved Methods for Model Pruning and Knowledge Distillation** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.14052)

+ **Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning** (Sep 2025, EMNLP'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.08255)

+ **Detecting and Pruning Prominent but Detrimental Neurons in Large Language Models** (2025, COLM'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=cRE1XrHf1h)

+ **ICP: Immediate Compensation Pruning for Mid-to-high Sparsity** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://doi.org/10.1109/CVPR52734.2025.00886)

---

## 🎓 Knowledge Distillation

### Rationale-based Distillation

+ **Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.02301)

+ **Orca: Progressive Learning from Complex Explanation Traces of GPT-4** (Jun 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02707)

+ **Orca 2: Teaching Small Language Models How to Reason** (Nov 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.11045)

+ **Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.01582)

+ **MCC-KD: Multi-CoT Consistent Knowledge Distillation** (Oct 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.14747)

+ **SCOTT: Self-Consistent Chain-of-Thought Distillation** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.01879)

+ **Distilling Reasoning Capabilities into Smaller Language Models** (Jun 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00917)

+ **Mixed Distillation Helps Smaller Language Model Better Reasoning** (Dec 2023, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.10730)

+ **Keypoint-based Progressive Chain-of-Thought Distillation for LLMs** (Jun 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02657)

+ **Learning to Maximize Mutual Information for Chain-of-Thought Distillation** (Jun 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.03348)

+ **Merge-of-Thought Distillation** (Sep 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.08814)

+ **Mitigating Spurious Correlations Between Question and Answer via Chain-of-Thought Correctness Perception Distillation** (Sep 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.05602)

+ **Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks** (Sep 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13203)

+ **On the Generalization vs Fidelity Paradox in Knowledge Distillation** (Dec 2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.12515)

+ **From Models to Microtheories: Distilling a Model's Topical Knowledge for Grounded Question Answering** (Oct 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.03726)

### Uncertainty-aware KD

+ **MiniLLM: Knowledge Distillation of Large Language Models** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08543)

+ **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.13649)

+ **f-Distill: f-Divergence Minimization for Sequence-Level Knowledge Distillation** (Jul 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.15190)

+ **SUNGen: Self-Guided Noise-Free Data Generation** (May 2023, ICLR'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.12679)

+ **TDG: Targeted Data Generation - Finding and Fixing Model Weaknesses** (Jun 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17804)

+ **ToD-Robust: To Distill or Not to Distill? On the Robustness of Robust Knowledge Distillation** (Jul 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.09758)

+ **TA-KD: Teaching-Assistant-in-the-Loop: Improving Knowledge Distillation** (May 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.02116)

+ **Bayesian KD: Bayesian Knowledge Distillation with Uncertainty Quantification** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/fang24a.html)

+ **ToDi: Token-wise Distillation via Fine-Grained Divergence Control** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16297)

### Multi-teacher Distillation

+ **Human + GPT-3 Expert Supervision: Want To Reduce Labeling Cost? GPT-3 Can Help** (Aug 2021, ACL'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.13487)

+ **Is GPT-3 a Good Data Annotator?** (Dec 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10450)

+ **FuseLLM: Knowledge Fusion of Large Language Models** (Jan 2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.10491)

+ **MTKD-RL: Multi-Teacher Knowledge Distillation with Reinforcement Learning** (2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.09445)

+ **DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search** (2025, SIGIR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.05997)

+ **EKD4Rec: Ensemble Knowledge Distillation from LLM-based Models to Sequential Recommenders** (2025, WWW'25)  
  [![Paper](https://img.shields.io/badge/WWW-0066CC.svg)](https://doi.org/10.1145/3701716.3715527)

### Dynamic and Adaptive Strategies

+ **SAKD: Spot-Adaptive Knowledge Distillation** (2022, TIP'22)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://doi.org/10.1109/TIP.2022.3146653)

+ **Cost-effective Distillation of Large Language Models** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2023.findings-acl.507/)

+ **On-Policy KD: Learning from Self-Generated Mistakes** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.13649)

+ **Lion: An Empirically Optimized Approach to Align Language Models** (Jul 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.06542)

+ **PromptKD: Unsupervised Prompt Distillation for Vision-Language Models** (Apr 2024, CVPR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.02781)

+ **Dual-Space KD: Dual-Space Knowledge Distillation for Large Language Models** (Jun 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.17328)

+ **DistiLLM: Streamlined Distillation for Large Language Models** (Feb 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03898)

+ **AMM: Adversarial Moment-Matching Distillation of Large Language Models** (Jun 2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02959)

+ **DDK: Distilling Domain Knowledge for Efficient Large Language Models** (Jun 2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.11498)

+ **Markov KD: Knowledge Distillation from a Nasty Teacher** (2024, ECCV'24)  
  [![Paper](https://img.shields.io/badge/Springer-004D99.svg)](https://doi.org/10.1007/978-3-031-73024-5_10)

+ **POCL: Being Strong Progressively! Curriculum Learning Framework for KD** (Jan 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.04193)

+ **Rethink KL: Rethinking Kullback-Leibler Divergence in Knowledge Distillation** (Apr 2024, COLING'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02657)

+ **MoE KD: LLaVA-MoD - Making LLaVA Tiny via MoE Knowledge Distillation** (Dec 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.15881)

+ **Hybrid DF-KD: Hybrid Data-Free Knowledge Distillation** (Dec 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.13525)

+ **AlignFD: Beyond Logits - Aligning Feature Dynamics for Effective KD** (2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2025.acl-long.1030/)

+ **PTKD: Pre-training Distillation for Large Language Models** (Oct 2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16215)

### Task-specific and Foundations

+ **MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning** (Sep 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.05653)

+ **PersDistill: Personalised Distillation - Empowering Open-Sourced LLMs** (Aug 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.02226)

+ **VanillaKD: Revisit the Power of Vanilla Knowledge Distillation** (May 2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.15781)

+ **SKR: Self-Knowledge Guided Retrieval Augmentation for Large Language Models** (Oct 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05002)

+ **DistillSeq: Safety-Alignment Testing of Large Language Models via KD** (2024, ISSTA'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.05087)

+ **WizardCoder: Empowering Code Large Language Models with Evol-Instruct** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08568)

+ **PGKD: Performance-Guided LLM Knowledge Distillation** (Sep 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.17998)

+ **AdvDistill: Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation** (Jul 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.00054)

+ **SparseLS: Sparse Logit Sampling - Accelerating Knowledge Distillation** (2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.10947)

---

## 🔀 Low-Rank Factorization

### Training-Time Low-Rank (PEFT)

+ **LoRA: Low-Rank Adaptation of Large Language Models** (2022, ICLR'22)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=nZeVKeeFYf9)
  [![Code](https://img.shields.io/github/stars/microsoft/LoRA.svg?style=social&label=Star)](https://github.com/microsoft/LoRA)

+ **AdaLoRA: Adaptive Low-Rank Adaptation** (Mar 2023, ICLR'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.10512)

+ **QLoRA: Efficient Finetuning of Quantized LLMs** (2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=OUIFPHEgJU)
  [![Code](https://img.shields.io/github/stars/artidoro/qlora.svg?style=social&label=Star)](https://github.com/artidoro/qlora)

+ **rsLoRA: A Rank Stabilization Scaling Factor for LoRA** (Dec 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.03732)

+ **ReLoRA: Train High-Rank Networks via Low-Rank Updates** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=1_4j0LQJnZ)

+ **LongLoRA: Efficient Finetuning for Long-Context LLMs** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=2rrn7j4EqV)

+ **Bayesian-LoRA: Bayesian Low-Rank Adaptation** (Aug 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.13111)

+ **LoRA+: Efficient Low Rank Adaptation of Large Models** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/hayou24a.html)

+ **DoRA: Weight-Decomposed Low-Rank Adaptation** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=XcCaHBDTwt)

+ **AutoLoRA: Automatic Rank Search for Low-Rank Adaptation** (2024, NAACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.09113)

+ **PiSSA: Principal Singular Values and Singular Vectors Adaptation** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02948)

+ **O-LoRA: Orthonormal Low-Rank Adaptation of Large Language Models** (Jun 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01775)

+ **Delta-LoRA: Fine-Tuning High-Rank Parameters with the Delta of Low-Rank Matrices** (Sep 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.02411)

+ **KronA: Parameter Efficient Tuning with Kronecker Adapter** (2024, CVPR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10650)

+ **dEBORA: Efficient Bilevel Optimization-based Low-Rank Adaptation** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=l8MJbQ9K62)

+ **SALR: Efficient Learning with Sine-Activated Low-Rank Matrices** (2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13361)

+ **LoRA-Pro: Are Low-Rank Adapters Properly Optimized?** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=pm2XlXJ1ZB)

+ **Lily: Low-rank Interconnected Adaptation Across Layers** (2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.16956)

+ **DenseLoRA: Dense Low-Rank Adaptation of Large Language Models** (Jan 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.03109)

### Post-Training Low-Rank

+ **LPLR: Low-Precision Plus Low-Rank Compression** (2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.12047)

+ **LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=LzPWWPAdY4)

+ **CALDERA: Joint Low-Precision and Low-Rank Compression** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.00630)

+ **QDyLoRA: Dynamic Rank with Quantization for Stability at Low Bits** (Oct 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10462)

+ **LLRC: Differentiable Rank Selection for Post-Training Low-Rank** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.19132)

+ **SVD-LLM: Truncation-Aware Singular Value Decomposition for Large Language Models** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=1qCcKw5qvD)

### Architectural Low-Rank and Linear Attention

+ **ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations** (2020, ICLR'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1909.11942)

+ **Linformer: Self-Attention with Linear Complexity** (2020, ICML'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.04768)

+ **Performer: Rethinking Attention with Performers** (2020, NeurIPS'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2009.14794)

+ **Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention** (2021, AAAI'21)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.03902)

+ **Monarch: Expressive Structured Matrices for Efficient Transformers** (2022, ICML'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.00595)

+ **RetNet: Retentive Network - A Successor to Transformer** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=UU9Icwbhin)

+ **Maestro: Trainable Low-Rank Decomposition to Uncover Architectural Low-Rank** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.12843)

+ **WD-LRA: Weight Decay Induces Low-Rank Attention Layers** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.10947)

+ **RALA: Breaking the Low-Rank Dilemma of Linear Attention** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.02049)

+ **MFA: Multi-Matrix Factorization Attention for KV Cache Reduction** (2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.11027)

---

## 🔗 Hybrid Compression

### Quantization + Sparsity

+ **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.03078)

+ **SqueezeLLM: Dense-and-Sparse Quantization** (Jun 2023, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07629)

+ **JSQ: Compressing Large Language Models by Joint Sparsification and Quantization** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://openreview.net/forum?id=wQFD6w9Cvk)

+ **Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23416)

### Quantization + Low-Rank

+ **LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning** (Nov 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.12023)

+ **QA-LoRA: Quantization-Aware Low-Rank Adaptation** (2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.14717)

+ **LQER: Low-Rank Quantization Error Reconstruction for LLMs** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.02446)

+ **DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for LLMs** (Sep 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.09223)

+ **Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition** (Jun 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.02077)

+ **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-bit Diffusion Models** (Nov 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.05007)

### Pruning + Low-Rank

+ **LoRAPrune: Structured Pruning with Low-Rank Compensation** (2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18403)

+ **MoE-I²: MoE-oriented Pipeline Combining Inter-expert Pruning and Intra-expert Low-rank** (2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.12013)

+ **SLTrain: Sparse Plus Low-Rank Parameterization During Pretraining** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02214)

### Quantization + Distillation

+ **LLM-QAT: Data-Free Quantization-Aware Training with Self-Distillation** (2023, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17888)

+ **BitDistiller: QAT with Self-Distillation for Sub-4-bit LLMs** (Aug 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2024.acl-long.259/)

+ **Optimizing Quantized Diffusion Models via Distillation with Cross-Timestep Error Correction** (2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.15935)

### Distillation + Pruning

+ **EfficientVLM: Fast and Accurate Vision-Language Models via KD and Modal-Adaptive Pruning** (Oct 2022, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.07795)

+ **EPSD: Early Pruning with Self-Distillation for Efficient Model Compression** (2024, AAAI'24)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/29108)

+ **IEPD-LMM: Large Multimodal Model Compression via Iterative Efficient Pruning and Distillation** (2024, WWW'24)  
  [![Paper](https://img.shields.io/badge/WWW-0066CC.svg)](https://doi.org/10.1145/3589335.3651461)

+ **Compact LMs: Best-Practice Guide Combining Pruning with Knowledge Distillation** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.14679)

### Distillation + Low-Rank

+ **OPDF: Over-parameterized Distillation via Tensor Decomposition** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=fT1RkAgrC3)

---

## 📊 Statistics

- **Model Quantization**: 27 methods (PTQ: 10, QAT: 8, Ultra-low bit: 7, KV cache: 2)
- **Model Pruning**: 25 methods (Structured: 14, Unstructured: 11)
- **Knowledge Distillation**: 61 methods
  - Rationale-based Distillation: 14 methods
  - Uncertainty-aware KD: 9 methods
  - Multi-teacher Distillation: 6 methods
  - Dynamic and Adaptive Strategies: 16 methods
  - Task-specific and Foundations: 9 methods
- **Low-Rank Factorization**: 35 methods (PEFT: 19, Post-training: 6, Architectural: 10)
- **Hybrid Compression**: 21 methods
  - Quantization + Sparsity: 4 methods
  - Quantization + Low-Rank: 6 methods
  - Pruning + Low-Rank: 3 methods
  - Quantization + Distillation: 3 methods
  - Distillation + Pruning: 4 methods
  - Distillation + Low-Rank: 1 method

**Total**: 169 papers covering model compression and system optimization for on-device LLMs

---

> **Note**: This list is continuously updated. Contributions are welcome! Please feel free to open an issue or pull request to add new papers. Code and project links are provided where publicly available.
