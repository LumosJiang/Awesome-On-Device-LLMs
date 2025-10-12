<div align="center">

# Awesome On-Device Large Language Models

**A curated list of papers on on-device large language models, focusing on model compression and system optimization techniques.**

[![GitHub stars](https://img.shields.io/github/stars/LumosJiang/Awesome-On-Device-LLMs?style=social)](https://github.com/LumosJiang/Awesome-On-Device-LLMs/stargazers)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Awesome-On-Device-LLMs&left_color=green&right_color=red)

</div>

## 📋 Contents
* Model Compression
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
* System Optimization
  * [⚙️ Compiler Optimizations](#️-compiler-optimizations)
    * [Front-end & IR Layer](#front-end--ir-layer)
    * [Middle-end Layer](#middle-end-layer)
    * [Back-end Layer](#back-end-layer)
  * [🏗️ Inference Frameworks](#️-inference-frameworks)
  * [💾 Memory Optimization](#-memory-optimization)
  * [🔧 Hardware Support](#-hardware-support)
  * [☁️ Edge-Cloud Collaboration](#️-edge-cloud-collaboration)

---

## 🔢 Model Quantization

### Post-Training Quantization

+ **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Aug 2022, NeurIPS'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.07339)
  [![Code](https://img.shields.io/github/stars/TimDettmers/bitsandbytes.svg?style=social&label=Star)](https://github.com/TimDettmers/bitsandbytes)

+ **ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers** (Jun 2022, NeurIPS'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.01861)
  [![Code](https://img.shields.io/github/stars/deepspeedai/DeepSpeed.svg?style=social&label=Star)](https://github.com/deepspeedai/DeepSpeed)

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
  [![Code](https://img.shields.io/github/stars/OpenGVLab/OmniQuant.svg?style=social&label=Star)](https://github.com/OpenGVLab/OmniQuant)
+ **FPTQuant: Function-Preserving Transforms for LLM Quantization** (Jun 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.04985)

+ **FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design** (Aug 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.04405)
  [![Code](https://img.shields.io/github/stars/FlyFoxPlayer/FlexQ.svg?style=social&label=Star)](https://github.com/FlyFoxPlayer/FlexQ)
+ **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.18079)
  [![Code](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social&label=Star)](https://github.com/SqueezeAILab/KVQuant)

+ **QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead** (Jun 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.03482)
  [![Code](https://img.shields.io/github/stars/amirzandieh/QJL.svg?style=social&label=Star)](https://github.com/amirzandieh/QJL)
### Quantization-Aware Training

+ **PACT: Parameterized Clipping Activation for Quantized Neural Networks** (May 2018, ICLR'18)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1805.06085)

+ **HAWQV3: Dyadic Neural Network Quantization** (Nov 2020, arXiv'21)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.10680)
  [![Code](https://img.shields.io/github/stars/zhen-dong/hawq.svg?style=social&label=Star)](https://github.com/zhen-dong/hawq)
+ **Low-Rank Quantization-Aware Training for LLMs** (Jun 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.06385)
  [![Code](https://img.shields.io/github/stars/qualcomm-ai-research/LR-QAT.svg?style=social&label=Star)](https://github.com/qualcomm-ai-research/LR-QAT)
+ **AutoMPQ: Automatic Mixed-Precision Neural Network Search via Few-Shot Quantization Adapter** (2024, TETCI'24)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://ieeexplore.ieee.org/document/10523945)

+ **BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation** (Aug 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2024.acl-long.7)
  [![Code](https://img.shields.io/github/stars/DD-DuDa/BitDistiller.svg?style=social&label=Star)](https://github.com/DD-DuDa/BitDistiller)
+ **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models** (Jul 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.11062)
  [![Code](https://img.shields.io/github/stars/OpenGVLab/EfficientQAT.svg?style=social&label=Star)](https://github.com/OpenGVLab/EfficientQAT)
+ **Precision Neural Network Quantization via Learnable Adaptive Modules** (Apr 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.17263)

+ **Stabilizing Quantization-Aware Training by Implicit-Regularization on Hessian Matrix** (Mar 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.11159)

### Ultra-Low Bit Quantization

+ **SqueezeLLM: Dense-and-Sparse Quantization** (Jun 2023, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07629)
  [![Code](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM.svg?style=social&label=Star)](https://github.com/SqueezeAILab/SqueezeLLM)
+ **Extreme Compression of Large Language Models via Additive Quantization** (Jan 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.06118)

+ **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks** (Feb 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.04396)
  [![Code](https://img.shields.io/github/stars/Cornell-RelaxML/quip-sharp.svg?style=social&label=Star)](https://github.com/Cornell-RelaxML/quip-sharp)
+ **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** (Feb 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.17764)

+ **LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid** (Jul 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.10032)

+ **Treasures in Discarded Weights for LLM Quantization** (Apr 2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/34376)

+ **Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models** (Jul 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2025.acl-long.1382/)
  [![Code](https://img.shields.io/github/stars/snudm-starlab/UniQuanF.svg?style=social&label=Star)](https://github.com/snudm-starlab/UniQuanF)
---

## ✂️ Model Pruning

### Structured Pruning

+ **LLM-Pruner: On the Structural Pruning of Large Language Models** (2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/file/44956951349095f74492a5471128a7e0-Paper-Conference.pdf)
  [![Code](https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star)](https://github.com/horseee/LLM-Pruner)
+ **Fluctuation-based Adaptive Structured Pruning for Large Language Models** (2024, AAAI'24)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://doi.org/10.1609/aaai.v38i10.28960)
  [![Code](https://img.shields.io/github/stars/CASIA-IVA-Lab/FLAP.svg?style=social&label=Star)](https://github.com/CASIA-IVA-Lab/FLAP)
+ **SlimGPT: Layer-wise Structured Pruning for Large Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c1c44e46358e0fb94dc94ec495a7fb1a-Paper-Conference.pdf)

+ **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf)
  [![Code](https://img.shields.io/github/stars/princeton-nlp/LLM-Shearing.svg?style=social&label=Star)](https://github.com/princeton-nlp/LLM-Shearing)
+ **APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference** (Jul 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/zhao24g.html)
  [![Code](https://img.shields.io/github/stars/ROIM1998/APT.svg?style=social&label=Star)](https://github.com/ROIM1998/APT)
+ **LaCo: Large Language Model Pruning via Layer Collapse** (Nov 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/EMNLP-2E8B57.svg)](https://aclanthology.org/2024.findings-emnlp.372/)
  [![Code](https://img.shields.io/github/stars/yangyifei729/LaCo.svg?style=social&label=Star)](https://github.com/yangyifei729/LaCo)
+ **DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/file/84a7fc24ed52e8eff514c33e8ac76ea3-Paper-Conference.pdf)

+ **SlimLLM: Accurate Structured Pruning for Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://icml.cc/virtual/2025/poster/46559)

+ **Olica: Efficient Structured Pruning of Large Language Models without Retraining** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=hhhcwCgyM1)
  [![Code](https://img.shields.io/github/stars/BetterTMrR/LLM-Olica.svg?style=social&label=Star)](https://github.com/BetterTMrR/LLM-Olica)
+ **GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching** (Jun 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.20480)
  [![Code](https://img.shields.io/github/stars/Guinan-Su/auto-merge-llm.svg?style=social&label=Star)](https://github.com/Guinan-Su/auto-merge-llm)
+ **Let LLM Tell What to Prune and How Much to Prune** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=zFR5aWGaUs)

+ **Instruction-Following Pruning for Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=juARG7yu4P)

+ **Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=WOt1owGfuN)
  [![Code](https://img.shields.io/github/stars/Qi-Le1/Probe_Pruning.svg?style=social&label=Star)](https://github.com/Qi-Le1/Probe_Pruning)
+ **Runtime Adaptive Pruning for LLM Inference** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.17138)

### Unstructured Pruning

+ **SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot** (Jul 2023, ICML'23)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v202/frantar23a.html)
  [![Code](https://img.shields.io/github/stars/IST-DASLab/sparsegpt.svg?style=social&label=Star)](https://github.com/IST-DASLab/sparsegpt)
+ **A Simple and Effective Pruning Approach for Large Language Models** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/14c856c7a41297804de4c4890e846b25-Paper-Conference.pdf)
  [![Code](https://img.shields.io/github/stars/locuslab/wanda.svg?style=social&label=Star)](https://github.com/locuslab/wanda)
+ **Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/ICLR-006400.svg)](https://proceedings.iclr.cc/paper_files/paper/2024/file/0147d967a5db3b8dde08d2a327b24568-Paper-Conference.pdf)
  [![Code](https://img.shields.io/github/stars/zyxxmu/DSnoT.svg?style=social&label=Star)](https://github.com/zyxxmu/DSnoT)
+ **One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models** (2024, ICASSP'24)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://doi.org/10.1109/ICASSP48485.2024.10445737)
  [![Code](https://img.shields.io/github/stars/talkking/MixGPT.svg?style=social&label=Star)](https://github.com/talkking/MixGPT)
+ **SparseLLM: Towards Global Pruning of Pre-trained Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/522134ee1c52c7a2b929bc87cfe1781c-Abstract-Conference.html)
  [![Code](https://img.shields.io/github/stars/BaiTheBest/SparseLLM.svg?style=social&label=Star)](https://github.com/BaiTheBest/SparseLLM)
+ **DLP: Dynamic Layerwise Pruning in Large Language Models** (2025, ICML'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=11id5ppGZ8)
  [![Code](https://img.shields.io/github/stars/ironartisan/DLP.svg?style=social&label=Star)](https://github.com/ironartisan/DLP)
+ **Z-Pruner: Post-Training Pruning of Large Language Models for Efficiency without Retraining** (Aug 2025, IEEE AICCSA'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2508.15828)
  [![Code](https://img.shields.io/github/stars/sazzadadib/Z-Pruner.svg?style=social&label=Star)](https://github.com/sazzadadib/Z-Pruner)
+ **Improved Methods for Model Pruning and Knowledge Distillation** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.14052)

+ **Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning** (Sep 2025, EMNLP'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.08255)
  [![Code](https://img.shields.io/github/stars/secretflow/ACoLab.svg?style=social&label=Star)](https://github.com/secretflow/ACoLab)
+ **Detecting and Pruning Prominent but Detrimental Neurons in Large Language Models** (2025, COLM'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=cRE1XrHf1h)

+ **ICP: Immediate Compensation Pruning for Mid-to-high Sparsity** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/IEEE-0C479D.svg)](https://doi.org/10.1109/CVPR52734.2025.00886)

---

## 🎓 Knowledge Distillation

### Rationale-based Distillation

+ **Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.02301)
  [![Code](https://img.shields.io/github/stars/google-research/distilling-step-by-step.svg?style=social&label=Star)](https://github.com/google-research/distilling-step-by-step)
+ **Orca: Progressive Learning from Complex Explanation Traces of GPT-4** (Jun 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02707)

+ **Orca 2: Teaching Small Language Models How to Reason** (Nov 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.11045)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://www.microsoft.com/en-us/research/project/orca/)

+ **MCC-KD: Multi-CoT Consistent Knowledge Distillation** (Oct 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.14747)
  [![Code](https://img.shields.io/github/stars/homzer/MCC-KD.svg?style=social&label=Star)](https://github.com/homzer/MCC-KD)
+ **SCOTT: Self-Consistent Chain-of-Thought Distillation** (May 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.01879)
  [![Code](https://img.shields.io/github/stars/wangpf3/consistent-CoT-distillation.svg?style=social&label=Star)](https://github.com/wangpf3/consistent-CoT-distillation)
+ **Distilling Reasoning Capabilities into Smaller Language Models** (Jun 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00193)
  [![Code](https://img.shields.io/github/stars/kumar-shridhar/Distiiling-LM.svg?style=social&label=Star)](https://github.com/kumar-shridhar/Distiiling-LM)
+ **Mixed Distillation Helps Smaller Language Model Better Reasoning** (Dec 2023, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.10730)

+ **Keypoint-based Progressive Chain-of-Thought Distillation for LLMs** (Jun 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.16064)

+ **Learning to Maximize Mutual Information for Chain-of-Thought Distillation** (Jun 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.03348)
  [![Code](https://img.shields.io/github/stars/xinchen9/cot_distillation_ACL2024.svg?style=social&label=Star)](https://github.com/xinchen9/cot_distillation_ACL2024)
+ **Merge-of-Thought Distillation** (Sep 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.08814)

+ **Mitigating Spurious Correlations Between Question and Answer via Chain-of-Thought Correctness Perception Distillation** (Sep 2024, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.05602)

+ **Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks** (Sep 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13203)
  [![Code](https://img.shields.io/github/stars/Xnhyacinth/NesyCD.svg?style=social&label=Star)](https://github.com/Xnhyacinth/NesyCD)
+ **On the Generalization vs Fidelity Paradox in Knowledge Distillation** (Dec 2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15442)

+ **From Models to Microtheories: Distilling a Model's Topical Knowledge for Grounded Question Answering** (Oct 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.17701)
  [![Code](https://img.shields.io/github/stars/nweir127/microtheories.svg?style=social&label=Star)](https://github.com/nweir127/microtheories)
### Uncertainty-aware KD

+ **MiniLLM: Knowledge Distillation of Large Language Models** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08543)
  [![Code](https://img.shields.io/github/stars/microsoft/LMOps.svg?style=social&label=Star)](https://github.com/microsoft/LMOps)
+ **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.13649)

  **f-Divergence Minimization for Sequence-Level Knowledge Distillation**  (Jul 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.15190)
  [![Code](https://img.shields.io/github/stars/MANGA-UOFA/fdistill.svg?style=social&label=Star)](https://github.com/MANGA-UOFA/fdistill)
  **Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning**  (May 2023, ICLR'23)

  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.12679)

+ **Targeted Data Generation: Finding and Fixing Model Weaknesses** (Jun 2023, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17804)

+ **To Distill or Not to Distill? On the Robustness of Robust Knowledge Distillation** (Jul 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.09758)

+ **Teaching-Assistant-in-the-Loop: Improving Knowledge Distillation** (May 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.05322)

  **Bayesian Knowledge Distillation: A Bayesian Perspective of Distillation with Uncertainty Quantification**(2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/fang24a.html)

+ **ToDi: Token-wise Distillation via Fine-Grained Divergence Control** (May 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16297)
  [![Code](https://img.shields.io/github/stars/jungseongryong/ToDi.svg?style=social&label=Star)](https://github.com/jungseongryong/ToDi)
### Multi-teacher Distillation

+ **Want To Reduce Labeling Cost? GPT-3 Can Help** (Aug 2021, ACL'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2108.13487)

+ **Is GPT-3 a Good Data Annotator?** (Dec 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10450)

+ **FuseLLM: Knowledge Fusion of Large Language Models** (Jan 2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.10491)
  [![Code](https://img.shields.io/github/stars/fanqiwan/FuseAI.svg?style=social&label=Star)](https://github.com/fanqiwan/FuseAI)
+ **Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition** (2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.18510)
  [![Code](https://img.shields.io/github/stars/winycg/MTKD-RL.svg?style=social&label=Star)](https://github.com/winycg/MTKD-RL)
+ **DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search** (2025, SIGIR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.14609)
  [![Code](https://img.shields.io/github/stars/SimonLupart/disco-conv-splade.svg?style=social&label=Star)](https://github.com/SimonLupart/disco-conv-splade)
+ **EKD4Rec: Ensemble Knowledge Distillation from LLM-based Models to Sequential Recommenders** (2025, WWW'25)  
  [![Paper](https://img.shields.io/badge/WWW-0066CC.svg)](https://doi.org/10.1145/3701716.3715527)

### Dynamic and Adaptive Strategies

+ **SAKD: Spot-Adaptive Knowledge Distillation** (2022, TIP'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.02399)
  [![Code](https://img.shields.io/github/stars/zju-vipa/spot-adaptive-pytorch.svg?style=social&label=Star)](https://github.com/zju-vipa/spot-adaptive-pytorch)

+ **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.13649)

+ **Lion: An Empirically Optimized Approach to Align Language Models** (Jul 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.06542)
  [![Code](https://img.shields.io/github/stars/Columbia-NLP-Lab/LionAlignment.svg?style=social&label=Star)](https://github.com/Columbia-NLP-Lab/LionAlignment)
+ **PromptKD: Unsupervised Prompt Distillation for Vision-Language Models** (Apr 2024, CVPR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.02781)
  [![Code](https://img.shields.io/github/stars/zhengli97/PromptKD.svg?style=social&label=Star)](https://github.com/zhengli97/PromptKD)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://zhengli97.github.io/PromptKD/)
+ **Dual-Space KD: Dual-Space Knowledge Distillation for Large Language Models** (Jun 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.17328)
  [![Code](https://img.shields.io/github/stars/songmzhang/DSKD.svg?style=social&label=Star)](https://github.com/songmzhang/DSKD)
+ **DistiLLM: Streamlined Distillation for Large Language Models** (Feb 2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03898)
  [![Code](https://img.shields.io/github/stars/jongwooko/distillm.svg?style=social&label=Star)](https://github.com/jongwooko/distillm)
+ **Adversarial Moment-Matching Distillation of Large Language Models** (Jun 2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02959)
  [![Code](https://img.shields.io/github/stars/jiachenwestlake/MMKD.svg?style=social&label=Star)](https://github.com/jiachenwestlake/MMKD)
+ **DDK: Distilling Domain Knowledge for Efficient Large Language Models** (Jun 2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.16154)

+ **Markov Knowledge Distillation: Make Nasty Teachers Trained by Self-undermining Knowledge Distillation Fully Distillable** (2024, ECCV'24)  
  [![Paper](https://img.shields.io/badge/Springer-004D99.svg)](https://doi.org/10.1007/978-3-031-73024-5_10)

+ **Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework** (Jun 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.05695)
  [![Code](https://img.shields.io/github/stars/liuliuyuan6/POCL.svg?style=social&label=Star)](https://github.com/liuliuyuan6/POCL)
+ **Rethink KL: Rethinking Kullback-Leibler Divergence in Knowledge Distillation** (Apr 2024, COLING'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02657)
  [![Code](https://img.shields.io/github/stars/wutaiqiang/LLM_KD_AKL.svg?style=social&label=Star)](https://github.com/wutaiqiang/LLM_KD_AKL)
+ **LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation** (Dec 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.15881)
  [![Code](https://img.shields.io/github/stars/shufangxun/LLaVA-MoD.svg?style=social&label=Star)](https://github.com/shufangxun/LLaVA-MoD)
+ **Hybrid Data-Free Knowledge Distillation** (Dec 2024, AAAI'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.13525)
  [![Code](https://img.shields.io/github/stars/tangjialiang97/HiDFD.svg?style=social&label=Star)](https://github.com/tangjialiang97/HiDFD)
+ **AlignFD: Beyond Logits - Aligning Feature Dynamics for Effective KD** (2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/ACL-AC6600.svg)](https://aclanthology.org/2025.acl-long.1125/)

+ **Pre-training Distillation for Large Language Models: A Design Space Exploration** (Oct 2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16215)

### Task-specific and Foundations

+ **MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning** (Sep 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.05653)
  [![Code](https://img.shields.io/github/stars/TIGER-AI-Lab/MAmmoTH.svg?style=social&label=Star)](https://github.com/TIGER-AI-Lab/MAmmoTH)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://tiger-ai-lab.github.io/MAmmoTH/)
+ **Personalised Distillation: Empowering Open-Sourced LLMs with Adaptive Learning for Code Generation** (Aug 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.18628)
  [![Code](https://img.shields.io/github/stars/SalesforceAIResearch/PersDistill.svg?style=social&label=Star)](https://github.com/SalesforceAIResearch/PersDistill)
+ **VanillaKD: Revisit the Power of Vanilla Knowledge Distillation from Small Scale to Large Scale** (May 2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.15781)
  [![Code](https://img.shields.io/github/stars/Hao840/vanillaKD.svg?style=social&label=Star)](https://github.com/Hao840/vanillaKD)
+ **Self-Knowledge Guided Retrieval Augmentation for Large Language Models** (Oct 2023, EMNLP'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05002)

+ **DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation** (2024, ISSTA'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.10106)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://distillseq.github.io/page/)
+ **WizardCoder: Empowering Code Large Language Models with Evol-Instruct** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08568)

+ **Performance-Guided LLM Knowledge Distillation for Efficient Text Classification at Scale** (Sep 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.05045)

+ **Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation** (Jul 2025, arXiv'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.00054)

+ **Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs** (2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.16870)

---

## 🔀 Low-Rank Factorization

### Training-Time Low-Rank (PEFT)

+ **LoRA: Low-Rank Adaptation of Large Language Models** (2022, ICLR'22)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=nZeVKeeFYf9)
  [![Code](https://img.shields.io/github/stars/microsoft/LoRA.svg?style=social&label=Star)](https://github.com/microsoft/LoRA)

+ **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** (Mar 2023, ICLR'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.10512)
  [![Code](https://img.shields.io/github/stars/QingruZhang/AdaLoRA.svg?style=social&label=Star)](https://github.com/QingruZhang/AdaLoRA)
+ **QLoRA: Efficient Finetuning of Quantized LLMs** (2023, NeurIPS'23)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=OUIFPHEgJU)
  [![Code](https://img.shields.io/github/stars/artidoro/qlora.svg?style=social&label=Star)](https://github.com/artidoro/qlora)

+ **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA** (Dec 2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.03732)

+ **ReLoRA: Train High-Rank Networks via Low-Rank Updates** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.05695)
  [![Code](https://img.shields.io/github/stars/guitaricet/relora.svg?style=social&label=Star)](https://github.com/guitaricet/relora)
+ **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.12307)
  [![Code](https://img.shields.io/github/stars/dvlab-research/LongLoRA.svg?style=social&label=Star)](https://github.com/dvlab-research/LongLoRA)
+ **Bayesian Low-rank Adaptation for Large Language Models** (Aug 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.13111)
  [![Code](https://img.shields.io/github/stars/adamxyang/laplace-lora.svg?style=social&label=Star)](https://github.com/adamxyang/laplace-lora)
+ **LoRA+: Efficient Low Rank Adaptation of Large Models** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://proceedings.mlr.press/v235/hayou24a.html)
  [![Code](https://img.shields.io/github/stars/nikhilgsh/loraplus.svg?style=social&label=Star)](https://github.com/nikhilgsh/loraplus)
+ **DoRA: Weight-Decomposed Low-Rank Adaptation** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.09353)
  [![Code](https://img.shields.io/github/stars/NVlabs/DoRA.svg?style=social&label=Star)](https://github.com/NVlabs/DoRA)
+ **AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning** (2024, NAACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.09113)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://anonymous.4open.science/r/AutoLoRA)
+ **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02948)
  [![Code](https://img.shields.io/github/stars/GraphPKU/PiSSA.svg?style=social&label=Star)](https://github.com/GraphPKU/PiSSA)
+ **OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models** (Jun 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01775)

+ **Delta-LoRA: Fine-Tuning High-Rank Parameters with the Delta of Low-Rank Matrices** (Sep 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.02411)

+ **KronA: Parameter Efficient Tuning with Kronecker Adapter** (2024, CVPR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10650)

+ **dEBORA: Efficient Bilevel Optimization-based Low-Rank Adaptation** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=5M0ic2RxQZ)

+ **Efficient Learning With Sine-Activated Low-rank Matrices** (2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.19243)

+ **LoRA-Pro: Are Low-Rank Adapters Properly Optimized?** (2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.18242)
  [![Code](https://img.shields.io/github/stars/mrflogs/LoRA-Pro.svg?style=social&label=Star)](https://github.com/mrflogs/LoRA-Pro)
+ **Low-Rank Interconnected Adaptation across Layers** (2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.09946)
  [![Code](https://img.shields.io/github/stars/yibozhong/lily.svg?style=social&label=Star)](https://github.com/yibozhong/lily)
+ **DenseLoRA: Dense Low-Rank Adaptation of Large Language Models** (Jan 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23808)
  [![Code](https://img.shields.io/github/stars/mulin-ahu/DenseLoRA.svg?style=social&label=Star)](https://github.com/mulin-ahu/DenseLoRA)
### Post-Training Low-Rank

+ **LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models** (2024, ICLR'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=LzPWWPAdY4)
  [![Code](https://img.shields.io/github/stars/yxli2123/LoftQ.svg?style=social&label=Star)](https://github.com/yxli2123/LoftQ)
+ **Compressing Large Language Models using Low Rank and Low Precision Decomposition** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.18886)
  [![Code](https://img.shields.io/github/stars/pilancilab/caldera.svg?style=social&label=Star)](https://github.com/pilancilab/caldera)
+ **QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning** (Oct 2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10462)

+ **Low-Rank Compression of Language Models Via Differentiable Rank Selection** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=960Ny6IjEr)

+ **SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression** (2025, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.07378)
  [![Code](https://img.shields.io/github/stars/AIoT-MLSys-Lab/SVD-LLM.svg?style=social&label=Star)](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
### Architectural Low-Rank and Linear Attention

+ **ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations** (2020, ICLR'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1909.11942)
  [![Code](https://img.shields.io/github/stars/google-research/ALBERT.svg?style=social&label=Star)](https://github.com/google-research/ALBERT)
+ **Linformer: Self-Attention with Linear Complexity** (2020, ICML'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.04768)

+ **Rethinking Attention with Performers** (2020, NeurIPS'20)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2009.14794)
  [![Code](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star)](https://github.com/google-research/google-research/tree/master/performer)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://research.google/blog/rethinking-attention-with-performers/)
+ **Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention** (2021, AAAI'21)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.03902)
  [![Code](https://img.shields.io/github/stars/mlpen/Nystromformer.svg?style=social&label=Star)](https://github.com/mlpen/Nystromformer)
+ **Monarch: Expressive Structured Matrices for Efficient and Accurate Training** (2022, ICML'22)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.00595)
  [![Code](https://img.shields.io/github/stars/HazyResearch/monarch.svg?style=social&label=Star)](https://github.com/HazyResearch/monarch)
+ **Retentive Network: A Successor to Transformer for Large Language Models** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/OpenReview-228B22.svg)](https://openreview.net/forum?id=UU9Icwbhin)

+ **Maestro: Uncovering Low-Rank Structures via Trainable Decomposition** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.14929)
  [![Code](https://img.shields.io/github/stars/SamuelHorvath/Maestro-LoD.svg?style=social&label=Star)](https://github.com/SamuelHorvath/Maestro-LoD)
+ **Weight decay induces low-rank attention layers** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.23819)

+ **Breaking the Low-Rank Dilemma of Linear Attention** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.07635)
  [![Code](https://img.shields.io/github/stars/qhfan/RALA.svg?style=social&label=Star)](https://github.com/qhfan/RALA)
+ **Multi-matrix Factorization Attention** (2024, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.19255)

---

## 🔗 Hybrid Compression

### Quantization + Sparsity

+ **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** (Jun 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.03078)
  [![Code](https://img.shields.io/github/stars/Vahe1994/SpQR.svg?style=social&label=Star)](https://github.com/Vahe1994/SpQR)
+ **SqueezeLLM: Dense-and-Sparse Quantization** (Jun 2023, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07629)
  [![Code](https://img.shields.io/github/stars/SqueezeAILab/SqueezeLLM.svg?style=social&label=Star)](https://github.com/SqueezeAILab/SqueezeLLM)
+ **Compressing Large Language Models by Joint Sparsification and Quantization** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/ICML-0A5F8C.svg)](https://openreview.net/forum?id=sCGRhnuMUJ&referrer=%5Bthe%20profile%20of%20Xianglong%20Liu%5D(%2Fprofile%3Fid%3D~Xianglong_Liu3))
  [![Code](https://img.shields.io/github/stars/uanu2002/JSQ.svg?style=social&label=Star)](https://github.com/uanu2002/JSQ)
+ **KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction** (2025, CVPR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23416)
  [![Code](https://img.shields.io/github/stars/snu-mllab/KVzip.svg?style=social&label=Star)](https://github.com/snu-mllab/KVzip)
### Quantization + Low-Rank

+ **LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning** (Nov 2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.12023)
  [![Code](https://img.shields.io/github/stars/HanGuo97/lq-lora.svg?style=social&label=Star)](https://github.com/HanGuo97/lq-lora)
+ **QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models** (2023, ICLR'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.14717)
  [![Code](https://img.shields.io/github/stars/yuhuixu1993/qa-lora.svg?style=social&label=Star)](https://github.com/yuhuixu1993/qa-lora)
+ **LQER: Low-Rank Quantization Error Reconstruction for LLMs** (2024, ICML'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.02446)
  [![Code](https://img.shields.io/github/stars/ChengZhang-98/lqer.svg?style=social&label=Star)](https://github.com/ChengZhang-98/lqer)
+ **DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models** (Sep 2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.09223)

+ **Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition** (Jun 2025, ACL'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.02077)

+ **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-bit Diffusion Models** (Nov 2024, ICLR'25)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.05007)
  [![Code](https://img.shields.io/github/stars/nunchaku-tech/nunchaku.svg?style=social&label=Star)](https://github.com/nunchaku-tech/nunchaku)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://hanlab.mit.edu/projects/svdquant)
### Pruning + Low-Rank

+ **LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning** (2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18403)
  [![Code](https://img.shields.io/github/stars/aim-uofa/LoRAPrune.svg?style=social&label=Star)](https://github.com/aim-uofa/LoRAPrune)
+ **MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router** (2024, EMNLP'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.12013)

+ **SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02214)
  [![Code](https://img.shields.io/github/stars/andyjm3/SLTrain.svg?style=social&label=Star)](https://github.com/andyjm3/SLTrain)
### Quantization + Distillation

+ **LLM-QAT: Data-Free Quantization Aware Training for Large Language Models** (2023, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17888)

+ **BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation** (Aug 2024, ACL'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10631)
  [![Code](https://img.shields.io/github/stars/DD-DuDa/BitDistiller.svg?style=social&label=Star)](https://github.com/DD-DuDa/BitDistiller)
+ **Optimizing Quantized Diffusion Models via Distillation with Cross-Timestep Error Correction** (2025, AAAI'25)  
  [![Paper](https://img.shields.io/badge/AAAI-FF6F00.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/34039)

### Distillation + Pruning

+ **EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning** (Oct 2022, ACL'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.07795)
  [![Code](https://img.shields.io/github/stars/swaggy-TN/EfficientVLM.svg?style=social&label=Star)](https://github.com/swaggy-TN/EfficientVLM)
+ **EPSD: Early Pruning with Self-Distillation for Efficient Model Compression** (2024, AAAI'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.00084)

+ **IEPD-LMM: Large Multimodal Model Compression via Iterative Efficient Pruning and Distillation** (2024, WWW'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.05795)

+ **Compact Language Models via Pruning and Knowledge Distillation** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.14679)
  [![Code](https://img.shields.io/github/stars/NVlabs/Minitron.svg?style=social&label=Star)](https://github.com/NVlabs/Minitron)
  [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
### Distillation + Low-Rank

+ **OPDF: Over-parameterized Distillation via Tensor Decomposition** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.06448)
  [![Code](https://img.shields.io/github/stars/intell-sci-comput/OPDF.svg?style=social&label=Star)](https://github.com/intell-sci-comput/OPDF)

---


## ⚙️ Compiler Optimizations

#### Front-end & IR Layer

+ **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (2018, OSDI'18)  
  [![Paper](https://img.shields.io/badge/OSDI-228B22.svg)](https://arxiv.org/abs/1802.04799)
  [![Code](https://img.shields.io/github/stars/apache/tvm.svg?style=social&label=Star)](https://github.com/apache/tvm)

+ **Glow: Graph Lowering Compiler Techniques for Neural Networks** (2019, arXiv'19)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1805.00907)

+ **Relay: A High-Level Compiler for Deep Learning** (2019, arXiv'19)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1904.08368)
  [![Code](https://img.shields.io/github/stars/apache/tvm.svg?style=social&label=Star)](https://github.com/apache/tvm)

+ **MLIR: A Compiler Infrastructure for the End of Moore's Law** (2020, PLDI'20)  
  [![Paper](https://img.shields.io/badge/PLDI-006400.svg)](https://arxiv.org/abs/2002.11054)

#### Middle-end Layer

+ **Learning to Optimize Tensor Programs** (2018, NeurIPS'18)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://arxiv.org/abs/1805.08166)
  [![Code](https://img.shields.io/github/stars/apache/tvm.svg?style=social&label=Star)](https://github.com/apache/tvm)

+ **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** (2019, MAPL'19)  
  [![Paper](https://img.shields.io/badge/MAPL-0A5F8C.svg)](https://doi.org/10.1145/3315508.3329973)
  [![Code](https://img.shields.io/github/stars/openai/triton.svg?style=social&label=Star)](https://github.com/openai/triton)

+ **Ansor: Generating High-Performance Tensor Programs for Deep Learning** (2021, MLSys'21)  
  [![Paper](https://img.shields.io/badge/MLSys-0A5F8C.svg)](https://arxiv.org/abs/2006.06762)
  [![Code](https://img.shields.io/github/stars/apache/tvm.svg?style=social&label=Star)](https://github.com/apache/tvm)

+ **MetaSchedule: Learning to Optimize Tensor Programs** (2022, MLSys'22)  
  [![Paper](https://img.shields.io/badge/MLSys-0A5F8C.svg)](https://proceedings.mlsys.org/paper/2022/hash/4e732ced3463d06de0ca9a15b6153677-Abstract.html)
  [![Code](https://img.shields.io/github/stars/apache/tvm.svg?style=social&label=Star)](https://github.com/apache/tvm)

+ **Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs** (2023, ASPLOS'23)  
  [![Paper](https://img.shields.io/badge/ASPLOS-FF6F00.svg)](https://doi.org/10.1145/3575693.3575702)
  [![Code](https://img.shields.io/github/stars/hidet-org/hidet.svg?style=social&label=Star)](https://github.com/hidet-org/hidet)

#### Back-end Layer

+ **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022, NeurIPS'22)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
  [![Code](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social&label=Star)](https://github.com/Dao-AILab/flash-attention)

+ **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (2023, arXiv'23)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.08691)
  [![Code](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social&label=Star)](https://github.com/Dao-AILab/flash-attention)

+ **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** (2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.18079)
  [![Code](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social&label=Star)](https://github.com/SqueezeAILab/KVQuant)

+ **PAGED-KV: Demand-Paging KV Cache for LLM Serving** (2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02069)
  [![Code](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social&label=Star)](https://github.com/vllm-project/vllm)

+ **FlashDecoding++: Faster Large Language Model Inference on GPUs** (2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.01282)

+ **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** (2024, NeurIPS'24)  
  [![Paper](https://img.shields.io/badge/NeurIPS-8A2BE2.svg)](https://openreview.net/forum?id=tVConYid20)
  [![Code](https://img.shields.io/github/stars/Dao-AILab/flash-attention.svg?style=social&label=Star)](https://github.com/Dao-AILab/flash-attention)

+ **PyramidKV: Dynamic KV Cache Compression for Efficient Long Sequence Processing** (2024, arXiv'24)  
  [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02069)

## 🏗️ Inference Frameworks

*TODO: Add inference framework papers*

## 💾 Memory Optimization

+ **Deep Compression: Compressing DNNs with Pruning, Trained Quantization and Huffman Coding** (2016, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1510.00149-b31b1b.svg)](https://arxiv.org/abs/1510.00149)

+ **The State of Sparsity in Deep Neural Networks** (2019, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1902.09574-b31b1b.svg)](https://arxiv.org/abs/1902.09574)  
  [![Code](https://img.shields.io/github/stars/google-research/state_of_sparsity.svg?style=social&label=Star)](https://github.com/google-research/google-research/tree/master/state_of_sparsity)

+ **Mixed Precision Training** (2018, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1710.03740-b31b1b.svg)](https://arxiv.org/abs/1710.03740)

+ **Training Deep Nets with Sublinear Memory Cost (Gradient Checkpointing)** (2016, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1604.06174-b31b1b.svg)](https://arxiv.org/abs/1604.06174)

+ **Dynamic Tensor Rematerialization** (2021, ICLR)  
  [![Paper](https://img.shields.io/badge/arXiv-2006.09616-b31b1b.svg)](https://arxiv.org/abs/2006.09616)  
  [![Code](https://img.shields.io/github/stars/uwsampl/dtr-prototype.svg?style=social&label=Star)](https://github.com/uwsampl/dtr-prototype)

+ **MODEl: Memory Optimizations for Deep Learning** (2023, ICML)  
  [![Paper](https://img.shields.io/badge/ICML-2023-0A5F8C.svg)](https://proceedings.mlr.press/v202/steiner23a.html)  
  [![Code](https://img.shields.io/github/stars/facebookresearch/MODel_opt.svg?style=social&label=Star)](https://github.com/facebookresearch/MODel_opt)

+ **ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models** (2020, IEEE SC)  
  [![Paper](https://img.shields.io/badge/IEEE-SC-0C479D.svg)](https://doi.org/10.1109/SC41405.2020.00024)

+ **ZeRO-Offload: Democratizing Billion-Scale Model Training** (2021, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2101.06840-b31b1b.svg)](https://arxiv.org/abs/2101.06840)

+ **Memory and Bandwidth are All You Need for Fully Sharded Data Parallel (FSDP)** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2504.03655-b31b1b.svg)](https://arxiv.org/abs/2504.03655)

+ **COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2410.19313-b31b1b.svg)](https://arxiv.org/abs/2410.19313)  
  [![Code](https://img.shields.io/github/stars/COAT-Lab/coat.svg?style=social&label=Star)](https://github.com/COAT-Lab/coat)

+ **Reducing Transformer Key-Value Cache Size with Cross-Layer Attention** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2405.12981-b31b1b.svg)](https://arxiv.org/abs/2405.12981)

+ **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** (2024, ICML)  
  [![Paper](https://im.shields.io/badge/arXiv-2402.02750-b31b1b.svg)](https://arxiv.org/abs/2402.02750)  
  [![Code](https://img.shields.io/github/stars/jy-yuan/KIVI.svg?style=social&label=Star)](https://github.com/jy-yuan/KIVI)

+ **Ring Attention with Blockwise Transformers for Near-Infinite Context** (2023, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2310.01889-b31b1b.svg)](https://arxiv.org/abs/2310.01889)  
  [![Code](https://img.shields.io/github/stars/lhao499/llm_large_context.svg?style=social&label=Star)](https://github.com/lhao499/llm_large_context)
  
+ **KV Cache Compression, But What Must We Give in Return? (Comprehensive Benchmark)** (2024, EMNLP Findings)  
  [![Paper](https://img.shields.io/badge/ACL-EMNLP_Findings-AC6600.svg)](https://aclanthology.org/2024.findings-emnlp.266/)

+ **PagedAttention: Efficient Memory Management for Large Language Model Serving with PagedAttention** (2023, arXiv)
  [![Paper](https://img.shields.io/badge/arXiv-2312.07953-b31b1b.svg)](https://arxiv.org/abs/2309.06180)

+ **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** (2024, NeurIPS)  
  [![Paper](https://img.shields.io/badge/NeurIPS-2024-8A2BE2.svg)](https://arxiv.org/abs/2401.18079)  
  [![Code](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social&label=Star)](https://github.com/SqueezeAILab/KVQuant)

## 🔧 Hardware Support

+ **Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2410.04466-8A2BE2.svg)](https://arxiv.org/abs/2410.04466)  
  [![Code](https://img.shields.io/github/stars/Kimho666/LLM_Hardware_Survey.svg?style=social&label=Star)](https://github.com/Kimho666/LLM_Hardware_Survey)

+ **Hardware Acceleration of LLMs: A Comprehensive Survey and Comparison** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2409.03384-b31b1b.svg)](https://arxiv.org/abs/2409.03384)

+ **Understanding the Performance and Power of LLM Inferencing on Edge Accelerators** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2506.09554-b31b1b.svg)](https://arxiv.org/abs/2506.09554)

+ **Fast On-device LLM Inference with NPUs** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2407.05858-b31b1b.svg)](https://arxiv.org/abs/2407.05858)

+ **CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs** (2018, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1801.06601-b31b1b.svg)](https://arxiv.org/abs/1801.06601)

+ **MLPerf Tiny Benchmark** (2021, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2106.07597-b31b1b.svg)](https://arxiv.org/abs/2106.07597)
  
+ **LLM-Inference-Bench: Inference Benchmarking of LLMs on AI Accelerators** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2411.00136-b31b1b.svg)](https://arxiv.org/abs/2411.00136) 
  [![Code](https://img.shields.io/github/stars/argonne-lcf/LLM-Inference-Bench.svg?style=social&label=Star)](https://github.com/argonne-lcf/LLM-Inference-Bench)

+ **Evaluating Multi-Instance DNN Inferencing on Multiple Accelerators of an Edge Device** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2503.09546-b31b1b.svg)](https://arxiv.org/abs/2503.09546)  

+ **Dissecting the Graphcore IPU Architecture via Microbenchmarking** (2019, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-1912.03413-b31b1b.svg)](https://arxiv.org/abs/1912.03413)  

+ **Eyeriss: An Energy-Efficient Reconfigurable Accelerator for CNNs** (2017, IEEE JSSC)  
  [![Paper](https://img.shields.io/badge/IEEE-JSSC-0C479D.svg)](https://doi.org/10.1109/JSSC.2016.2616357)
  
+ **SCNN: An Accelerator for Compressed-Sparse CNNs** (2017, ACM SIGARCH)  
  [![Paper](https://img.shields.io/badge/ACM-Digital_Library-2E8B57.svg)](https://doi.org/10.1145/3140659.3080254)

+ **EdgeLLM: A Highly Efficient CPU–FPGA Heterogeneous Edge Accelerator for LLMs** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2407.21325-b31b1b.svg)](https://arxiv.org/abs/2407.21325)

+ **HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs via HLS** (2024, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2405.00738-b31b1b.svg)](https://arxiv.org/abs/2405.00738)

+ **LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and HW Co-design** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2502.15260-b31b1b.svg)](https://arxiv.org/abs/2502.15260)

+ **TerEffic: Highly Efficient Ternary LLM Inference on FPGA** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2502.16473-b31b1b.svg)](https://arxiv.org/abs/2502.16473)
  
+ **PIM Is All You Need: A CXL-Enabled GPU-Free System for LLM Inference** (2025, ASPLOS’25)  
  [![Paper](https://img.shields.io/badge/ACM-Digital_Library-2E8B57.svg)](http://dx.doi.org/10.1145/3676641.3716267)

+ **PIM-LLM: A High-Throughput Hybrid PIM Architecture for 1-bit LLMs** (2025, arXiv)  
  [![Paper](https://img.shields.io/badge/arXiv-2504.01994-b31b1b.svg)](https://arxiv.org/abs/2504.01994)

+ **Benchmarking Energy & Latency in TinyML** (2025, IJCNN’25)  
  [![Paper](https://img.shields.io/badge/arXiv-2505.15622-b31b1b.svg)](https://arxiv.org/abs/2505.15622)

+ **MicroFlow: An Efficient Rust-Based Inference Engine for TinyML** (2024, arXiv / Internet of Things)  
  [![Paper](https://img.shields.io/badge/arXiv-2409.19432-8A2BE2.svg)](https://arxiv.org/abs/2409.19432)  
  [![Code](https://img.shields.io/github/stars/matteocarnelos/microflow-rs.svg?style=social&label=Star)](https://github.com/matteocarnelos/microflow-rs)

+ **llama.cpp: Port of LLaMA in C/C++** (2023, GitHub)  
[![Code](https://img.shields.io/github/stars/ggml-org/llama.cpp.svg?style=social&label=Star)](https://github.com/ggml-org/llama.cpp)


## ☁️ Edge-Cloud Collaboration

*TODO: Add edge-cloud collaboration papers*

---

## 📝 TODO List


### 🚀 System Optimization Section
- [x] **System Optimization Main Section** (Structure added)
- [x] **⚙️ Compiler Optimizations** (Papers added - 3 layers: Front-end/IR, Middle-end, Back-end)
- [ ] **🏗️ Inference Frameworks** (Papers to be added)
- [ ] **💾 Memory Optimization** (Papers to be added)
- [ ] **🔧 Hardware Support** (Papers to be added)
- [ ] **☁️ Edge–Cloud Collaboration** (Papers to be added)

---

> **Note**: This list is continuously updated. Contributions are welcome! Please feel free to open an issue or pull request to add new papers. Code and project links are provided where publicly available.
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LumosJiang/Awesome-On-Device-LLMs&type=Date)](https://star-history.com/#LumosJiang/Awesome-On-Device-LLMs&Date)

---
