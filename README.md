# GAMBIT - Game-based Adversarial Model Improvement Technique

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ModelMystic/GAMBIT) [![Chatbot on Spaces](https://img.shields.io/badge/%F0%9F%A4%96%20Chatbot-Spaces-blue)](https://huggingface.co/spaces/ModelMystic/GAMBIT)
---
## Overview  
This project explores an adversarial game designed between two AI entities, **Detective LLM** and **Cheater LLM**, with the integration of a **Classifier** to mediate and enhance adversarial interactions. The main objective was to fine-tune each model to become proficient in their respective tasks:
- **Detective LLM**: Detecting adversarial inputs effectively.
- **Cheater LLM**: Generating adversarial inputs to evade detection.

### Key Highlights:
- Implementation of cutting-edge fine-tuning techniques like **RLHF (Reinforcement Learning with Human Feedback)** and **LoRA (Low-Rank Adaptation)**.
- Use of diverse datasets (**HC3** and **QRECC**) tailored for adversarial tasks.
- Dynamic data handling with heap-based clustering and queueing mechanisms to maintain dataset diversity and reduce overfitting.

---

## Problem Statement  
The goal was to design a system where two adversarial LLMs interact under the supervision of a Classifier to improve their respective capabilities. By introducing adversarial prompts and optimizing fine-tuning, we aimed to establish a robust pipeline that balances creativity, coherence, and efficiency.

---
## Folder Structure
```
GAMBIT/
├── __init__.py
├── run.py
└── utils/
    ├── __init__.py
    ├── detection.py
    ├── finetune_Cheater.py
    ├── finetune_Detective.py
    ├── finetune_pipeline.py
    ├── generate_answer.py
    ├── generate_question.py
    ├── response_filter.py
    ├── model_setup.py
    └── misc.py
```
---
## 📌 Running Locally
```sh
# Clone the repository
git clone https://github.com/Cicada-33-01/GAMBIT.git
cd GAMBIT

# Install dependencies
pip install -r requirements.txt

# Run the script
python Run_GAMBIT.py
```
---
## Datasets  
### 1. **HC3 Dataset**  
- Adapted with Cheater LLM-generated responses replacing ChatGPT outputs.  
- Designed to mimic real-world evasion strategies.

### 2. **QRECC Dataset**  
- Integrated to add diversity and evaluate models under varied contextual scenarios.  

---

## Methodology  
1. **Model Selection**  
   - **Detective LLM**: FLAN-T5, optimized for instruction-following tasks.  
   - **Cheater LLM**: Llama 3B, chosen for its lightweight architecture and ability to handle complex prompts.  

2. **Fine-Tuning Techniques**  
   - **Detective LLM**:
     - **Top-p Sampling**: Controlled response diversity.
     - **Temperature Tuning**: Balanced randomness for adversarial interactions.
     - **Repetition Penalty**: Reduced redundant outputs.  
   - **Cheater LLM**: Focused on generating responses designed to bypass detection filters.

3. **Dynamic Dataset Handling**  
   - **Heap-Based Cluster**: Prioritized adversarial responses based on confidence scores.
   - **Dynamic Queue Updates**: Ensured dataset diversity by monitoring and updating datasets dynamically.
   - **Random Removal**: Maintained variety and reduced risks of overfitting.

---

## Results  
- **Cheater LLM** successfully bypassed multiple AI detection platforms, demonstrating its capability to mimic human-like content.  
- **Detective LLM** achieved high accuracy in detecting adversarial prompts by leveraging advanced fine-tuning and data augmentation strategies.  
- **Dynamic Clustering and Queuing** improved computational efficiency and dataset quality.

---

## Challenges and Solutions  
### Challenges  
1. **Determinism during fine-tuning**: Required selecting the optimal framework (e.g., RLHF, LoRA).  
2. **Incorrect data buildup**: Mitigated by using priority queuing mechanisms.

### Solutions  
- Implemented a **custom loss function** based on distribution metrics rather than point-to-point comparisons.  
- Introduced dynamic mechanisms to handle growing datasets without clearing the queue.

---

## Conclusion  
The adversarial game pipeline was successfully established, achieving significant performance improvements in both detection and evasion tasks. The use of innovative methodologies like dynamic clustering, queueing, and fine-tuning highlighted the potential for scalable and robust adversarial LLM systems.

### Future Work  
- Exploring cross-domain adaptability for adversarial scenarios.  
- Scaling the system for larger datasets and real-time applications.

---

## References  
1. Wang, Y., et al. *LLM-GAN: Construct Generative Adversarial Network Through Large Language Models For Explainable Fake News Detection*. arXiv:2409.01787 (2024).  
2. Lee, D. Hee, and Jang, B. *Enhancing Machine-Generated Text Detection: Adversarial Fine-Tuning of Pre-Trained Language Models*. IEEE Access, vol. 12, pp. 65333-65340, 2024.
