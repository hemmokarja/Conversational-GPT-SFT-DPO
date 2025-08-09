# GPT2 Fine-tuning for Conversational AI

**⚠️ Work in Progress**

A from-scratch PyTorch implementation of GPT-2 fine-tuning for conversational AI, featuring custom pipelines for Supervised Fine-Tuning (SFT) and planned Direct Preference Optimization (DPO). SFT is used for multi-turn dialogue alignment, teaching context awareness and turn-taking, while DPO enables reward-free preference optimization for higher-quality responses. Built without relying on high-level training APIs from frameworks like Hugging Face Transformers to maximize low-level control and hands-on understanding of modern fine-tuning techniques.

## 🚀 Key Features

### ✅ Completed

- **GPT2 Architecture**: Complete PyTorch implementation of the GPT2 transformer architecture from scratch
- **Weight Loading**: Integration with Hugging Face pre-trained GPT2 weights
- **Training Pipeline**: Comprehensive custom training and evaluation suite
- **Supervised Fine-Tuning (SFT)**: SFT implementation using OASST1 dataset to teach reciprocal conversation patterns
- **LoRA (Low-Rank Adaptation)**: From-scratch PyTorch implementation for parameter-efficient fine-tuning, reducing memory usage and training time

### 🚧 In Development

- **Direct Preference Optimization (DPO)**: Implementation to improve response quality by learning to distinguish between informative and uninformative responses
- **Dataset Expansion**: Integration of OASST2 and additional conversational datasets

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

## 📚 Methodology

### Supervised Fine-Tuning (SFT)

The SFT phase teaches the model to engage in reciprocal conversations rather than simply completing documents. This involves:

- Converting instruction-following datasets to conversational format
- Teaching turn-taking and context awareness
- Maintaining coherence across multi-turn dialogues

### Direct Preference Optimization (DPO) [Planned]

The DPO phase will improve response quality by:

- Learning from preference data
- Distinguishing high-quality from low-quality responses
- Optimizing for helpfulness and informativeness without explicit reward modeling

DPO is a relatively new method, replacing the more traditional, albeit more unstable and finicky, two-stage RLHF.

## 📝 License

This project is licensed under the MIT License.
