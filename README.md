# GPT2 Fine-tuning for Conversational AI

**⚠️ Work in Progress**

A **from-scratch PyTorch implementation of GPT-2 fine-tuning for conversational AI**, featuring **custom pipelines for Supervised Fine-Tuning (SFT) and [planned] Direct Preference Optimization (DPO) with LoRA (Low-Rank Adaptation).** SFT is used for multi-turn dialogue alignment, teaching context awareness and turn-taking, while DPO enables reward-free preference optimization for higher-quality responses. Built without relying on high-level training APIs from frameworks like Hugging Face Transformers to maximize low-level control and hands-on understanding of modern fine-tuning techniques. This project is a personal deep dive to really get under the hood of how these fine-tuning methods work in practice.

## 🚀 Key Features

### ✅ Completed

- **GPT2 Architecture**: Complete PyTorch implementation of the GPT2 transformer architecture from scratch
- **Weight Loading**: Integration with Hugging Face pre-trained GPT2 weights for document completion
- **Training Pipeline**: Custom training and evaluation suite for low-level control
- **Supervised Fine-Tuning (SFT)**: SFT implementation supporting OASST1/OASST2 datasets to teach reciprocal conversation patterns
- **LoRA (Low-Rank Adaptation)**: From-scratch PyTorch implementation for parameter-efficient fine-tuning, reducing memory usage and training time

### 🚧 In Development

- **Direct Preference Optimization (DPO)**: Implementation to improve response quality by learning to distinguish between informative and uninformative responses

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

## 📊 Sample Outputs

The following examples demonstrate the impact of Supervised Fine-Tuning on model responses using the smallest GPT-2 model (124M parameters) and only ~1 hour of training on a MacBook Pro (CPU). While this model has severely limited capacity compared to modern LLMs with parameter counts in the hundreds of billions, the examples illustrate the fundamental improvements SFT provides in conversational alignment, compared to a pre-trained document-completing model.

### "What are the best attractions in Rome, Italy?"

Before:
```
If you're asking how places are worth living in Rome, France, Finland and the US, I'm
still not sure that they really represent you. That probably goes without saying, but
they're the basic elements in many cities - a rising sea level and frequent exposure to
an unwanted summer air all add up to having a place worth living in despite not big
enough real estate.
```

After:
```
As I know Rome, it is not a city and it is not a tourist attraction but it is a city
with some amazing attractions.  Here are some places to stay in Rome, Italy:

Museum is the oldest museum in Rome with its historical attractions and the famous
Pompeii Square.

Currency market is a thriving market of the Roman gold and Roman numismatics.

Grizzoli and Ruggiano are two of the most famous historical museums in Rome and its
heritage. They are the largest museums in Rome and can be enjoyed from anywhere.

Pizza is one of Rome's great and most popular cuisine and many are often held by
tourists as well.

There are many activities you can take advantage of during your stay here with many
other attractions including sports such as Olympic events, concerts, film shoots,
theatre productions and more.
```

### "What does an architect do?"

Before:
```
Get a library of data. Copy through Schapiroz's channel (thus copying over the
accumulated data) to show the same bits/Octangles you see in every piece of the diagram.
To achieve these end results, increase the sample size.
```

After:
```
Architects are engineers and experts who provide solutions to building systems, and
they often have particular knowledge of architecture or engineering techniques.
This can have a huge impact on how you do a project, such as when you're designing new
buildings or installing them in a new neighborhood.

In addition to doing your design, architects may also assist you with building
management and planning, as well as provide knowledge about product development, design
concepts, design patterns, and design strategies.
```

----

Perfect? Nope. Factual? Weeell, that’s a stretch. Improvement? You be the judge :)

## 📝 License

This project is licensed under the MIT License.
