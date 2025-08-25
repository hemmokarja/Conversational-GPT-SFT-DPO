# GPT2 Fine-tuning for Conversational AI

A **from-scratch PyTorch implementation of GPT-2 fine-tuning for conversational AI**, featuring **custom pipelines for Supervised Fine-Tuning (SFT) and [planned] Direct Preference Optimization (DPO) with LoRA (Low-Rank Adaptation).** SFT is used for multi-turn dialogue alignment, teaching context awareness and turn-taking, while DPO enables reward-free preference optimization for higher-quality responses. Built without relying on high-level training APIs from frameworks like Hugging Face Transformers to maximize low-level control and hands-on understanding of modern fine-tuning techniques. Includes a lightweight CLI chat interface for interacting with fine-tuned models via terminal. 

This project is a personal deep dive to really get under the hood of how these fine-tuning methods work in practice.

Note: **repo still work in progress ⚠️**

## 🚀 Key Features

### ✅ Completed

- **GPT2 Architecture**: Complete PyTorch implementation of the GPT2 transformer architecture from scratch
- **Weight Loading**: Integration with Hugging Face pre-trained GPT2 weights for document completion
- **Training Pipeline**: Custom training and evaluation suite for low-level control
- **Supervised Fine-Tuning (SFT)**: SFT implementation supporting OASST1/OASST2 datasets to teach reciprocal conversation patterns
- **LoRA (Low-Rank Adaptation)**: From-scratch PyTorch implementation for parameter-efficient fine-tuning, reducing memory usage and training time
- **CLI Chat Interface**: Lightweight terminal-based chat interface for interacting with fine-tuned models

### 🚧 In Development

- **Direct Preference Optimization (DPO)**: Implementation to improve response quality by learning to distinguish between informative and uninformative responses

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

## 🚀 Quick Start

### Fine-tuning with SFT
To apply Supervised Fine-Tuning, run:
```bash
uv run sft.py
```
Remember to adjust configs (in that file) to your liking.

### Chatting with a Fine-tuned Model
To chat with a fine-tuned model, run:
```bash
uv run chat.py
```
Remember to configure the correct checkpoint (in that file).

## 📚 Methodology

### Supervised Fine-Tuning (SFT)

The SFT phase teaches the model to engage in reciprocal conversations rather than simply completing documents. This involves:

- Converting instruction-following datasets to conversational format
- Teaching turn-taking and context awareness
- Maintaining coherence across multi-turn dialogues

### Direct Preference Optimization (DPO) [Planned]

The DPO phase will improve response quality by directly aligning model behavior with human preferences. Unlike traditional RLHF, which relies on reward modeling and policy optimization, DPO offers a simpler and more stable approach. This phase focuses on:

- Learning from preference data
- Distinguishing high-quality from low-quality responses
- Optimizing for helpfulness and informativeness without explicit reward modeling

Together, **SFT provides the baseline alignment**, while **DPO sharpens the model’s ability to capture nuanced human preferences**.

## 📊 Sample Outputs  

The following example demonstrates the impact of **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** on model responses using a 355 M parameter GPT-2 model (`gpt2-medium`) and approximately five hours of training on a `g5.4xlarge` EC2 instance on AWS. While this model has severely limited capacity compared to modern LLMs with parameter counts in the hundreds of billions, the examples illustrate the fundamental improvements both methods provide in conversational alignment.

### Before SFT

```
========== user ==========
What does a Data Scientist do?

========== assistant ==========
How do they actually work? How are they paid? What is a data scientist doing like a foot
soldier? 9. How do you become a weather editor? (Write a research paper by hand, in a
journal, or scan the Internet) 10. What women do they want to marry? In 9 of the 10 ways
we usually use figures, the definition is different (gendered roles, getting older, etc)
but you're definitely feeling the female trappings too.

========== user ==========
What about a Machine Learning Engineer?

========== assistant ==========
When founders leave, how often do you see employees who were engineer? You may not get
any sense of it until previous phase, yet now you've just been told "you're ears will
too." What if you take to the Internet and try to teach about the joys of machine
learning.
```

On its own, a pre-trained GPT trained for document completion cannot hold a back-and-forth conversation or provide direct answers to questions.

### After SFT and DPO

```
========== user ==========
What does a Data Scientist do?

========== assistant ==========
A data scientist plays a crucial role in driving the business and organization in which
you work to achieve more and better outcomes for your users. A data scientist is a data
analyst who uses statistical, machine learning, and analytics abilities to extract
insights from data, identify patterns, and make informed decisions.

They analyze reports, dashboards, and data sources on various stakeholders, including
customers, partners, and competitors, to gain a deeper understanding of their behaviors,
interests, and pain points. This analysis helps them develop targeted and effective
solutions, which can benefit the business or organization in several ways.

Data scientists also contribute in various technical aspects, such as schema design,
data visualization, data optimization, and data modeling. By working collaboratively
with different teams and departments, they can provide valuable insights and tools for
decision-making and growth.

When working with complex data-driven challenges, a data scientist can focus on the
technical aspects and develop effective strategies to understand the data, extract
insights, and communicate with stakeholders effectively.

========== user ==========
What about a Machine Learning Engineer?

========== assistant ==========
A machine learning engineer is someone who is responsible for developing and
maintaining data-driven algorithms, and responsible for understanding and optimizing
the system-wide requirements of machine learning.

As the generalist in machine learning, their primary responsibility is to ensure that
machine learning algorithms are reliable, scalable, and well-behaved, ensuring that
the system remains sensitive to individual preferences while performing well across
different domains.

Another important aspect of a machine learning engineer's role is to ensure that the
algorithm works as expected in real-world applications, without any biases or edge
cases. This involves conducting thorough and unbiased evaluations, and ensuring that
the performance of the machine learning model in terms of accuracy, precision, and
reliability is kept up to the NDA criteria.

While a machine learning engineer typically holds a bachelor's degree in mathematics
or computer science, they can also specialize in areas such as machine learning,
computer vision, or machine learning theory. In general, the key roles of a machine
learning engineer are to develop and refine algorithms, support the development and
implementation of machine learning models, and implement and maintain the
infrastructure associated with the system.
```

Perfect? Well, that's a stretch. Improvement over the baseline? You be the judge :)

## 📝 License

This project is licensed under the MIT License.
