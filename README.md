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

### After SFT

```
========== user ==========
What does a Data Scientist do?

========== assistant ==========
A Data Scientist is a professional who specializes in managing data in the research
industry. They handle data using sophisticated tools and techniques to analyze and
interpret complex data sets, and make decisions about its quality and applicability to
their customer's needs. While a Data Scientist is a critical part of your organization's
data science efforts and might provide significant benefits, it's likely the most
challenging role in the industry, since they often work alone or within a team, and
must master technical skills designed for software programmers.

Learning from seasoned data scientists may help you stay ahead of the curve, and develop
your skills on the cutting edge of computing, analytics, and machine learning, as well
as on innovative new technologies in business.

========== user ==========
What about a Machine Learning Engineer?

========== assistant ==========
A Machine Learning Engineer is responsible for developing advanced analytical,
learning, and machine learning models.

A Machine Learning Engineer strives to create the most appropriate machine learning
models to achieve the highest levels of accuracy and predictive power.

The job is to create algorithms that are able to efficiently extract, process, and
store knowledge from large datasets, leveraging and extending other factors such as
machine learning frameworks, data support systems, human-like models, and AI software.

The most important feature of the job is to make use of the latest advances in machine
learning techniques and algorithms, and to create systems that are capable of solving
a variety of tasks in deep neural networks, sentiment generation, object recognition,
and other complex applications across different domains.
```

With SFT, the model’s output goes from random text continuation to straightforward, instruction-following replies. There's certainly room for improvement though.

### After DPO

```
========== user ==========
What does a Data Scientist do?

========== assistant ==========
A data scientist is a person who has technical expertise or professional experience in
the field of data, such as data science, statistical analysis or database management.
Data science is a broad term that encompasses a wide range of skills and techniques that
can be applied to the analysis or design of data, from data mining to machine learning.

Data scientists include people who understand the foundations of data science, as well
as the scientific method and data analysis. They specialize in identifying patterns or
trends in large volumes of data, extracting values and identifying data sources. They
also work on various aspects of the process, including the design, optimization, and
integration of these principles.

In addition to analytical skills, data scientists also can learn how to apply these
techniques to their own projects to improve efficiency and the effectiveness of data
analysis and statistical analysis. They can also use those insights and tools to create
innovative products and solutions.

Data scientists often work for data analytics companies, such as those offering machine
translation, machine learning, or analytic data sets, and in some cases, work independently.
They can also contribute to projects like data science standards, machine learning
libraries, or related tools like data engineering, data visualization, and data mining.

Generally, these data scientists have strong analytical and technological expertise in
various areas, but often have little formal mathematical background. They also might
have some business-related backgrounds, such as being a consultant, business leader, or
scientist in other fields, but will often specialize in machine learning or statistical
analysis, with the specific emphasis on the areas where they have technical expertise.

Although relatively new in the area of data science, it's growing in importance because
of the opportunities it provides for improving the productivity of businesses and the
world at large. The data scientist, on the job, should demonstrate a broad understanding
of the scientific method and the importance of data science in improving productivity
and innovation.
```

Perfect? Well, that's a stretch. Improvement over the baseline? You be the judge :)

## 📝 License

This project is licensed under the MIT License.
