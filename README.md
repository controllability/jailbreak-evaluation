<p align="center">
  <img src="https://raw.githubusercontent.com/controllability/jailbreak-evaluation/main/logo.png">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
![PyPI](https://img.shields.io/pypi/v/jailbreak-evaluation.svg)
![GitHub stars](https://img.shields.io/github/stars/controllability/jailbreak-evaluation.svg)
![GitHub forks](https://img.shields.io/github/forks/controllability/jailbreak-evaluation.svg)

The jailbreak-evaluation is an easy-to-use Python package for language model jailbreak evaluation.
The jailbreak-evaluation is designed for comprehensive and accurate evaluation of language model jailbreak attempts.
Currently, jailbreak-evaluation support evaluating a language model jailbreak attempt on multiple metrics: **Safeguard Violation** and **Relative Truthfulness**.

This is the official package repository for "[Rethinking How to Evaluate Language Model Jailbreak](https://arxiv.org/abs/2404.06407)", by Hongyu Cai, Arjun Arunasalam, Leo Y. Lin, Antonio Bianchi, and Z. Berkay Celik.

## Installation

**Note on PyTorch and FastChat Installation**:
The jailbreak-evaluation depends on [PyTorch](https://pytorch.org/get-started/locally/) and [FastChat](https://github.com/lm-sys/FastChat?tab=readme-ov-file#install).
To streamline the installation, jailbreak-evaluation does **NOT** install these libraries for you.

It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed, as jailbreak-evaluation is updated frequently:

```bash
python -m pip install --upgrade pip

# Please install the PyTorch: https://pytorch.org/get-started/locally/
# Please install the FastChat: https://github.com/lm-sys/FastChat?tab=readme-ov-file#install

# Install the jailbreak-evaluation
python -m pip install -U jailbreak-evaluation

# Initialize the Natural Language Toolkit (NLTK)
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start
Evaluate a Language Model Jailbreak Attempt Using jailbreak-evaluation.
You should have an OpenAI API key to use the jailbreak-evaluation, which can be obtained from [OpenAI Platform](https://platform.openai.com/api-keys).

```python
from jailbreak_evaluation import MultifacetedEvaluation

# Initialize the jailbreak evaluation
openai_api_key = "your openai_api_key"
multifaceted_evaluation = MultifacetedEvaluation(openai_api_key)

# Construct a jailbreak attempt example
intent = "intent"
response = "response"

# Evaluate the above jailbreak attempt example
result = multifaceted_evaluation.evaluate(intent, response)

print("The evaluation result for all levels:")
print(result)

print("The evaluation result for combination level, which is our best accurate result:")
print(result.combination_level)
```

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/controllability/jailbreak-evaluation/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

## Cite

If you use jailbreak-evaluation in a scientific publication, we would appreciate citations to the following paper:
```
@article{cai2024rethinking,
  title={Rethinking How to Evaluate Language Model Jailbreak}, 
  author={Hongyu Cai and Arjun Arunasalam and Leo Y. Lin and Antonio Bianchi and Z. Berkay Celik},
  year={2024},
  journal={arXiv}
}
```

## The Team

The jailbreak-evaluation is developed and maintained by [PurSec Lab](https://pursec.cs.purdue.edu/).

## License

The jailbreak-evaluation uses Apache License 2.0.
