# Introduction

Pytorch implementation for Learning Beyond Domains: Misleading Prompts and Pseudo-Label Contrast for Text Domain Generalization (AAAI 2026)

The code includes our method as well as all the comparative methods.


# Requirements

- RoBERTa-base: https://huggingface.co/FacebookAI/roberta-base
- Pytorch: 2.6.0
- transformers: 4.57.3

# Framework

This is the model architecture of GenPromptCL. There are four parts in our method: 
1. **Template construction and backbone**: This module tokenizes the input texts and concatenates tokens with domain-misleading and classification prompts.
2. **CLS Module**: This module aligns text features with corresponding labels to improve the model’s discriminative power.
3. **DMPL Module**: This module randomly shuffles the domain labels in each epoch, thereby misleading the model and making it difficult for it to correctly classify the domain of the text. This encourages the model to focus more on domain-invariant features in the text, ensuring better generalization on downstream tasks.
4. **PCL Module**: This module assigns pseudo labels to each piece of data, and uses these pseudo labels to implement contrastive learning, thereby enhancing the discriminative power of the model within and between classes.


# Experimental results


# Running GenPromptCL

Just execute the following code in the terminal:

```
python run/run_model.py --target_domain book --scl --md_adv --seed 9 --cuda 0 --cov
```



