# **ML Project - RLHF**
JELASSI Meriem

---

# Table of Contents

**Introduction**

**Transformers**
   - Encoder-Decoder Architecture
   - Attention Mechanism
   - Multi-Head Attention

**Causal Transformers and Self-Supervised Training**
   - Autoregressive Mechanism
   - Self-Supervised Learning

**Reinforcement Learning from Human Feedback (RLHF)**
   - Supervised Fine-Tuning
   - Reward Model Training
   - Benefits and Limitations

**Conclusion**

**References**

---

# Introduction

The emergence of large language models (LLMs) has transformed the landscape of artificial intelligence, enabling remarkable advancements in natural language processing. Models such as GPT, BERT, and Llama 2 have demonstrated impressive capabilities across a variety of tasks, ranging from text generation to machine translation.

However, despite their power, these models exhibit significant limitations. They may produce responses misaligned with human expectations, generate biased or toxic content, and lack precision in specific contexts. These issues present both ethical and practical challenges, particularly in deploying such models in sensitive environments.

To address these challenges, a method known as Reinforcement Learning from Human Feedback (RLHF) has emerged. This approach combines supervised learning techniques with reinforcement learning to align models with human preferences. By incorporating human feedback into the training process, RLHF enhances both the relevance and safety of generated responses.

This report aims to:

- Present the foundational principles of Transformers, which underpin modern LLMs.
- Provide a detailed description of the RLHF approach, with a particular focus on Proximal Policy Optimization (PPO).
- Offer practical recommendations and a methodology to integrate RLHF.

Drawing on scientific articles, concrete case studies, and in-depth technical concepts, this report delivers a clear and actionable synthesis to fully harness the potential of LLMs while addressing ethical and operational requirements.

# Transformers

Transformers have become the foundation of modern language models due to their ability to process complex sequences while capturing contextual relationships between tokens. Their architecture is built around two main components: the encoder and the decoder.

The encoder processes the input sequence to extract a rich contextual representation. Each encoder block applies a self-attention mechanism, enabling each token to interact with other tokens in the sequence. This helps capture relationships between distant words in a sentence. After identifying these relationships, fully connected layers transform the representations to enhance their expressiveness.

The decoder, on the other hand, generates the output sequence by leveraging both the representations produced by the encoder and the tokens generated so far. Through cross-attention, the decoder aligns each generated token with the relevant parts of the input. To prevent the decoder from "seeing" future tokens, it also uses a masked attention mechanism.

These two modules work together seamlessly. For instance, in machine translation, the encoder transforms a source sentence into a contextual representation, while the decoder uses this representation to generate the translated sentence.

## Attention Mechanism

The self-attention mechanism is the basis of Transformers. It allows the model to assign relative importance to each token in an input sequence by analyzing the relationships between the tokens. This mechanism relies on three types of vectors derived from the input tokens: query (\(Q\)), key (\(K\)), and value (\(V\)) vectors. These vectors are computed by multiplying the embeddings of the tokens with learned weight matrices, as shown below:

\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]

Here, \(X\) represents the token embeddings, and \(W^Q\), \(W^K\), \(W^V\) are weight matrices learned specifically for each type of vector.

Once these vectors are computed, the model calculates the similarity between each query (\(Q\)) and each key (\(K\)) to determine their relationship. These similarities, referred to as attention scores, are computed by taking the dot product of \(Q\) and \(K\), normalized by the square root of the dimension of the key vectors (\(d_k\)) to avoid extreme values. The formula for attention scores is:

\[
\text{Attention Score} = \frac{QK^T}{\sqrt{d_k}}
\]

To make these scores comparable and usable, they are converted into probabilities using the softmax function. This normalization ensures that the scores for each token sum to 1, making them interpretable as relative weights. The softmax function is defined as:

\[
\text{Softmax}(\text{Attention Score}) = \frac{\exp(\text{Attention Score})}{\sum \exp(\text{Attention Score})}
\]

Finally, the output of the attention mechanism is computed by combining these probabilities as weights with the value vectors (\(V\)). This produces a rich contextual representation where each token is influenced by its relationships with other tokens in the sequence. The output is expressed by the following formula:

\[
\text{Output} = \text{Softmax}(\text{Attention Score}) \cdot V
\]

This process is repeated for each token, enabling the model to capture complex relationships, whether they are local or distant, within the input sequences.

## Multi-Head Attention Mechanism

To better capture the diversity of relationships in the data, Transformers use a multi-head attention mechanism. Instead of extracting a single contextual relationship for each token, multiple independent attention heads are computed simultaneously. Each head focuses on a different aspect of the interactions between tokens, allowing the model to capture multiple perspectives in parallel.

Each head applies the self-attention mechanism, with its own set of weight matrices for \(Q\), \(K\), and \(V\). The output of each head is then concatenated and projected into a common space using a final projection matrix (\(W^O\)). This operation is described by the following formula:

\[
\text{MultiHead}(Q, K, V, \text{Mask}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]

where each head (\(\text{head}_i\)) is defined as:

\[
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i, \text{Mask})
\]

The multi-head attention mechanism enriches the model by enabling it to explore multiple aspects of contextual relationships. For example, in a sentence, some heads might focus on syntactic relationships, while others explore semantic ones.

## Causal Transformers and Self-Supervised Training

Causal Transformers, such as GPT and LLaMA, represent a class of autoregressive models designed for sequential text generation. These models are built on the Transformer architecture and are trained using self-supervised learning methodologies. This training model allows them to predict tokens based on prior context, making them particularly effective for generative tasks such as language modeling, text completion, and conversational AI.

### Autoregressive Mechanism of Causal Transformers

Causal Transformers operate autoregressively, meaning they predict each token in a sequence based solely on the preceding tokens. This unidirectional approach respects the temporal or sequential order of the data, ensuring that the model only "sees" past information when generating text. The autoregressive process is mathematically formulated as:

\[
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^n P(x_t | x_1, x_2, \ldots, x_{t-1})
\]

Here, \(P(x_t | x_1, x_2, \ldots, x_{t-1})\) represents the probability of the \(t\)-th token given its preceding context. During training, the model learns to maximize the likelihood of the observed sequences under this factorization.

The self-attention mechanism is central to this autoregressive process. To enforce causality, a triangular mask is applied to the attention scores, ensuring that tokens can only attend to preceding tokens and not future ones. This causal masking guarantees the unidirectional flow of information, a requirement for tasks like text generation.

### Self-Supervised Learning

Causal Transformers are trained using self-supervised learning, a methodology that allows models to create their own supervision signals from unlabeled data. This approach eliminates the need for extensive manual annotation. For autoregressive Transformers like GPT and LLaMA, the primary training objective is causal language modeling (CLM), where the model predicts the next token in a sequence given its context.

The training loss for causal language modeling is defined as:

\[
\mathcal{L} = -\sum_{t=1}^n \log P(x_t | x_1, x_2, \ldots, x_{t-1})
\]

During training, the model adjusts its parameters to minimize this loss, learning to generate coherent and contextually accurate text. This self-supervised approach enables the model to leverage massive amounts of unstructured data, capturing complex language patterns, syntax, and semantics.

# Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) is an innovative method that combines supervised learning and reinforcement learning to align language model outputs with human preferences. This approach has been successfully applied to improve models such as InstructGPT and Claude, enabling them to generate more relevant, safe, and user-aligned responses.

## Supervised Fine-Tuning

The RLHF pipeline begins with a supervised fine-tuning phase. In this step, the pretrained language model is adjusted using human-annotated data. These data, often in the form of question-answer pairs or task-specific examples, guide the model toward baseline behaviors aligned with human expectations. This phase establishes a solid foundation, preparing the model for further refinement through reinforcement learning.

## Reward Model Training

After supervised fine-tuning, a reward model is trained to predict human preferences. This model plays a critical role in the RLHF pipeline by evaluating the quality of the responses generated by the language model. Human annotators compare multiple responses and indicate which one they prefer. These preferences are then used to train the reward model, which assigns a score to each possible response. The training is guided by the following loss function:

\[
\text{Loss}(r_\theta) = -\log(\sigma(r_\theta(x, y_i) - r_\theta(x, y_{1-i})))
\]

Here, \(y_i\) is the preferred response, \(y_{1-i}\) is the rejected response, and \(\sigma\) is the sigmoid function. This step captures human preferences in a quantifiable framework, enabling the model to align its outputs accordingly.

## Benefits and Limitations

RLHF offers several benefits, including its ability to align model outputs with human expectations, reduce biases, and limit harmful content. Studies such as InstructGPT have demonstrated that RLHF significantly improves response quality. For instance, InstructGPT, with only 1.3 billion parameters, outperformed GPT-3 (175 billion parameters) in human evaluations while reducing hallucinations and biases.

However, RLHF is not without challenges. Its reliance on human-annotated data can introduce cultural or subjective biases. Additionally, the computational cost of this method is high, especially for large-scale models. Over-optimization against the reward model can also harm generalization and reduce the diversity of model behaviors.

# Conclusion

Large language models (LLMs) like GPT and LLaMA have transformed artificial intelligence by achieving significant success across various natural language processing tasks. However, their limitations, including biases, misalignment with human values, and ethical concerns, highlight the importance of developing more robust and aligned training methodologies.

By building on foundational architectures like Transformers and incorporating approaches such as Reinforcement Learning from Human Feedback (RLHF), LLMs can produce safer, more reliable, and human-aligned outputs. Although challenges such as computational costs and potential biases persist, advancements in techniques like Proximal Policy Optimization (PPO) and causal Transformers provide promising solutions for addressing these issues. Continued innovation in these methods will play a key role in the responsible development and application of LLMs.

---

# References
   - [GPT-2 Sentiment Analysis Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb)
   - [Reward Modeling Script](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py)
   - [Quickstart Guide](https://huggingface.co/docs/trl/quickstart#minimal-example)
   - [PPO Implementation](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py)
   - [PPO for Summarization](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py)
   - [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master)
   - ["Training Language Models to Follow Instructions with Human Feedback"](https://arxiv.org/pdf/2203.02155)
   - ["Learning to Summarize with Human Feedback"](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)
   - ["Training a Helpful and Harmless Assistant with RLHF"](https://arxiv.org/pdf/2204.05862)
   - ["Proximal Policy Optimization Algorithms (PPO)"](https://arxiv.org/pdf/1707.06347)
   - [RLHF with PPO - Implementation Details](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
   - [Explaining RLHF](https://huggingface.co/blog/rlhf)
   - [Reward Model Training](https://medium.com/towards-generative-ai/reward-model-training-2209d1befb5f)
   - [RLHF Lecture at Stanford](https://docs.google.com/presentation/d/1T6X8ZlwrBek14wGfKljLxikwkTBDdM88r0AZ6NiodU4/edit#slide=id.g2a01be3ce56_1_487)
   - [YouTube: PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
   - [YouTube: RLHF Explained](https://www.youtube.com/watch?v=2MBJOuVq380)
   - [YouTube: Live Coding PPO](https://www.youtube.com/watch?v=hlv79rcHws0)
   - ["A Mathematical Interpretation of Autoregressive GPT and Self-Supervised Learning"](https://www.mdpi.com/2227-7390/11/11/2451?utm_source=chatgpt.com)
   - ["Adapting LLaMA Decoder to Vision Transformer"](https://arxiv.org/abs/2404.06773)
   - [Transformers Tutorial (Data Science Tutorials)](https://github.com/AdilZouitine/data-science-tutorials/blob/master/deep-learning-tse/transformers.ipynb)
