# LLMs-GPT Series

For LLAMA Series, please refer to https://github.com/LiliMeng/Llama
## Useful resources
LLMs course with examples in Google Colab (link)[https://github.com/mlabonne/llm-course]


## Evolution of GPT, GPT-2, GPT-3, GPT-4 and GPT-4o
Here's a summary of the key differences among GPT, GPT-2, GPT-3, GPT-4, and GPT-4O:

### GPT (Generative Pre-trained Transformer):
- **Release**: June 2018
- **Parameters**: 117 million
- **Features**: 
  - First model in the GPT series.
  - Utilizes the transformer architecture.
  - Focused on language modeling with unsupervised learning.
  - Demonstrated that pre-training on a large corpus and fine-tuning on specific tasks can yield state-of-the-art results in NLP.

### GPT-2:
- **Release**: February 2019
- **Parameters**: 1.5 billion
- **Features**: 
  - Significantly larger than GPT-1.
  - Trained on a dataset called WebText, consisting of 8 million web pages.
  - Demonstrated strong zero-shot learning capabilities, meaning it could perform tasks it wasn't explicitly trained for by conditioning on task-specific prompts.
  - Concerns about misuse led to a staged release, with the full model released later.

### GPT-3:
- **Release**: June 2020
- **Parameters**: 175 billion
- **Features**: 
  - Much larger model size, enabling more nuanced and coherent text generation.
  - Trained on diverse datasets, including Common Crawl and WebText.
  - Showcased impressive few-shot, one-shot, and zero-shot learning abilities, further improving performance on a wide range of NLP tasks without needing task-specific training data.
  - Used for various applications, including chatbots, content generation, and code completion.

### GPT-4:
- **Release**: March 2024 (hypothetical, as OpenAI's naming convention continues sequentially)
- **Parameters**: Unspecified but expected to be significantly larger than GPT-3.
- **Features**: 
  - Expected advancements in few-shot and zero-shot learning capabilities.
  - Enhanced ability to understand and generate text with greater coherence and accuracy.
  - Potential improvements in handling more complex queries and producing more contextually relevant responses.
  - Likely to incorporate advancements in AI safety and ethical considerations to mitigate risks associated with misuse.

### GPT-4O (OpenAI's newer models):
- **Release**: Hypothetical future models.
- **Parameters**: Potentially even larger than GPT-4, but specifics would depend on advancements in computational power and data availability.
- **Features**:
  - Emphasis on optimization for specific applications, improving efficiency and accuracy in targeted tasks.
  - Incorporation of multi-modal capabilities, enabling the model to handle not just text but also images, audio, and other data types.
  - Enhanced contextual understanding and memory, allowing for more sophisticated interactions and long-term coherence in dialogues.
  - Continued focus on ethical AI, ensuring safe and responsible use of AI technologies.

### Summary of Differences:
- **Model Size**: Each subsequent version has a significantly larger number of parameters.
- **Capabilities**: Improvements in zero-shot, one-shot, and few-shot learning, allowing for better performance on a variety of tasks without task-specific training.
- **Applications**: Expanding use cases, including multi-modal processing, enhanced dialogue systems, and specialized applications.
- **Ethical Considerations**: Increasing emphasis on AI safety, reducing biases, and mitigating risks associated with misuse.

These differences illustrate the rapid advancements in the capabilities of GPT models, driven by increases in model size, diversity of training data, and sophisticated learning techniques.

### Sources:
- [GPT-2 Release](https://openai.com/blog/better-language-models/)
- [GPT-3 Release](https://openai.com/blog/gpt-3-apps/)
- [GPT-4 Expectations](https://openai.com/research/gpt-4)
- [Discussion on Future Models](https://towardsdatascience.com/)

For more detailed information, please refer to the official OpenAI blogs and research papers.

## Improving Language Understanding by Generative Pre-Training (GPT-1 by OpenAI)
[Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

The paper introduces the Generative Pre-trained Transformer (GPT). This work demonstrates how pre-training a transformer model on a large corpus of text can significantly enhance the model's performance on a variety of natural language processing (NLP) tasks through fine-tuning.

### Key Contributions:

1. **Model Architecture**:
   - The model is based on the Transformer architecture, which uses self-attention mechanisms to handle long-range dependencies in text more effectively than traditional RNNs or LSTMs.

2. **Unsupervised Pre-Training**:
   - GPT is pre-trained on a large corpus of text (BooksCorpus), learning to predict the next word in a sentence. This unsupervised learning phase helps the model capture a wide range of language patterns and knowledge.
<img width="1157" alt="Screenshot 2024-07-23 at 4 44 29 PM" src="https://github.com/user-attachments/assets/c3c8ef9e-d272-4cee-be22-8b39435c51f4">

3. **Supervised Fine-Tuning**:
   - After pre-training, GPT is fine-tuned on specific tasks with labeled data. The pre-trained model parameters provide a strong starting point, allowing the model to achieve better performance with less task-specific data. It's found that including language modeling as an auxiliary objective to the fine-tuning helped learning by (a) improving generalization of the supervised model, and (b) accelerating convergence.
   - 
  <img width="1143" alt="Screenshot 2024-07-23 at 4 44 41 PM" src="https://github.com/user-attachments/assets/e7621fdb-73f7-4296-85c0-02c2457d1c05">

     

4. **Task-specific input transformations**:
   - The paper shows that GPT achieves state-of-the-art results on several NLP benchmarks, including natural language inference (NLI), question answering (QA), and semantic similarity tasks.
   - Fine-tuning the pre-trained model on these tasks leads to significant improvements compared to training from scratch.
  <img width="1015" alt="Screenshot 2024-07-23 at 4 48 06 PM" src="https://github.com/user-attachments/assets/c43bf81a-683a-4860-9e35-8b039233df7a">

     

5. **Generalization and Transfer Learning**:
   - GPT demonstrates strong transfer learning capabilities, as the pre-trained model can be adapted to various downstream tasks with minimal task-specific modifications.
   - This approach reduces the need for large annotated datasets for each new task, making it more efficient and scalable.

### Results:
- The paper reports that GPT, when fine-tuned on the Multi-Genre Natural Language Inference (MNLI) dataset, achieves an accuracy comparable to models that use task-specific architectures.
- On the Stanford Question Answering Dataset (SQuAD), GPT's fine-tuned model performs competitively with state-of-the-art models, showcasing its versatility and effectiveness.

### Conclusion:
"Improving Language Understanding by Generative Pre-Training" highlights the potential of generative pre-training as a powerful method for improving language understanding. By leveraging large-scale unsupervised pre-training, GPT can generalize well across various NLP tasks, reducing the need for extensive labeled data and task-specific architectures.

## GPT-1: Decoder only Architecture
A decoder-only architecture, such as the one used in GPT-1, does include embeddings. The embeddings are a crucial part of the model, responsible for converting input tokens (words, subwords, or characters) into continuous vector representations that the model can process.

### Key Components of the Decoder-Only Architecture:

1. **Token Embeddings**:
   - The input tokens are mapped to dense vectors using an embedding matrix. This process translates discrete tokens into continuous-valued vectors, which are then fed into the transformer layers.
   - Each token in the input sequence has a corresponding embedding vector.

2. **Positional Embeddings**:
   - Since transformers do not have a built-in notion of token order, positional embeddings are added to the token embeddings to provide information about the position of each token in the sequence.
   - This allows the model to consider the order of tokens, which is crucial for understanding the context in natural language.

3. **Transformer Layers**:
   - The core of the architecture consists of multiple transformer decoder layers. Each layer has a multi-head self-attention mechanism and position-wise feed-forward networks.
   - The self-attention mechanism allows the model to focus on different parts of the input sequence when producing the output.

### How It Works in GPT-1:

1. **Input Tokens**: The input text is tokenized into a sequence of tokens.
2. **Embedding Layer**: Each token is mapped to its corresponding embedding vector, and positional embeddings are added to these token embeddings.
3. **Transformer Decoder Layers**: The embedded sequence is passed through several transformer decoder layers, where self-attention mechanisms allow the model to weigh the importance of different tokens in the sequence.
4. **Output**: The final layer generates the output sequence, predicting the next token in the sequence based on the previous tokens.

### Summary:
- **Embeddings**: Both token and positional embeddings are essential in the decoder-only architecture to provide meaningful vector representations and positional information.
- **Decoder-Only Architecture**: While it primarily focuses on generating sequences, it relies on embeddings to handle input tokens and their positions effectively.

Embeddings are a fundamental part of the architecture, ensuring that the model can effectively process and generate natural language sequences.

Here's a simplified implementation of GPT-1 in PyTorch, focusing on the key components such as embeddings, transformer decoder layers, and positional encodings. This code is meant to illustrate the structure and primary components of GPT-1:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT1(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, dropout=0.1):
        super(GPT1, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.02)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x, memory):
        seq_length = x.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        
        x = self.dropout(token_embeddings + position_embeddings)
        x = x.permute(1, 0, 2)  # Convert to (seq_length, batch_size, d_model) for Transformer
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)
        
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_length, d_model)
        
        logits = self.fc_out(output)
        return logits

# Hyperparameters
vocab_size = 50257  # Example vocab size (for GPT-1)
d_model = 768  # Dimension of the model
nhead = 12  # Number of attention heads
num_layers = 12  # Number of transformer layers
max_seq_length = 512  # Maximum sequence length
dropout = 0.1  # Dropout rate

# Example usage
model = GPT1(vocab_size, d_model, nhead, num_layers, max_seq_length, dropout)
input_ids = torch.randint(0, vocab_size, (2, max_seq_length))  # Example input (batch_size, seq_length)
memory = torch.zeros((max_seq_length, 2, d_model))  # Example memory (seq_length, batch_size, d_model)

output = model(input_ids, memory)
print(output.shape)  # Output shape: (batch_size, seq_length, vocab_size)
```

### Explanation:
1. **Embeddings**:
   - `token_embedding`: Embeds input tokens.
   - `position_embedding`: Embeds positional information to retain the order of tokens.

2. **Transformer Decoder**:
   - `nn.TransformerDecoderLayer`: Defines a single layer of the transformer decoder.
   - `nn.TransformerDecoder`: Stacks multiple decoder layers.

3. **Forward Pass**:
   - The input tokens are embedded and summed with their positional embeddings.
   - The embeddings are then passed through the transformer decoder.
   - The output is transformed through a linear layer to produce logits for each token in the vocabulary.

4. **Usage**:
   - The model is instantiated with hyperparameters.
   - Example input tensors are created and passed through the model to generate predictions.

This implementation simplifies some aspects of GPT-1 for clarity and learning purposes. For a production-grade model, additional components such as training loops, data preprocessing, and optimization routines would be required.

### Key Differences between BERT and GPT-1:
BERT (Bidirectional Encoder Representations from Transformers) and GPT-1 (Generative Pre-trained Transformer 1) are both influential language models developed by Google AI and OpenAI, respectively. They have different architectures and training objectives, leading to distinct use cases and strengths.

#### 1. **Architecture**:
- **BERT**:
  - **Encoder-Only Architecture**: BERT uses only the encoder part of the transformer architecture.
  - **Bidirectional**: BERT processes the text in both directions (left-to-right and right-to-left) simultaneously, allowing it to capture context from both directions.
  - **Token Masking**: BERT is trained using a masked language model (MLM) objective, where some percentage of the input tokens are masked, and the model learns to predict these masked tokens based on their context.

- **GPT-1**:
  - **Decoder-Only Architecture**: GPT-1 uses only the decoder part of the transformer architecture.
  - **Unidirectional**: GPT-1 processes the text in a left-to-right manner, predicting the next word based on the previous words in the sequence.
  - **Next Token Prediction**: GPT-1 is trained using a language model (LM) objective, where it learns to predict the next token in a sequence given the previous tokens.

#### 2. **Training Objectives**:
- **BERT**:
  - **Masked Language Modeling (MLM)**: During training, BERT randomly masks some tokens in the input and trains the model to predict these masked tokens based on their context.
  - **Next Sentence Prediction (NSP)**: BERT is also trained to predict whether a given sentence B follows sentence A, which helps the model understand sentence relationships.

- **GPT-1**:
  - **Autoregressive Language Modeling**: GPT-1 is trained to predict the next token in a sequence, making it naturally suited for text generation tasks.

#### 3. **Use Cases**:
- **BERT**:
  - **Natural Language Understanding (NLU)**: BERT excels in tasks that require a deep understanding of the text, such as question answering, sentiment analysis, named entity recognition, and text classification.
  - **Bidirectional Context**: Its ability to capture bidirectional context makes it particularly powerful for understanding the meaning and context of words within a sentence.

- **GPT-1**:
  - **Text Generation**: GPT-1 is primarily used for generating coherent and contextually relevant text, making it suitable for applications like language modeling, text completion, and conversational AI.
  - **Sequential Context**: Its unidirectional nature allows it to effectively generate text in a coherent sequence.

### Summary of Differences:

- **BERT**:
  - Encoder-only, bidirectional
  - Masked language modeling and next sentence prediction objectives
  - Excellent for NLU tasks

- **GPT-1**:
  - Decoder-only, unidirectional
  - Autoregressive language modeling objective
  - Primarily used for text generation

These differences highlight how BERT and GPT-1 are optimized for different types of NLP tasks, leveraging their unique architectures and training methodologies.

### Why not use encoder attention layer to learn embedding in GPT-1?
Using an attention encoder layer to learn embeddings for GPT-1 was not the chosen approach because of several design and objective considerations. Here’s a detailed explanation of why GPT-1 was designed with a decoder-only architecture and the implications of this choice:

### Design and Objective Considerations:

1. **Training Objective Alignment**:
   - **GPT-1**: The goal was to create a generative model that excels at text generation tasks. The decoder-only architecture with its autoregressive nature aligns perfectly with this objective. The model predicts the next word in a sequence, making it inherently suited for tasks like text completion, generation, and language modeling.
   - **Encoder Layers**: Encoders are typically designed to understand and process entire input sequences simultaneously, capturing bidirectional context. This is more aligned with tasks that require understanding and interpreting text, such as classification, sentiment analysis, and other comprehension tasks, rather than generating text.

2. **Architectural Simplicity and Focus**:
   - **Simplicity**: A decoder-only model is simpler and more focused on its task of generation. It reduces the complexity that comes with integrating encoder-decoder interactions. By focusing on a single-directional flow, it simplifies the training and inference processes.
   - **Efficiency**: Decoding involves sequential predictions, which fit naturally with an autoregressive process. Introducing an encoder layer would add complexity and potential redundancy in a model meant to generate sequences one token at a time.

3. **Different Use Cases**:
   - **BERT**: Uses an encoder-only architecture to perform well on understanding tasks. It is designed to create context-aware embeddings of input sequences, which are then used for various downstream tasks requiring comprehension.
   - **GPT-1**: Uses a decoder-only architecture to excel at generating coherent and contextually relevant text. The focus is on leveraging the learned sequential patterns in the training data to predict future tokens.

### Technical and Conceptual Points:

1. **Bidirectional vs. Unidirectional Context**:
   - **BERT’s Bidirectional Context**: BERT’s encoder learns embeddings that consider context from both directions (past and future tokens). This is beneficial for understanding tasks but introduces challenges for autoregressive generation tasks, where predicting the next token is the goal.
   - **GPT-1’s Unidirectional Context**: GPT-1’s decoder learns embeddings in a left-to-right manner, suitable for generating the next word based on previous words. This unidirectional approach is efficient and directly supports the generative objective.

2. **Embedding Learning**:
   - **BERT’s Approach**: Uses attention in encoders to learn embeddings by considering all positions in the input sequence simultaneously, enabling it to understand the full context.
   - **GPT-1’s Approach**: Learns embeddings that support the autoregressive generation process, focusing on maintaining the flow of text in a logical and coherent manner.

In conclusion, the design choice for GPT-1 to use a decoder-only architecture was driven by the goal of optimizing text generation capabilities, aligning the training objectives with the intended use cases, and maintaining architectural simplicity and efficiency.

## Language Models are Unsupervised Multitask Learners (GPT-2 by OpenAI)

[Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### Key Points and Contributions:

1. **Model and Dataset**:
   - The paper introduces GPT-2, a 1.5 billion parameter Transformer-based language model.
   - GPT-2 is trained on WebText, a dataset created from scraping and filtering content from outbound links on Reddit with at least 3 karma, resulting in 45 million links and 40 GB of text after cleaning and deduplication.

2. **Unsupervised Multitask Learning**:
   - The authors demonstrate that GPT-2 can perform various NLP tasks in a zero-shot setting, meaning the model can perform tasks it was not explicitly trained on by conditioning on appropriate prompts.
   - Tasks include question answering, machine translation, reading comprehension, summarization, and more.

3. **Zero-Shot Task Performance**:
   - **Question Answering**: GPT-2 achieves a 55 F1 score on the CoQA dataset, matching or exceeding 3 out of 4 baselines without using the 127,000+ training examples.
   - **Translation**: GPT-2 performs translation tasks by being conditioned on example translation pairs, showing significant performance improvements over baseline unsupervised methods.
   - **Summarization**: By conditioning on article text with a TL;DR prompt, GPT-2 generates summaries that are qualitatively coherent, though still rudimentary compared to state-of-the-art supervised models.

4. **Model Size and Performance**:
   - The paper shows that increasing model capacity improves performance log-linearly across tasks. Larger models achieve better results on various language modeling benchmarks and specific tasks.
   - GPT-2 achieves state-of-the-art results on 7 out of 8 tested language modeling datasets in a zero-shot setting, significantly outperforming previous models.

5. **Analysis and Implications**:
   - The results suggest that large language models trained on diverse and extensive datasets can learn to perform a wide range of tasks without explicit supervision.
   - The findings highlight the potential of unsupervised multitask learning, paving the way for more general and versatile AI systems.

### Conclusion:
The paper concludes that high-capacity language models like GPT-2, trained on sufficiently large and varied datasets, can perform numerous NLP tasks in a zero-shot setting. This represents a significant step towards developing more general AI systems capable of learning and adapting to various tasks from natural language alone.

### Is GPT-2 trained with multi-task learning objectives?
No, GPT-2 is not explicitly trained with multi-task learning objectives. Instead, GPT-2 follows a single-task training objective: unsupervised language modeling. Here are some key points to understand this better:

### Training Objective:
- **Unsupervised Language Modeling**: GPT-2 is trained to predict the next word in a sequence given all the previous words. This is done using a maximum likelihood estimation approach where the model learns to generate the next token based on the context provided by the preceding tokens. This training objective is simple but powerful, as it enables the model to learn a wide range of linguistic patterns and structures from the training data.

### Key Characteristics:
- **Large-Scale Pre-Training**: GPT-2 is pre-trained on a diverse and large-scale corpus called WebText, which contains high-quality web pages. The vast amount of training data allows GPT-2 to learn general language representations.
- **Zero-Shot, One-Shot, and Few-Shot Learning**: Despite being trained with a single-task objective, GPT-2 demonstrates strong zero-shot, one-shot, and few-shot learning capabilities. This means it can perform various tasks by conditioning on task-specific prompts without additional task-specific training. This flexibility arises from the model's ability to generalize the patterns learned during pre-training.

### Capabilities and Performance:
- **Generalization**: GPT-2 can generate coherent and contextually relevant text across a wide range of tasks, such as translation, summarization, question answering, and more. This is achieved through its ability to understand and generate text based on the input context.
- **No Explicit Task-Specific Objectives**: Unlike multi-task learning models, which are trained on multiple tasks with distinct objectives, GPT-2 does not have separate training phases or objectives for different tasks. Instead, it relies on its general language modeling capabilities to adapt to various tasks.

In summary, GPT-2 is trained with a single-task objective focused on unsupervised language modeling, but it exhibits strong generalization capabilities that allow it to perform well across a variety of tasks in a zero-shot or few-shot setting.

## Language Models are Few-Shot Learners (GPT-3 by OpenAI, 2020)

## Training language models to follow instructions with human feedback (InstructGPT, ChatGPT by OpenAI, 2022)
[Paper](https://arxiv.org/pdf/2203.02155)
<img width="1150" alt="Screenshot 2024-07-30 at 4 09 43 PM" src="https://github.com/user-attachments/assets/2c1575b3-1907-4455-903c-e8b85131dd96">



