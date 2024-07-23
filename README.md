# LLMs

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

