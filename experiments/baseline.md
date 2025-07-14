## Baseline experiment
### Environment setup
Experiment hardware: M1 Macbook Pro with 32 GB of RAM
```bash
git clone --recursive https://github.com/mxpoliakov/synthetic-missci.git && cd synthetic-missci
```
```bash
export PYTHONPATH=$(pwd):$(pwd)/missci
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Vector store
Create JSON vector store based on scraped articles (web, pdf) from MISSCI test split. 130 out of 154 articles were scraped and vectorized using [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) with a chunk size of 512 and chunk overlap of 64.
```bash
python create_vector_store.py
```

### Synthetic fallacies (fallacious premise and context)
Generate synthetic fallacies using the [single class prompt template](../prompt_templates/single-class-synthetic-fallacy-context.txt). A vector store is used to retrieve relevant article excerpts to support the argument claim—essentially functioning as a lightweight RAG with metadata filtering. The [OpenAI o3-mini](https://openai.com/index/openai-o3-mini) model is used to generate 30 synthetic fallacies per sample from the Missci [dev split](../missci/dataset/dev.missci.jsonl). Each fallacy includes both a fallacious premise and context.

Additionally, 15 synthetic claim–accurate premise pairs with real fallacies are generated for each entry in the dev split, using the [synthetic claim-premise template](../prompt_templates/synthetic-claim-premise.txt).

```bash
export OPENAI_API_KEY=...
python generate_synthetic_data.py --prompt-template single-class-synthetic-fallacy-context --n-synthetic-entries 30
python generate_synthetic_data.py --prompt-template synthetic-claim-premise --n-synthetic-entries 15
```
### Fine-tune LLM on synthetic fallacies
Create a dataset using raw data from the previous step. For the baseline experiment, we will classify fallacies with premise using [classify with definition template](../missci/prompt_templates/cls_with_premise/classify-D.txt). Given the synthetic fallacies generated, we can fill out the template and provide responses to fine-tune the LLM. Let's fine-tune [Phi-4 from Microsoft](https://huggingface.co/mlx-community/phi-4-8bit) with synthetic fallacies.

```bash
python create_fine_tuning_dataset.py

python -m mlx_lm.lora --model mlx-community/phi-4-8bit --data output \
--train --fine-tune-type lora --batch-size 1 --num-layers 32 --iters 500 --adapter-path adapters
```

### Benchmark vanilla model vs fine-tuned model
Benchmark on dev missci split to avoid data leakage:
```bash
python run_mlx_fallacy_classification.py --model-name phi-4-8bit
python run_mlx_fallacy_classification.py --model-name phi-4-8bit --adapter-path adapters
```
```bash
cd missci

python run-fallacy-classification-with-gold-premise.py parse-llm-output phi-4-8b
it_cls_with_premise_classify-D_test.jsonl

python run-fallacy-classification-with-gold-premise.py parse-llm-output phi-4-8b
it_cls_with_premise_classify-D_test_adapters.jsonl
```

| Model                     | Vanilla acc    | Vanilla F1    | Finetune acc | Finetune F1 | Lora layers | Params |
|---------------------------|----------------|---------------|--------------|-------------|-------------|--------|
| LLaMA 2                   | 0.577 (*)      | 0.464 (*)     | -            | -           | -           | 70B    |
| Phi-4 (8-bit)             | 0.667 (*)      | 0.550 (*)     | 0.791        | 0.693       | 32          | 15B    |

\* Table 3 from [MISSCI: Reconstructing Fallacies in Misrepresented Science](https://arxiv.org/pdf/2406.03181)
