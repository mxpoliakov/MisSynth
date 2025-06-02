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

### Synthetic fallacies
Generate synthetic fallacies based on [single class prompt template](../prompt_templates/single-class-synthetic-fallacy.txt). Vector store is used to get relevant article excerpts to argument claim (essentially this is mini RAG with metadata filtering). [OpenAI o3-mini](https://openai.com/index/openai-o3-mini) model is used to generate 20 synthetic fallacies per one sample from missci test split.

```bash
export OPENAI_API_KEY=...
python generate_synthetic_data.py --prompt_template single-class-synthetic-fallacy
```
### Fine-tune LLM on synthetic fallacies
Create a dataset using raw data from the previous step. For the baseline experiment, we will classify fallacies without premise using [p1-basic template](../missci/prompt_templates/gen_cls/p1-basic-D.txt). Given the synthetic fallacies generated, we can fill out the template and provide responses to fine-tune the LLM. Let's fine-tune [Phi-4 from Microsoft](https://huggingface.co/mlx-community/phi-4-8bit) with synthetic fallacies.

```bash
python create_fine_tuning_dataset.py --raw-output-folders o3-mini-single-class-synthetic-fallacy-20

python -m mlx_lm.lora --model mlx-community/phi-4-8bit --data output/o3-mini-single-class-synthetic-fallacy-20 \
--train --fine-tune-type lora --batch-size 1 --num-layers 16 --iters 1000 --adapter-path adapters
```

### Benchmark vanilla model vs fine-tuned model
Benchmark on dev missci split to avoid data leakage:
```bash
python run_mlx_fallacy_classification.py --model-name phi-4-8bit
python run_mlx_fallacy_classification.py--model-name phi-4-8bit --adapter-path adapters
```
```bash
cd missci

python run-fallacy-classification-without-premise.py parse-llm-output \
phi-4-8bit_cls_without_premise_p4-connect-cls-D_dev.jsonl --dev

python run-fallacy-classification-without-premise.py parse-llm-output \
phi-4-8bit_cls_without_premise_p4-connect-cls-D_dev_adapters.jsonl --dev
```
Vanilla output: `ACC: 0.222 ; F1 MACRO: 0.137`. Fine-tuned output: `ACC: 0.292 ; F1 MACRO: 0.259`.

### More models

| Model                     | Vanilla acc | Vanilla F1 | Finetune acc | Finetune F1 | Lora layers | Params |
|---------------------------|-------------|------------|--------------|-------------|-------------|--------|
| TinyLlama                 | 0.097       | 0.037      | 0.139        | 0.074       | 2           | 1.1B   |
| Qwen2.5 1.5B (4 bit)      | 0.111       | 0.023      | 0.194        | 0.137       | 2           | 1.5B   |
| Phi-4 (8 bit)             | 0.222       | 0.137      | 0.292        | 0.259       | 16          | 14.7B  |
