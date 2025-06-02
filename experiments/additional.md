## Add synthetic claim-premise
Refer to [baseline experiment](baseline.md). This experiment adds synthetic claim-accurate premise pairs with real fallacies to a training dataset from the baseline. This should increase dataset diversity.

```bash
export OPENAI_API_KEY=...
python generate_synthetic_data.py --prompt-template synthetic-claim-premise --n-synthetic-entries 10

python create_fine_tuning_dataset.py --raw-output-folders o3-mini-single-class-synthetic-fallacy-20 \
--raw-output-folders o3-mini-output/o3-mini-synthetic-claim-premise-10

python -m mlx_lm.lora --model mlx-community/phi-4-8bit --data output \
--train --fine-tune-type lora --batch-size 1 --num-layers 16 --iters 1000 --adapter-path adapters
```

Baseline fine-tuned output: `ACC: 0.292 ; F1 MACRO: 0.259`. Add synthetic claim-premise output: `ACC: 0.347 ; F1 MACRO: 0.23`.
