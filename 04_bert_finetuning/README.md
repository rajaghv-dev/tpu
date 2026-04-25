# 04 – BERT Fine-tuning (GLUE / SST-2)

Loads `bert-base-uncased` from HuggingFace and fine-tunes it for sentiment classification on the SST-2 task.

## Concepts
- `FlaxAutoModelForSequenceClassification` from `transformers`
- Linear LR warmup + decay via `optax.linear_schedule`
- `optax.adamw` weight decay
- `pmap` across all TPU cores

## Run
```bash
python train.py
```

## Expected output
```
Epoch 1/3  val_loss=0.2841  val_acc=91.74%
Epoch 2/3  val_loss=0.2103  val_acc=93.12%
Epoch 3/3  val_loss=0.2218  val_acc=93.46%
```
