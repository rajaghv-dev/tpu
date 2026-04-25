# Google TPU Examples

A collection of practical examples for training and running ML models on Google Cloud TPUs using JAX, Flax, and TensorFlow.

## Examples

| # | Example | Framework | Description |
|---|---------|-----------|-------------|
| 01 | [Hello TPU](01_hello_tpu/) | JAX | Verify TPU setup and run basic tensor ops |
| 02 | [MNIST Classification](02_mnist_classification/) | JAX + Flax | Simple CNN image classifier on MNIST |
| 03 | [ResNet ImageNet](03_resnet_imagenet/) | JAX + Flax | ResNet-50 training on ImageNet with `pmap` |
| 04 | [BERT Fine-tuning](04_bert_finetuning/) | JAX + Flax | Fine-tune BERT on GLUE benchmark tasks |
| 05 | [GPT Pre-training](05_gpt_pretraining/) | JAX + Flax | Minimal GPT trained from scratch |
| 06 | [Data Pipeline](06_data_pipeline/) | TF Data + JAX | Efficient TPU-compatible `tf.data` pipelines |
| 07 | [Custom Training Loop](07_custom_training_loop/) | JAX | Manual gradient accumulation and mixed precision |
| 08 | [Multi-Host Training](08_multi_host/) | JAX | Multi-host TPU pod training with `jax.distributed` |

## Prerequisites

### Hardware
- Google Cloud TPU v2/v3/v4 VM, or
- Google Colab (free TPU tier available)

### Software
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Provision a TPU VM** (Cloud TPU or Colab)
2. **Clone this repo** on the VM
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify your TPU**:
   ```bash
   python 01_hello_tpu/hello_tpu.py
   ```

## TPU Basics

### Detect TPU devices (JAX)
```python
import jax
print(jax.devices())          # e.g. [TpuDevice(id=0, ...), ...]
print(jax.device_count())     # 8 on a v3-8
```

### Parallelise across all TPU cores with `pmap`
```python
import jax
import jax.numpy as jnp

@jax.pmap
def f(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2  # should be all-ones

x = jnp.ones((jax.device_count(),))
print(f(x))
```

## Folder Layout
```
google-tpu-examples/
├── 01_hello_tpu/
├── 02_mnist_classification/
├── 03_resnet_imagenet/
├── 04_bert_finetuning/
├── 05_gpt_pretraining/
├── 06_data_pipeline/
├── 07_custom_training_loop/
├── 08_multi_host/
├── scripts/
│   ├── gcloud_setup.sh        # enable APIs, set project
│   ├── provision_tpu.sh       # create TPU VM + install deps
│   ├── gcloud_ssh_run.sh      # run any example on a remote TPU VM
│   ├── gcloud_upload_data.sh  # upload local data to GCS
│   ├── gcloud_pod_run.sh      # multi-host pod launch
│   └── teardown_tpu.sh        # delete TPU VM to stop billing
├── requirements.txt
└── README.md
```

## gcloud Scripts

### One-time project setup
```bash
./scripts/gcloud_setup.sh my-gcp-project-id
```

### Provision a TPU VM
```bash
./scripts/provision_tpu.sh tpu-demo us-central1-a v3-8
```

### Run an example remotely
```bash
./scripts/gcloud_ssh_run.sh tpu-demo us-central1-a 02_mnist_classification
./scripts/gcloud_ssh_run.sh tpu-demo us-central1-a 03_resnet_imagenet "--epochs=5"
```

### Upload data to GCS
```bash
./scripts/gcloud_upload_data.sh ./data/imagenet gs://my-bucket imagenet
```

### Launch multi-host pod training
```bash
./scripts/gcloud_pod_run.sh my-v3-32-pod us-central1-a
```

### Delete TPU when done
```bash
./scripts/teardown_tpu.sh tpu-demo us-central1-a
```

## Cloud TPU Cheatsheet

```bash
# Create a TPU v3-8 VM
gcloud compute tpus tpu-vm create tpu-demo \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-base

# SSH in
gcloud compute tpus tpu-vm ssh tpu-demo --zone=us-central1-a

# List running TPUs
gcloud compute tpus tpu-vm list --zone=us-central1-a

# Delete when done (billing stops immediately)
gcloud compute tpus tpu-vm delete tpu-demo --zone=us-central1-a
```

## License

MIT
