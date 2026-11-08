# looped_model_experments

Minimal, Colab-friendly scaffold to reproduce looped latent-reasoning experiments and compare optimizers.

## Google Colab setup

```python
!git clone https://github.com/Abrish-21/looped_model_experments.git
%cd looped_model_experments
!pip install -r requirements.txt
```

## Run a quick optimizer comparison

```python
!python run_optimizer_sweep.py --epochs 3 --steps 40 --optimizers adamw adam sgd --lr 0.001
```

This runs a compact recurrent/looped transformer-style model (shared block applied multiple times) on a parity reasoning task and reports final evaluation loss/accuracy per optimizer.

## Use from notebook/Python directly

```python
from looped_model_experiments import OptimizerSpec, run_optimizer_comparison

results = run_optimizer_comparison(
    [
        OptimizerSpec("adamw", 1e-3),
        OptimizerSpec("adam", 1e-3),
        OptimizerSpec("sgd", 1e-2),
    ],
    epochs=5,
    train_steps_per_epoch=50,
    batch_size=64,
    seq_len=24,
)
print(results)
```
