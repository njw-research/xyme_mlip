## E(3)-Equivariant Message Passing Neural Network for MLIP Training

Welcome to the Xyme repository for E(3)-equivariant message passing neural networks applied to machine learning interatomic potentials (MLIP). This repository provides a template for training a neural network using potential energy and forces data to predict the potential energy surface of a molecular system.

## Getting started 

1. Clone the repository from GitHub 

```
git clone https://github.com/njw-research/jax_ml.git
cd jax_ml
```

2. Create a virtual enviroment using uv

```
uv venv .venv
```

If a virtual enviroment already exists then just use ```source .venv/bin/activate```

3. Add package depandancies from pyproject.toml 

```
uv sync
```

4. Test the installation

```
pytest -v tests
```

## Train

To run the training script, use the following command

```
PYTHONPATH=. python run_script/run_training.py
```


## Tips 

To test multiple devices on a single machine, use the following command:

```
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```