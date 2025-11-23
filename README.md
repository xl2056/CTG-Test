# CTGTest

## Overview

This repository contains implementations for trajectory generation and reward learning using deep learning approaches.

### Components

- **MaxEntIRL**: Maximum Entropy Inverse Reinforcement Learning implementation for inferring reward feature weights using linear combinations of features
- **DiffusionGan**: Diffusion Model-based trajectory generation system that creates candidate trajectories for MaxEntIRL, with the inferred rewards used to guide the diffusion process

## Project Structure

```
CTGTest/
├── MaxEntIRL/          # Maximum Entropy IRL implementation
├── DiffusionGan/       # Diffusion-based trajectory generation
├── tbsim/              # Core simulation and model framework
│   ├── algos/          # Algorithm implementations
│   ├── configs/        # Configuration files
│   └── models/         # Model definitions
└── CTG/                # Additional components
```

## Adding New Models

Follow these steps to integrate a new model into the framework:

### Step 1: Algorithm Implementation
Add your algorithm implementation:
```bash
tbsim/algos/algo.py
```

### Step 2: Register Algorithm Factory
Update the algorithm factory:
```bash
tbsim/algos/factory.py
```

### Step 3: Configuration Setup
Define algorithm configuration:
```bash
tbsim/configs/algo_config.py
```

### Step 4: Registry Registration
Add to the configuration registry:
```bash
tbsim/configs/registry.py
```

### Step 5: Model Definition
Implement your model:
```bash
tbsim/models/your_model.py
```

## Getting Started

[Add installation and usage instructions here]

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]