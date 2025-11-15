# Poisson PDE - DeepONet Implementation

This subfolder contains the DeepONet implementation specifically for solving Poisson partial differential equations.

## Files in this folder:

- **`Deeponet_Poisson.ipynb`** - Main Jupyter notebook with the complete DeepONet implementation for Poisson equations
- **`Poisson_samples.npz`** - Training dataset containing diffusivity fields (m_samples) and corresponding solutions (u_samples)
- **`Poisson_FNO_samples.npz`** - Additional data samples for comparison or extended training

## Problem Description

The DeepONet learns to solve the Poisson equation:
```
-∇·(m(x)∇u(x)) = f(x)
```

Where:
- `m(x)` is the diffusivity field (input function)
- `u(x)` is the solution field (output function)
- `f(x)` is the source term

## Data Format

The NPZ files contain:
- `m_samples`: Input diffusivity fields
- `u_samples`: Corresponding solution fields
- `u_mesh_nodes`: Mesh coordinates for the domain
- `u_mesh_dirichlet_boundary_nodes`: Boundary node indices for Dirichlet conditions

## Usage

1. Open the Jupyter notebook: `Deeponet_Poisson.ipynb`
2. Run all cells to train and evaluate the DeepONet model
3. The notebook will automatically load the NPZ data files from this directory

## Model Architecture

- **Branch Network**: Processes the input diffusivity field
- **Trunk Network**: Processes the spatial coordinates
- **Output**: Predicted solution field via tensor product

## Training Parameters

- Training samples: 3,500
- Test samples: 1,000
- Input points: 2,601
- Output points: 2,601
- Network depth: 3 layers
- Network width: 64 neurons
- Learning rate: 1e-3
- Batch size: 20
- Epochs: 1,000
