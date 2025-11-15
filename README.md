# DeepONet for Poisson Equation

This repository contains an implementation of Deep Operator Networks (DeepONet) for solving Poisson equations. The DeepONet architecture learns to map between function spaces, specifically learning the operator that maps the diffusivity field to the solution of the Poisson equation.

## Overview

The project implements a DeepONet to solve the Poisson equation:
```
-∇·(m(x)∇u(x)) = f(x)
```
where `m(x)` is the diffusivity field and `u(x)` is the solution.

## Features

- **DeepONet Architecture**: Branch network for function inputs and trunk network for coordinate inputs
- **Data Processing**: Utilities for handling training/test data splits
- **Visualization**: Plotting utilities for loss curves and field comparisons
- **Boundary Conditions**: Support for Dirichlet boundary conditions
- **Model Persistence**: Save and load trained models

## Repository Structure

```
├── Poisson_PDE/                   # Poisson equation implementation
│   ├── Deeponet_Poisson.ipynb    # Main Jupyter notebook
│   ├── README.md                 # Poisson-specific documentation
│   └── requirements.txt          # Python dependencies
│   # Note: Data files (*.npz) need to be downloaded separately - see installation instructions
├── README.md                     # This file (main documentation)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── .gitignore                   # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Vision-jarvis/DeepONet-Project.git
cd DeepONet-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download the required data files**:
   
   🔗 **Data Download Link**: https://www.dropbox.com/scl/fo/5dg02otewg7j0bt7rhkuf/APWguPa5ZRka9ePzdNR_dAc/survey_work/problems/poisson/data?dl=0&rlkey=t900geej8y8z327y5f8wu4yc9&subfolder_nav_tracking=1
   
   - Download `Poisson_samples.npz` and `Poisson_FNO_samples.npz`
   - Place them in the `Poisson_PDE/` directory
   - **For Google Colab**: Upload the files to the same directory as the notebook

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook Deeponet_Poisson.ipynb
```

The notebook contains:
1. Data loading and preprocessing
2. DeepONet model definition
3. Training loop with loss monitoring
4. Model evaluation and visualization
5. Comparison plots between true and predicted solutions

## Model Architecture

- **Branch Network**: Multi-layer perceptron that processes the input function (diffusivity field)
- **Trunk Network**: Multi-layer perceptron that processes the output coordinates
- **Final Output**: Dot product between branch and trunk network outputs

## Training Configuration

- Training samples: 3,500
- Test samples: 1,000  
- Input function points: 2,601
- Output function points: 2,601
- Batch size: 20
- Epochs: 1,000
- Learning rate: 1e-3

## Results

The model achieves good accuracy in predicting solutions to the Poisson equation with various diffusivity fields. Training and validation loss curves, along with visual comparisons between true and predicted solutions, are generated during execution.

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Jupyter

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- DeepONet architecture based on the work by Lu et al. (2021)
- Implementation inspired by modern deep learning practices for scientific computing
