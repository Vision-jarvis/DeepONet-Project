"""
DeepONet for 2+1D Burgers' Equation
====================================
This script implements:
1. Burgers2DDataset: Efficient data loader for spatiotemporal data
2. DeepONet2D: Branch-Trunk architecture for operator learning
3. Training loop with visualization utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ==========================================
# 1. DATASET CLASS
# ==========================================
class Burgers2DDataset(Dataset):
    """
    PyTorch Dataset for 2D Burgers' equation spatiotemporal data.
    
    Dynamically samples random (x, y, t) query points to manage memory.
    """
    def __init__(self, npz_file, num_points_per_sample=1000, train=True):
        """
        Parameters:
        -----------
        npz_file : str
            Path to .npz data file
        num_points_per_sample : int
            Number of random spatiotemporal points to sample per trajectory
        train : bool
            If True, use random sampling. If False, use fixed grid for evaluation.
        """
        data = np.load(npz_file)
        
        # Load data - KEEP AS NUMPY for memory efficiency
        # Only convert to tensor in __getitem__
        self.u0 = data['u0_samples']  # (N, nx*ny) - numpy array
        self.solutions = data['solutions']  # (N, nt, nx, ny) - numpy array
        
        self.t = data['t']
        self.x = data['x']
        self.y = data['y']
        
        # Store domain bounds for normalization
        self.x_max = 2 * np.pi  # Domain is [0, 2π]
        self.y_max = 2 * np.pi
        self.t_max = self.t[-1]  # Should be 1.0
        
        self.num_samples = self.u0.shape[0]
        self.nt = len(self.t)
        self.nx = len(self.x)
        self.ny = len(self.y)
        
        self.num_points = num_points_per_sample
        self.train = train
        
        print(f"Dataset loaded: {self.num_samples} samples")
        print(f"  Grid: {self.nx} × {self.ny}")
        print(f"  Time steps: {self.nt}")
        print(f"  Points per sample: {self.num_points}")
        print(f"  Memory-efficient: data kept as numpy arrays")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
        --------
        branch_input : Tensor (nx*ny,)
            Initial condition (flattened)
        trunk_input : Tensor (num_points, 3)
            Query coordinates (x, y, t) - NORMALIZED
        target : Tensor (num_points, 1)
            Solution values at query points
        """
        # Branch Input: The Initial Condition (convert to tensor here)
        branch_input = torch.tensor(self.u0[idx], dtype=torch.float32)
        
        if self.train:
            # Training: Random sampling of (x, y, t) points
            t_idx = torch.randint(0, self.nt, (self.num_points,))
            x_idx = torch.randint(0, self.nx, (self.num_points,))
            y_idx = torch.randint(0, self.ny, (self.num_points,))
        else:
            # Evaluation: Fixed grid sampling
            # Sample uniformly across the domain
            t_idx = torch.randint(0, self.nt, (self.num_points,))
            x_idx = torch.randint(0, self.nx, (self.num_points,))
            y_idx = torch.randint(0, self.ny, (self.num_points,))
        
        # Get the coordinate values
        t_val = self.t[t_idx.numpy()]
        x_val = self.x[x_idx.numpy()]
        y_val = self.y[y_idx.numpy()]
        
        # *** CRITICAL FIX: NORMALIZE TRUNK INPUTS ***
        # x, y ∈ [0, 2π] → normalize to [0, 1]
        # t already ∈ [0, 1] but normalize for consistency
        x_norm = x_val / self.x_max
        y_norm = y_val / self.y_max
        t_norm = t_val / self.t_max
        
        # Stack to form (P, 3) input for Trunk
        trunk_input = torch.stack([
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(t_norm, dtype=torch.float32)
        ], dim=1)
        
        # Get the Target values u(x, y, t) (convert to tensor here)
        target_vals = self.solutions[idx, t_idx.numpy(), x_idx.numpy(), y_idx.numpy()]
        target = torch.tensor(target_vals, dtype=torch.float32).unsqueeze(1)  # (P, 1)
        
        return branch_input, trunk_input, target


# ==========================================
# 2. DEEPONET MODEL
# ==========================================
class DeepONet2D(nn.Module):
    """
    Deep Operator Network for 2+1D problems.
    
    Architecture:
    - Branch Net: Encodes the initial condition
    - Trunk Net: Encodes the spatiotemporal coordinates (x, y, t)
    - Output: Dot product of branch and trunk features
    """
    def __init__(self, branch_input_dim=2601, trunk_input_dim=3, 
                 hidden_dim=128, output_dim=100):
        """
        Parameters:
        -----------
        branch_input_dim : int
            Dimension of flattened IC (nx*ny)
        trunk_input_dim : int
            Dimension of coordinates (3 for x, y, t)
        hidden_dim : int
            Hidden layer width
        output_dim : int
            Number of basis functions (p)
        """
        super(DeepONet2D, self).__init__()
        
        # Branch Net: Maps IC to feature space
        # Using GELU for smooth input functions (Gaussian Random Fields)
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trunk Net: Maps coordinates to feature space
        # Using Tanh for coordinate encoding (works well for bounded inputs)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, coords):
        """
        Forward pass.
        
        Parameters:
        -----------
        u : Tensor (batch, branch_input_dim)
            Initial conditions
        coords : Tensor (batch, num_points, trunk_input_dim)
            Query coordinates
            
        Returns:
        --------
        output : Tensor (batch, num_points, 1)
            Predicted solution values
        """
        batch_size = u.shape[0]
        num_points = coords.shape[1]
        
        # Branch output: (batch, output_dim)
        B = self.branch(u)
        
        # Trunk output: (batch, num_points, output_dim)
        # Need to reshape coords for batch processing
        coords_reshaped = coords.view(batch_size * num_points, -1)
        T = self.trunk(coords_reshaped)
        T = T.view(batch_size, num_points, -1)
        
        # Dot product: (batch, 1, output_dim) * (batch, num_points, output_dim)
        # Sum over the feature dimension
        B_expanded = B.unsqueeze(1)  # (batch, 1, output_dim)
        output = torch.sum(B_expanded * T, dim=2, keepdim=True) + self.bias
        
        return output  # (batch, num_points, 1)


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_burgers_deeponet(data_path='Burgers2D_samples.npz', 
                          epochs=100, 
                          batch_size=32,
                          learning_rate=1e-3,
                          hidden_dim=128,
                          output_dim=100,
                          num_points=2000,
                          save_path='models'):
    """
    Train DeepONet on Burgers' equation data.
    
    Parameters:
    -----------
    data_path : str
        Path to training data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    hidden_dim : int
        Hidden layer width
    output_dim : int
        Number of basis functions
    num_points : int
        Points sampled per trajectory
    save_path : str
        Directory to save models
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("DEEPONET TRAINING - 2D BURGERS' EQUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Load Dataset
    dataset = Burgers2DDataset(data_path, num_points_per_sample=num_points, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    model = DeepONet2D(
        branch_input_dim=dataset.nx * dataset.ny,
        trunk_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    
    # Training loop
    loss_history = []
    best_loss = float('inf')
    
    print("\nStarting training...\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (u, coords, target) in enumerate(dataloader):
            u = u.to(device)
            coords = coords.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(u, coords)
            
            # Loss
            loss = criterion(prediction, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, 'best_model.pth'))
    
    print("\n" + "="*60)
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print("="*60)
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss History')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_history.png'), dpi=150)
    print(f"Loss plot saved to {save_path}/loss_history.png")
    
    return model, loss_history


# ==========================================
# 4. VISUALIZATION
# ==========================================
def visualize_prediction(model, dataset, sample_idx=0, time_indices=[0, 10, 19], 
                        save_path='models'):
    """
    Visualize model predictions vs ground truth at different time snapshots.
    
    Parameters:
    -----------
    model : DeepONet2D
        Trained model
    dataset : Burgers2DDataset
        Dataset object
    sample_idx : int
        Sample to visualize
    time_indices : list
        Time indices to plot
    save_path : str
        Directory to save plots
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get sample (convert to tensor)
    u0 = torch.tensor(dataset.u0[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    solution_true = dataset.solutions[sample_idx]  # numpy array
    
    # Create full grid of coordinates
    X, Y = np.meshgrid(dataset.x, dataset.y, indexing='ij')
    
    fig, axes = plt.subplots(2, len(time_indices), figsize=(5*len(time_indices), 10))
    
    for i, t_idx in enumerate(time_indices):
        t_val = dataset.t[t_idx]
        
        # Create coordinate tensor for this time
        coords_x = X.flatten()
        coords_y = Y.flatten()
        coords_t = np.full_like(coords_x, t_val)
        
        # *** CRITICAL: NORMALIZE COORDINATES (same as in dataset) ***
        coords_x_norm = coords_x / dataset.x_max
        coords_y_norm = coords_y / dataset.y_max
        coords_t_norm = coords_t / dataset.t_max
        
        coords = torch.stack([
            torch.tensor(coords_x_norm, dtype=torch.float32),
            torch.tensor(coords_y_norm, dtype=torch.float32),
            torch.tensor(coords_t_norm, dtype=torch.float32)
        ], dim=1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(u0, coords).squeeze().cpu().numpy()
        pred = pred.reshape(dataset.nx, dataset.ny)
        
        # Ground truth
        true = solution_true[t_idx]
        
        # Plot
        vmin = min(pred.min(), true.min())
        vmax = max(pred.max(), true.max())
        
        im1 = axes[0, i].imshow(true.T, origin='lower', cmap='RdBu_r', 
                               vmin=vmin, vmax=vmax, aspect='auto')
        axes[0, i].set_title(f'True (t={t_val:.3f})')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, i])
        
        im2 = axes[1, i].imshow(pred.T, origin='lower', cmap='RdBu_r', 
                               vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, i].set_title(f'Predicted (t={t_val:.3f})')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'prediction_sample_{sample_idx}.png'), dpi=150)
    print(f"Prediction plot saved to {save_path}/prediction_sample_{sample_idx}.png")
    plt.close()


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Check if data exists
    data_file = 'Burgers2D_samples.npz'  # Production data
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found!")
        print("Please run 'python Burgers2D_DataGen.py' first to generate the data.")
    else:
        print("\n" + "="*60)
        print("PRODUCTION: Full-Scale Training")
        print("Normalized inputs + GELU/Tanh activations")
        print("="*60 + "\n")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠ WARNING: No GPU detected. Training will use CPU (slower).")
            print("  Expected training time: 4-6 hours on CPU")
            print("  Consider installing CUDA-enabled PyTorch for faster training.")
        
        # Train model - PRODUCTION SETTINGS
        model, history = train_burgers_deeponet(
            data_path=data_file,
            epochs=500,             # Full training with early stopping
            batch_size=32,          # Larger batches for production
            learning_rate=1e-3,     # Standard Adam LR
            hidden_dim=128,         # Deep network
            output_dim=100,         # 100 basis functions
            num_points=2000,        # More query points for better coverage
            save_path='models_production'
        )
        
        # Visualize results
        print("\nGenerating visualizations...")
        dataset = Burgers2DDataset(data_file, num_points_per_sample=1000, train=False)
        visualize_prediction(model, dataset, sample_idx=0, save_path='models_production')
        
        # Generate additional visualizations
        print("\nGenerating additional samples...")
        for sample_idx in [1, 2, 5, 10]:
            visualize_prediction(model, dataset, sample_idx=sample_idx, 
                               save_path='models_production', 
                               time_indices=[0, 25, 50, 75, 99])

        print("\n✓ Production training and visualization complete!")
        print("  Check the 'models_production' directory for saved results.")
        print("\n" + "="*60)
        print("RESULTS:")
        print(f"  Best Loss: {min(history):.6f}")
        print(f"  Final Loss: {history[-1]:.6f}")
        print(f"  Total Epochs: {len(history)}")
        print("="*60)

