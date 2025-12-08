"""
Burgers2D Data Generation Script (Fixed & Visuals Optimized)
============================================================
Generates synthetic training data for the 2+1D Burgers' equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class Burgers2DSolver:
    def __init__(self, nx=64, ny=64, T=1.0, nt=100, nu=0.01, length=2*np.pi):
        self.nx, self.ny = nx, ny
        self.T, self.nt = T, nt
        self.nu = nu
        self.length = length
        self.dt = T / (nt - 1)
        
        # Spatial Domain
        self.x = np.linspace(0, length, nx, endpoint=False)
        self.y = np.linspace(0, length, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Spectral Domain (Real FFT)
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, d=length/nx)
        self.ky = 2 * np.pi * np.fft.rfftfreq(ny, d=length/ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.t = np.linspace(0, T, nt)

        # --- DE-ALIASING MASK (FIXED) ---
        self.mask = np.ones_like(self.KX)
        
        # X-Mask (Standard FFT: high freqs in middle)
        kmax_x = nx // 2
        cutoff_x = int(2/3 * kmax_x)
        self.mask[cutoff_x : nx-cutoff_x, :] = 0.0
        
        # Y-Mask (Real FFT: high freqs at end)
        # CRITICAL FIX: Keep bottom 2/3rds, cut top 1/3rd
        ny_rfft = ny // 2 + 1
        cutoff_y = int(2/3 * ny_rfft) 
        self.mask[:, cutoff_y:] = 0.0 

    def generate_initial_condition(self, num_samples, l=0.25, sigma=1.5):
        """
        FIXED: Generates smooth 'blob-like' initial conditions.
        
        Changes:
        1. Increased default 'l' to 0.6 (creates larger blobs).
        2. Switched to Gaussian Filter (removes high-freq grit/patchiness).
        """
        print(f"Generating {num_samples} initial conditions (Smooth GRF)...")
        u0_samples = []
        
        for _ in tqdm(range(num_samples), desc="Generating ICs"):
            # 1. White noise in physical space
            noise = np.random.randn(self.nx, self.ny)
            
            # 2. Transform to Spectral Space
            noise_hat = np.fft.rfft2(noise)
            
            # 3. Apply Gaussian Smoothing Filter (Heat kernel type)
            # This creates very smooth, rounded features
            k_sq = self.K2  # Wave number squared
            envelope = sigma * np.exp(-0.5 * (l**2) * k_sq)
            
            u0_hat = noise_hat * envelope
            
            # 4. Transform back to Physical Space
            u0 = np.fft.irfft2(u0_hat, s=(self.nx, self.ny))
            
            # 5. Normalize to order O(1)
            # This ensures the amplitude is consistent for the solver
            u0 = (u0 - np.mean(u0)) / (np.std(u0) + 1e-10)
            
            u0_samples.append(u0)
            
        return np.array(u0_samples)

    def compute_rhs(self, u_hat):
        # Spectral derivatives
        u_hat_x = 1j * self.KX * u_hat
        u_hat_y = 1j * self.KY * u_hat
        
        # Physical domain for nonlinear term
        u = np.fft.irfft2(u_hat, s=(self.nx, self.ny))
        u_x = np.fft.irfft2(u_hat_x, s=(self.nx, self.ny))
        u_y = np.fft.irfft2(u_hat_y, s=(self.nx, self.ny))
        
        # Nonlinear term: u * (ux + uy)
        nonlinear = u * (u_x + u_y)
        
        # De-alias
        nonlinear_hat = np.fft.rfft2(nonlinear) * self.mask
        
        # Diffusion
        diffusion = -self.nu * self.K2 * u_hat
        
        return -nonlinear_hat + diffusion

    def solve(self, u0):
        u_hat = np.fft.rfft2(u0)
        solution = [u0]
        
        for i in range(1, self.nt):
            # RK4 Integration
            k1 = self.compute_rhs(u_hat)
            k2 = self.compute_rhs(u_hat + 0.5 * self.dt * k1)
            k3 = self.compute_rhs(u_hat + 0.5 * self.dt * k2)
            k4 = self.compute_rhs(u_hat + self.dt * k3)
            
            u_hat = u_hat + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Stability check
            if np.isnan(u_hat).any():
                return None
            
            u_phys = np.fft.irfft2(u_hat, s=(self.nx, self.ny))
            solution.append(u_phys)
            
        return np.array(solution)

def visualize_data_check(filename='Burgers2D_samples.npz'):
    if not os.path.exists(filename): return
    print("\nVisualizing data check...")
    data = np.load(filename)
    sol = data['solutions'][0]
    t = data['t']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    times = [0, len(t)//2, -1]
    
    # Use fixed scale to see decay/evolution clearly
    vmin, vmax = -2, 2 
    
    for ax, idx in zip(axes, times):
        im = ax.imshow(sol[idx].T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f"t = {t[idx]:.2f}")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('data_check_sample.png')
    print("✓ Saved 'data_check_sample.png'")
    plt.show()

def generate_burgers_data(num_samples=100, nx=64, ny=64, nt=100, nu=0.01):
    solver = Burgers2DSolver(nx=nx, ny=ny, nt=nt, nu=nu)
    u0s = solver.generate_initial_condition(num_samples)
    
    print("Solving...")
    solutions = []
    valid_u0 = []
    for i in tqdm(range(num_samples)):
        sol = solver.solve(u0s[i])
        if sol is not None:
            solutions.append(sol)
            valid_u0.append(u0s[i])
            
    solutions = np.array(solutions)
    valid_u0 = np.array(valid_u0).reshape(len(valid_u0), -1) # Flatten Branch Input
    
    np.savez_compressed('Burgers2D_samples.npz', 
                        u0_samples=valid_u0, solutions=solutions,
                        t=solver.t, x=solver.x, y=solver.y)
    
    visualize_data_check('Burgers2D_samples.npz')

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRODUCTION: Full-Scale Data Generation")
    print("="*60 + "\n")
    
    # 1. Switch back to Gaussian Random Fields (The class handles this by default)
    # 2. Use a slightly higher viscosity for the random data to be safe, 
    #    or keep 0.002 if you want to challenge the network with shocks.
    #    Recommendation: nu=0.005 is a happy medium.
    
    generate_burgers_data(
        num_samples=1000,        # 1000 samples for training
        nx=64, ny=64, 
        nt=50,                   # 50 time steps is sufficient
        nu=0.005,                # Low viscosity for shocks
        # output_file='Burgers2D_samples.npz'
    )