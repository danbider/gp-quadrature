#!/usr/bin/env python
"""
Weather data visualization using GP regression with different kernels
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import time

from kernels.squared_exponential import SquaredExponential
from kernels.matern import Matern
from efgpnd import EFGPND

def load_data(filepath):
    """Load the temperature data."""
    print(f"Loading temperature data from {filepath}")
    data = torch.load(filepath)
    return data

def train_gp_model(x, y, kernel, sigmasq=5, epsilon=1e-4, max_iters=50):
    """Train a GP model with the given kernel."""
    print(f"Training GP model with {type(kernel).__name__} kernel")
    
    # Calculate data bounds for initializing lengthscale
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    
    # Update kernel lengthscale
    kernel.lengthscale = 1
    
    # Create and fit the model
    model = EFGPND(
        x=x,
        y=y,
        kernel=kernel,
        sigmasq=sigmasq,
        eps=epsilon
    )
    
    # Optimize hyperparameters
    start_time = time.time()
    
    # Track per-iteration timing and hyperparameters
    iteration_params = []
    
    # Optimize hyperparameters without the callback parameter
    model.optimize_hyperparameters(
        epsilon_values=[epsilon],
        trace_samples_values=[20],
        max_iters=max_iters,
        base_lr=0.005,
        x0=x0, x1=x1  # Pass the data bounds for optimization
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Fit the model
    model.fit()
    
    # Extract trajectory of hyperparameters from the training log after optimization
    trajectory = None
    if hasattr(model, 'training_log') and model.training_log:
        log = model.training_log[0]  # Only one epsilon and trace_samples setting
        trajectory = {
            'lengthscale': log['tracked_lengthscale'],
            'variance': log['tracked_variance'],
            'total_time': total_time,
            'iteration_params': log['tracked_hyperparameters']
        }
    
    return model, trajectory

def plot_temperature_map(x_unscaled, observed_values, posterior_mean, title):
    """Create a map visualization of temperature data with posterior contours."""
    # Extract latitude and longitude
    lats = x_unscaled[:, 0]  # First column contains latitudes
    lons = x_unscaled[:, 1]  # Second column contains longitudes
    
    # Create figure
    fig = plt.figure(figsize=(13, 6.5))
    
    # Create a single plot
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-95, central_latitude=35))
    
    # Set map extent to cover continental USA
    ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())
    
    # Add map features with higher contrast background
    ax.coastlines(resolution="50m", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")  # Lighter gray for better contrast
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")  # Light blue for ocean
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='#6baed6')  # Blue rivers
    
    # Get value range for colormap
    min_val = observed_values.min()
    max_val = observed_values.max()
    
    # Create a colormap with hot temps as bright red and cool temps as blue
    cmap = plt.cm.coolwarm  # Using coolwarm: blue for cool, red for hot
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    # Ensure all arrays have the same length
    if len(observed_values) == len(lats) and len(observed_values) == len(lons):
        # Plot observed data as points
        sc = ax.scatter(
            lons, lats,
            c=observed_values,
            s=20,                          # slightly larger marker size
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=1.0,                     # full opacity for better visibility
            linewidths=0.8,                # thicker outline
            edgecolors='white',            # white outline for better contrast
            zorder=3,                      # ensure points are on top
            label='Observed Data'
        )
        
        # Create a grid for interpolation and contour plotting
        grid_resolution = 100
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        
        grid_lon = np.linspace(lon_min, lon_max, grid_resolution)
        grid_lat = np.linspace(lat_min, lat_max, grid_resolution)
        
        mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
        
        # Interpolate posterior mean values onto the grid
        points = np.column_stack((lons, lats))
        grid_mean = griddata(points, posterior_mean, (mesh_lon, mesh_lat), method='cubic')
        
        # Plot contours of the posterior mean with higher contrast
        contour = ax.contourf(
            mesh_lon, mesh_lat, grid_mean,
            levels=15,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=0.8,                     # slightly higher alpha for better visibility
            zorder=2
        )
        
        # Add contour lines for better readability
        contour_lines = ax.contour(
            mesh_lon, mesh_lat, grid_mean,
            levels=8,                      # fewer levels for clarity
            colors='white',                # white lines stand out on the colormap
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=2.5                     # above the filled contours but below points
        )
        
        # Add a colorbar with improved styling
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04,
                          fraction=0.05, extend="both")
        cbar.set_label("Temperature Values", fontsize=12, weight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Add a legend to distinguish between observed points and posterior contours
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='Observed Data (Hot)', markeredgecolor='white'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, label='Observed Data (Cool)', markeredgecolor='white'),
            Line2D([0], [0], color='red', lw=4, label='Hot Temperature Contours'),
            Line2D([0], [0], color='blue', lw=4, label='Cool Temperature Contours')
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        ax.set_title(title, fontsize=14, weight="bold")
        
        plt.tight_layout()
        return fig
    else:
        print(f"Data length mismatch: lats={len(lats)}, lons={len(lons)}, observed_values={len(observed_values)}")
        return None

def compute_loss_landscape(x, y, kernel_class, kernel_name, epsilon=1e-2, fixed_sigmasq=1.0, num_points=20):
    """
    Compute negative log marginal likelihood values over a grid of lengthscales and variances.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Training data
    kernel_class : class
        Kernel class (e.g., SquaredExponential or Matern)
    kernel_name : str
        Name of the kernel for Matern kernels
    epsilon : float
        Quadrature accuracy parameter
    fixed_sigmasq : float
        Fixed noise variance to use for all evaluations
    num_points : int
        Number of grid points along each dimension
        
    Returns:
    --------
    lengthscale_grid, variance_grid : numpy arrays
        2D grid of parameter values
    loss_values : numpy array
        2D grid of negative log marginal likelihood values
    """
    print(f"Computing loss landscape for {kernel_class.__name__}{' ' + kernel_name if kernel_name else ''} kernel on {len(x)} data points")
    
    # Calculate data bounds for gradient computation
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    
    # Create parameter grids
    dimension = x.shape[1]
    
    # Determine reasonable parameter ranges
    data_range = (x1 - x0).max().item()
    
    # Lengthscale range from 1% to 100% of data range
    lengthscales = np.logspace(-2, 0, num_points) * data_range
    
    # Variance range from 0.01 to 10 times the data variance
    data_variance = y.var().item()
    variances = np.logspace(-2, 1, num_points) * data_variance
    
    # Create 2D grids
    lengthscale_grid, variance_grid = np.meshgrid(lengthscales, variances)
    loss_values = np.zeros_like(lengthscale_grid)
    sigmasq_tensor = torch.tensor(fixed_sigmasq, dtype=torch.float64)
    
    print(f"Grid size: {num_points}x{num_points}, evaluating {num_points*num_points} combinations...")
    total_points = num_points * num_points
    success_count = 0
    error_count = 0
    
    # Show progress with tqdm if available
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        
    if use_tqdm:
        progress_iter = tqdm(range(num_points), desc="Row")
    else:
        progress_iter = range(num_points)
        
    # Evaluate NLML at each grid point
    for i in progress_iter:
        for j in range(num_points):
            try:
                # Create kernel with current hyperparameters
                if kernel_class == Matern:
                    # For Matern kernel, additional name parameter is needed
                    kernel = kernel_class(dimension=dimension, name=kernel_name, 
                                        lengthscale=float(lengthscale_grid[i, j]), 
                                        variance=float(variance_grid[i, j]))
                else:
                    kernel = kernel_class(dimension=dimension, 
                                        lengthscale=float(lengthscale_grid[i, j]), 
                                        variance=float(variance_grid[i, j]))
                
                # Use the kernel's log_marginal method to compute negative log marginal likelihood
                with torch.no_grad():
                    try:
                        # Negative log marginal likelihood (lower is better)
                        nlml = -kernel.log_marginal(x, y, sigmasq_tensor.item())
                        loss_values[i, j] = nlml.item()
                        success_count += 1
                    except RuntimeError as e:
                        # Handle case where kernel matrix is not positive definite
                        # Assign a high loss value to indicate this is a bad region
                        print(f"Warning: Failed at lengthscale={lengthscale_grid[i, j]:.3f}, variance={variance_grid[i, j]:.3f} - {str(e)}")
                        loss_values[i, j] = 1e6  # Very high loss value
                        error_count += 1
            except Exception as e:
                print(f"Unexpected error at lengthscale={lengthscale_grid[i, j]:.3f}, variance={variance_grid[i, j]:.3f} - {str(e)}")
                loss_values[i, j] = 1e6  # Very high loss value
                error_count += 1
                
            # If not using tqdm, print progress every few iterations
            if not use_tqdm and (i * num_points + j + 1) % max(1, num_points // 4) == 0:
                print(f"Progress: {i * num_points + j + 1}/{total_points} combinations evaluated")
    
    print(f"Completed: {success_count} successful evaluations, {error_count} errors out of {total_points} total")
    return lengthscale_grid, variance_grid, loss_values

def plot_loss_landscape(lengthscale_grid, variance_grid, loss_values, kernel_type, optimal_lengthscale=None, optimal_variance=None, trajectory=None):
    """
    Create contour plot of the loss landscape
    
    Parameters:
    -----------
    lengthscale_grid, variance_grid : numpy arrays
        2D grid of parameter values
    loss_values : numpy array
        2D grid of NLML values
    kernel_type : str
        Name of the kernel for the plot title
    optimal_lengthscale, optimal_variance : float, optional
        Optimal parameter values to mark on the plot
    trajectory : dict, optional
        Dictionary containing 'lengthscale' and 'variance' lists for the optimization trajectory
    """
    # Cap very high loss values to make visualization more informative
    # This handles regions where the kernel matrix was not positive definite
    median_loss = np.median(loss_values)
    loss_values_capped = np.clip(loss_values, None, median_loss * 5)
    
    # Apply a more aggressive normalization to increase contrast
    # Find the minimum and range of values after capping
    min_loss = np.min(loss_values_capped)
    range_loss = np.max(loss_values_capped) - min_loss
    
    # Normalize to highlight differences - use exponential scaling for better contrast
    normalized_loss = (loss_values_capped - min_loss) / range_loss
    log_loss = np.log1p(normalized_loss * 10)  # log1p(x) = log(1+x)
    
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Use a colormap with more contrast - 'plasma', 'inferno', or 'RdYlBu_r' are good options
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed (blue=low/good, red=high/bad)
    
    # Create contour plot with log-scaled axes
    contour = ax.contourf(
        lengthscale_grid, variance_grid, log_loss, 
        levels=20, cmap=cmap, alpha=0.9
    )
    
    # Add contour lines for better readability
    contour_lines = ax.contour(
        lengthscale_grid, variance_grid, log_loss,
        levels=8, colors='black', linewidths=0.5, alpha=0.5
    )
    
    # Add optimization trajectory if provided
    if trajectory and 'lengthscale' in trajectory and 'variance' in trajectory:
        # Extract trajectories
        ls_traj = trajectory['lengthscale']
        var_traj = trajectory['variance']
        
        # Plot trajectory as a line with points
        ax.plot(ls_traj, var_traj, 'lime', linewidth=2, alpha=0.8, zorder=4)
        
        # Add points for each iteration
        ax.scatter(
            ls_traj, var_traj, 
            c='lime', s=20, alpha=0.6, zorder=5,
            label='Optimization Path'
        )
        
        # Mark start point
        ax.scatter(
            ls_traj[0], var_traj[0],
            c='blue', s=100, marker='o', 
            edgecolors='white', linewidths=1, zorder=6,
            label='Start'
        )
    
    # Mark optimal value if provided
    if optimal_lengthscale is not None and optimal_variance is not None:
        ax.scatter(
            optimal_lengthscale, optimal_variance, 
            c='yellow', s=150, marker='*', 
            edgecolors='black', linewidths=1, zorder=6,
            label=f'Optimal (ℓ={optimal_lengthscale:.3f}, σ²={optimal_variance:.3f})'
        )
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Set log-scale for axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels and title
    ax.set_xlabel('Lengthscale (ℓ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance (σ²)', fontsize=12, fontweight='bold')
    ax.set_title(f'Negative Log Marginal Likelihood for {kernel_type} Kernel', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Log NLML (blue=better, red=worse)', fontsize=10, fontweight='bold')
    
    # Annotate the minimum value on the plot
    min_idx = np.unravel_index(np.argmin(loss_values_capped), loss_values_capped.shape)
    min_ls = lengthscale_grid[min_idx]
    min_var = variance_grid[min_idx]
    
    ax.scatter(
        min_ls, min_var,
        c='white', s=100, marker='x', 
        edgecolors='black', linewidths=2, zorder=7,
        label=f'NLML Min (ℓ={min_ls:.3f}, σ²={min_var:.3f})'
    )
    
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def train_models_with_epsilons(x, y, kernel_class, kernel_name=None, epsilon_values=None, sigmasq=5, max_iters=50):
    """
    Train models with different epsilon values.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Training data
    kernel_class : class
        Kernel class to use
    kernel_name : str, optional
        Name parameter for Matern kernel
    epsilon_values : list, optional
        List of epsilon values to use
        
    Returns:
    --------
    list of (model, trajectory) tuples
    """
    if epsilon_values is None:
        epsilon_values = [1e-2, 1e-3, 1e-4]
    
    results = []
    
    for eps in epsilon_values:
        print(f"\nTraining with epsilon={eps}")
        
        if kernel_class == Matern:
            kernel = kernel_class(dimension=x.shape[1], name=kernel_name, lengthscale=1.0, variance=1.0)
        else:
            kernel = kernel_class(dimension=x.shape[1], lengthscale=1.0, variance=1.0)
            
        model, trajectory = train_gp_model(x, y, kernel, sigmasq=sigmasq, epsilon=eps, max_iters=max_iters)
        results.append((model, trajectory, eps))
        
    return results

def get_combined_grid_ranges(trajectories, padding_factor=1.5, vertical_padding_factor=2.5):
    """
    Determine grid ranges that encompass all trajectories.
    
    Parameters:
    -----------
    trajectories : list of trajectory dicts
        Each trajectory contains 'lengthscale' and 'variance' lists
    padding_factor : float
        Factor to expand the range horizontally
    vertical_padding_factor : float
        Factor to expand the range vertically (for variance)
        
    Returns:
    --------
    lengthscales, variances : tuple of arrays
        Arrays defining the grid ranges
    """
    # Find min/max of all trajectories
    min_ls = float('inf')
    max_ls = 0
    min_var = float('inf')
    max_var = 0
    
    for traj in trajectories:
        if traj is None:
            continue
        
        ls_values = traj['lengthscale']
        var_values = traj['variance']
        
        min_ls = min(min_ls, min(ls_values))
        max_ls = max(max_ls, max(ls_values))
        min_var = min(min_var, min(var_values))
        max_var = max(max_var, max(var_values))
    
    # Apply padding
    range_ls = max_ls - min_ls
    range_var = max_var - min_var
    
    min_ls = max(min_ls - range_ls * (padding_factor - 1), 0.1)  # ensure positive
    max_ls = max_ls + range_ls * (padding_factor - 1)
    min_var = max(min_var - range_var * (padding_factor - 1), 0.1)  # ensure positive
    max_var = max_var + range_var * vertical_padding_factor  # Apply more padding vertically
    
    # Create log-spaced grid
    num_points = 10  # Reduced number of grid points (was originally higher)
    lengthscales = np.logspace(np.log10(min_ls), np.log10(max_ls), num_points)
    variances = np.logspace(np.log10(min_var), np.log10(max_var), num_points)
    
    return lengthscales, variances

def compute_loss_landscape_for_kernel(x, y, kernel_class, kernel_name, model_results, fixed_sigmasq=1.0):
    """
    Compute a single loss landscape for a kernel type, which is independent of epsilon.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Training data
    kernel_class : class
        Kernel class
    kernel_name : str
        Name of the kernel for Matern kernels
    model_results : list of (model, trajectory, epsilon) tuples
        Models and trajectories for different epsilon values
    fixed_sigmasq : float
        Fixed noise variance
        
    Returns:
    --------
    lengthscale_grid, variance_grid, loss_values
    """
    # Extract trajectories for grid range calculation
    trajectories = [traj for _, traj, _ in model_results]
    
    # Get common grid ranges with enhanced vertical padding
    lengthscales, variances = get_combined_grid_ranges(
        trajectories, 
        padding_factor=1.2,
        vertical_padding_factor=2.5
    )
    
    # Use the noise value from the middle epsilon model (typically the best one)
    middle_model = model_results[1][0]  # Second element is usually middle epsilon
    sigmasq_tensor = middle_model.sigmasq
    
    print(f"\nComputing loss landscape for {kernel_class.__name__} kernel")
    print(f"This is computed once and used for all epsilon values")
    
    # Create grid - using fewer points to make computation faster
    # Rather than creating a completely new grid, we're using a subset of the existing grid
    # We'll sample only 10x10 points instead of the full grid
    lengthscale_subset = lengthscales  # Already reduced to 10 points in get_combined_grid_ranges
    variance_subset = variances         # Already reduced to 10 points in get_combined_grid_ranges
    
    lengthscale_grid, variance_grid = np.meshgrid(lengthscale_subset, variance_subset)
    loss_values = np.zeros_like(lengthscale_grid)
    
    # Calculate loss values
    print(f"Grid size: {len(lengthscale_subset)}x{len(variance_subset)}, evaluating {len(lengthscale_subset)*len(variance_subset)} combinations...")
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        
    if use_tqdm:
        progress_iter = tqdm(range(len(variance_subset)), desc="Row")
    else:
        progress_iter = range(len(variance_subset))
    
    success_count = 0
    error_count = 0
    
    for i in progress_iter:
        for j in range(len(lengthscale_subset)):
            try:
                # Create kernel with current hyperparameters
                if kernel_class == Matern:
                    # For Matern kernel, additional name parameter is needed
                    kernel = kernel_class(dimension=x.shape[1], name=kernel_name, 
                                        lengthscale=float(lengthscale_grid[i, j]), 
                                        variance=float(variance_grid[i, j]))
                else:
                    kernel = kernel_class(dimension=x.shape[1], 
                                        lengthscale=float(lengthscale_grid[i, j]), 
                                        variance=float(variance_grid[i, j]))
                
                # Use the kernel's log_marginal method to compute negative log marginal likelihood
                with torch.no_grad():
                    try:
                        # Negative log marginal likelihood (lower is better)
                        nlml = -kernel.log_marginal(x, y, sigmasq_tensor.item())
                        loss_values[i, j] = nlml.item()
                        success_count += 1
                    except RuntimeError as e:
                        # Handle case where kernel matrix is not positive definite
                        # Assign a high loss value to indicate this is a bad region
                        loss_values[i, j] = 1e6  # Very high loss value
                        error_count += 1
            except Exception as e:
                loss_values[i, j] = 1e6  # Very high loss value
                error_count += 1
    
    print(f"Completed: {success_count} successful evaluations, {error_count} errors")
    return lengthscale_grid, variance_grid, loss_values

def plot_epsilon_subplots(kernel_type, landscape_data, model_results, figsize=(15, 5)):
    """
    Create a figure with subplots for different epsilon values, using the same loss landscape.
    
    Parameters:
    -----------
    kernel_type : str
        Name of the kernel for the plot title
    landscape_data : tuple of (lengthscale_grid, variance_grid, loss_values)
        Single loss landscape for the kernel type
    model_results : list of (model, trajectory, epsilon) tuples
        Models and trajectories for different epsilon values
        
    Returns:
    --------
    matplotlib figure
    """
    ls_grid, var_grid, loss_values = landscape_data
    
    # Create a figure
    fig = plt.figure(figsize=figsize)
    
    # Create a grid for the subplots and colorbar
    gs = plt.GridSpec(1, len(model_results) + 1, width_ratios=[1]*len(model_results) + [0.05])
    
    # Create subplots
    axes = []
    for i in range(len(model_results)):
        axes.append(fig.add_subplot(gs[0, i]))
    
    # Pre-process loss values once for better contrast
    median_loss = np.median(loss_values)
    loss_values_capped = np.clip(loss_values, None, median_loss * 5)
    
    # Apply normalization for better contrast
    min_loss = np.min(loss_values_capped)
    range_loss = np.max(loss_values_capped) - min_loss
    normalized_loss = (loss_values_capped - min_loss) / range_loss
    log_loss = np.log1p(normalized_loss * 10)
    
    # Find global minimum
    min_idx = np.unravel_index(np.argmin(loss_values_capped), loss_values_capped.shape)
    min_ls = ls_grid[min_idx]
    min_var = var_grid[min_idx]
    
    # Create color map
    cmap = plt.cm.RdYlBu_r
    levels = 15
    
    # Collect handles and labels for the legend
    all_handles = []
    all_labels = []
    
    for i, (ax, (model, trajectory, eps)) in enumerate(zip(axes, model_results)):
        # Create contour plot using the SAME processed loss values
        contour = ax.contourf(
            ls_grid, var_grid, log_loss, 
            levels=levels, cmap=cmap, alpha=0.9
        )
        
        # Add contour lines
        contour_lines = ax.contour(
            ls_grid, var_grid, log_loss,
            levels=6, colors='black', linewidths=0.5, alpha=0.5
        )
        
        # Add optimization trajectory
        if trajectory and 'lengthscale' in trajectory and 'variance' in trajectory:
            ls_traj = trajectory['lengthscale']
            var_traj = trajectory['variance']
            
            # Plot trajectory
            path_line = ax.plot(ls_traj, var_traj, 'lime', linewidth=2, alpha=0.8, zorder=4)[0]
            if i == 0:  # Only add to legend once
                all_handles.append(path_line)
                all_labels.append('Optimization Path')
            
            # Mark start and end points
            start_point = ax.scatter(
                ls_traj[0], var_traj[0],
                c='blue', s=80, marker='o', 
                edgecolors='white', linewidths=1, zorder=6
            )
            if i == 0:  # Only add to legend once
                all_handles.append(start_point)
                all_labels.append('Start')
            
            end_point = ax.scatter(
                ls_traj[-1], var_traj[-1],
                c='yellow', s=120, marker='*', 
                edgecolors='black', linewidths=1, zorder=6
            )
            if i == 0:  # Only add to legend once
                all_handles.append(end_point)
                all_labels.append('Optimal (End)')
        
        # Mark the global minimum value on all plots
        min_point = ax.scatter(
            min_ls, min_var,
            c='white', s=80, marker='x', 
            linewidths=2, zorder=7
        )
        if i == 0:  # Only add to legend once
            all_handles.append(min_point)
            all_labels.append(f'NLML Min (ℓ={min_ls:.2f}, σ²={min_var:.2f})')
        
        # Set log-scale for axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add labels
        ax.set_xlabel('Lengthscale (ℓ)', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Variance (σ²)', fontsize=11, fontweight='bold')
        
        # Add title with epsilon value
        ax.set_title(f'ε = {eps:.0e}', fontsize=12, fontweight='bold')
        
        # Add final hyperparameter values text
        final_ls = model.kernel.lengthscale
        final_var = model.kernel.variance
        final_noise = model.sigmasq.item()
        
        ax.text(
            0.05, 0.05, 
            f'Final:\nℓ = {final_ls:.2f}\nσ² = {final_var:.2f}\nnoise = {final_noise:.2f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
            fontsize=9
        )
    
    # Add overall title
    fig.suptitle(f'Negative Log Marginal Likelihood for {kernel_type} Kernel', fontsize=14, fontweight='bold')
    
    # Add colorbar on the right side of the plots
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Log NLML (blue=better, red=worse)', fontsize=10, fontweight='bold')
    
    # Add legend to the first subplot
    axes[0].legend(
        handles=all_handles, 
        labels=all_labels, 
        loc='upper right',
        fontsize=9,
        framealpha=0.9
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    return fig

def main():
    # Load the USA temperature data
    data = load_data('usa_temp_data.pt')
    x = data['x']  # Scaled coordinates
    y = data['y']  # Temperature values
    x_unscaled = data.get('x_unscaled', x)  # Original lat/lon coordinates if available
    
    print(f"Data loaded: {len(x)} points with dimensions {x.shape[1]}")
    
    # Check if the data is in the right format
    if x.dim() == 1:
        x = x.unsqueeze(1)  # Add dimension if 1D
    
    # Define epsilon values to test
    epsilon_values = [1e-2, 1e-3, 1e-4]
    
    # Train models with different epsilon values
    print("\nTraining Squared Exponential models with different epsilon values...")
    se_results = train_models_with_epsilons(
        x, y, 
        SquaredExponential, None, 
        epsilon_values=epsilon_values,
        sigmasq=5,
        max_iters=50
    )
    
    print("\nTraining Matern models with different epsilon values...")
    matern_results = train_models_with_epsilons(
        x, y, 
        Matern, 'matern32', 
        epsilon_values=epsilon_values,
        sigmasq=5,
        max_iters=50
    )
    
    # Compute loss landscapes ONCE for each kernel type
    # Using reduced grid resolution for faster computation
    se_landscape = compute_loss_landscape_for_kernel(
        x, y, 
        SquaredExponential, None, 
        se_results
    )
    
    matern_landscape = compute_loss_landscape_for_kernel(
        x, y, 
        Matern, 'matern32', 
        matern_results
    )
    
    # Plot and save subplots using the SAME landscape for all epsilon values
    se_fig = plot_epsilon_subplots(
        'Squared Exponential', 
        se_landscape, 
        se_results,
        figsize=(16, 5)
    )
    se_fig.savefig('se_epsilon_comparison.png', dpi=300, bbox_inches='tight')
    
    matern_fig = plot_epsilon_subplots(
        'Matérn-3/2', 
        matern_landscape, 
        matern_results,
        figsize=(16, 5)
    )
    matern_fig.savefig('matern_epsilon_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\nEpsilon comparison figures saved to:")
    print("- se_epsilon_comparison.png")
    print("- matern_epsilon_comparison.png")
    
    # Get predictions from the models with the smallest epsilon value (1e-4)
    # Use the last element (index -1) from se_results and matern_results
    se_model = se_results[-1][0]     # Last element is smallest epsilon (1e-4)
    matern_model = matern_results[-1][0]  # Last element is smallest epsilon (1e-4)
    
    se_mean, _ = se_model.predict(x, return_variance=False)
    matern_mean, _ = matern_model.predict(x, return_variance=False)
    
    # Create visualizations of temperature predictions
    se_fig = plot_temperature_map(
        x_unscaled, y, se_mean.numpy(), 
        f"Temperature with Squared Exponential Kernel (ε=1e-4)\n(lengthscale={se_model.kernel.lengthscale:.3f}, variance={se_model.kernel.variance:.3f})"
    )
    se_fig.savefig('temperature_squared_exponential.png', dpi=300, bbox_inches='tight')
    
    matern_fig = plot_temperature_map(
        x_unscaled, y, matern_mean.numpy(), 
        f"Temperature with Matérn-3/2 Kernel (ε=1e-4)\n(lengthscale={matern_model.kernel.lengthscale:.3f}, variance={matern_model.kernel.variance:.3f})"
    )
    matern_fig.savefig('temperature_matern.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualization saved to:")
    print("- temperature_squared_exponential.png")
    print("- temperature_matern.png")

if __name__ == "__main__":
    main() 