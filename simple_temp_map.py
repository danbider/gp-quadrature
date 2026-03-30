#!/usr/bin/env python
"""
Simple temperature data visualization - just raw data points on a map
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

def load_data(filepath):
    """Load the temperature data."""
    print(f"Loading temperature data from {filepath}")
    data = torch.load(filepath)
    return data

def plot_raw_temperature_map(x_unscaled, observed_values, title="USA Temperature Data"):
    """Create a simple map visualization of raw temperature data."""
    # Extract latitude and longitude
    lats = x_unscaled[:, 0]  # First column contains latitudes
    lons = x_unscaled[:, 1]  # Second column contains longitudes
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Create map with simple projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map extent to cover continental USA
    ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())
    
    # Add simple map features - keep all land the same color
    ax.add_feature(cfeature.LAND, facecolor='#f8f8f8')  # Light gray for all land
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')  # Light blue for ocean
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='black')
    ax.add_feature(cfeature.STATES, linewidth=0.2, color='gray')
    
    # Get value range for colormap
    min_val = observed_values.min()
    max_val = observed_values.max()
    
    print(f"Temperature range: {min_val:.1f}°F to {max_val:.1f}°F")
    print(f"Number of data points: {len(observed_values)}")
    
    # Create colormap: blue for cool, red for hot
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    # Plot temperature data points
    scatter = ax.scatter(
        lons, lats,
        c=observed_values,
        s=20,                          # marker size
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=0.9,                     # slight transparency
        linewidths=0.3,                # thin outline
        edgecolors='white',            # white outline for contrast
        zorder=5,                      # ensure points are on top
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", 
                       pad=0.05, fraction=0.06, shrink=0.8)
    cbar.set_label("Temperature (°F)", fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add title
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    
    plt.tight_layout()
    return fig

def main():
    # Load the USA temperature data
    data_path = 'data/usa_temp_data.pt'  # Adjust path as needed
    
    try:
        data = load_data(data_path)
        x = data['x']  # Scaled coordinates
        y = data['y']  # Temperature values
        x_unscaled = data.get('x_unscaled', x)  # Original lat/lon coordinates
        
        print(f"Data loaded: {len(x)} points")
        
        # Convert to numpy if needed
        if torch.is_tensor(y):
            y_np = y.cpu().numpy()
        else:
            y_np = np.array(y)
            
        if torch.is_tensor(x_unscaled):
            x_unscaled_np = x_unscaled.cpu().numpy()
        else:
            x_unscaled_np = np.array(x_unscaled)
        
        # Create the visualization
        fig = plot_raw_temperature_map(
            x_unscaled_np, y_np, 
            f"USA Temperature Data\n{len(y_np)} weather stations"
        )
        
        # Save the figure
        output_file = 'raw_temperature_map.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"\nMap saved to: {output_file}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at '{data_path}'")
        print("Please adjust the data_path variable to point to your temperature data file.")
    except Exception as e:
        print(f"Error loading or processing data: {e}")

if __name__ == "__main__":
    main() 