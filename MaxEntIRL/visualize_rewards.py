import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from irl_config import FeatureExtractionConfig

if __name__ == "__main__":
    # load the rewards from a file and visualize them
    reward_path = os.path.join(FeatureExtractionConfig().output_dir, "weights", "boston_99.pkl")
    if not os.path.exists(reward_path):
        raise FileNotFoundError(f"Reward file not found at {reward_path}")
    
    with open(reward_path, "rb") as f:
        rewards = pickle.load(f)
        theta = rewards['theta']

    # Get feature names from config
    feature_names = FeatureExtractionConfig().feature_names
    
    # Create DataFrame for bar plot
    df = pd.DataFrame({
        'feature': feature_names,
        'theta': theta
    })
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better styling
    ax = sns.barplot(data=df, x='feature', y='theta', palette='viridis')
    
    # Customize the plot
    plt.title("Final Theta Values (Reward Weights) by Feature", fontsize=16, fontweight='bold')
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Theta Value", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (feature, value) in enumerate(zip(feature_names, theta)):
        ax.text(i, value + (max(theta) - min(theta)) * 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(FeatureExtractionConfig().output_dir, "final_theta_barplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create a horizontal bar plot (alternative visualization)
    plt.figure(figsize=(10, 8))
    
    # Sort by theta value for better visualization
    df_sorted = df.sort_values('theta', ascending=True)
    
    ax2 = sns.barplot(data=df_sorted, y='feature', x='theta', palette='viridis')
    plt.title("Final Theta Values (Sorted)", fontsize=16, fontweight='bold')
    plt.xlabel("Theta Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    
    # Add value labels
    for i, (feature, value) in enumerate(zip(df_sorted['feature'], df_sorted['theta'])):
        ax2.text(value + (max(theta) - min(theta)) * 0.01, i, f'{value:.3f}', 
                va='center', ha='left', fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save horizontal version
    output_path_h = os.path.join(FeatureExtractionConfig().output_dir, "final_theta_horizontal_barplot.png")
    plt.savefig(output_path_h, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final theta values
    print("\nFinal Theta Values:")
    print("=" * 50)
    for feature, value in zip(feature_names, theta):
        print(f"{feature:20}: {value:8.4f}")
    
    print(f"\nHighest weighted feature: {feature_names[np.argmax(theta)]} ({max(theta):.4f})")
    print(f"Lowest weighted feature:  {feature_names[np.argmin(theta)]} ({min(theta):.4f})")
    print(f"Theta magnitude (L2 norm): {np.linalg.norm(theta):.4f}")
