import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import io

def extract_costs_from_yaml(file_path):
    """
    Extract cost values from the YAML file.
    """
    with open(file_path, 'r') as f:
        yaml_content = f.read()
    
    # Extract all Cost values using regex
    cost_regex = r'Cost: (\d+)'
    costs = [int(match) for match in re.findall(cost_regex, yaml_content)]
    return costs

def plot_cost_boxplot(costs):
    """
    Create a detailed box plot visualization of the costs with additional statistics.
    """
    # Convert to numpy array to avoid pandas indexing issues
    costs_array = np.array(costs)
    
    # Calculate statistics
    mean_cost = np.mean(costs_array)
    median_cost = np.median(costs_array)
    min_cost = min(costs_array)
    max_cost = max(costs_array)
    q1 = np.percentile(costs_array, 25)
    q3 = np.percentile(costs_array, 75)
    
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Create subplot layout (1 row, 2 columns)
    plt.subplot(1, 2, 1)
    
    # Create the box plot with seaborn for better appearance
    sns.boxplot(y=costs_array, color='skyblue', width=0.3)
    
    # Add a swarm plot to show individual data points
    sns.swarmplot(y=costs_array, color='darkblue', alpha=0.7, size=4)
    
    # Add a line for the mean
    plt.axhline(y=mean_cost, color='red', linestyle='-', alpha=0.7, label=f'Mean: {mean_cost:.2f}')
    
    # Add labels and title
    plt.title('Box Plot of Cost Values', fontsize=14)
    plt.ylabel('Cost', fontsize=12)
    plt.ylim(-1, max_cost + 5)  # Add some padding above the max value
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Create histogram on the right side
    plt.subplot(1, 2, 2)
    
    # Create histogram with KDE
    # Convert to numpy array to avoid pandas issues
    # sns.histplot(x=costs_array, kde=True, color='skyblue', bins=10, edgecolor='black')
    
    # Add vertical lines for key statistics
    plt.axvline(x=mean_cost, color='red', linestyle='-', label=f'Mean: {mean_cost:.2f}')
    plt.axvline(x=median_cost, color='green', linestyle='--', label=f'Median: {median_cost}')
    plt.axvline(x=q1, color='purple', linestyle=':', label=f'Q1: {q1}')
    plt.axvline(x=q3, color='purple', linestyle=':', label=f'Q3: {q3}')
    
    # Add labels and title
    plt.title('Distribution of Cost Values', fontsize=14)
    plt.xlabel('Cost', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Add a title for the entire figure
    plt.suptitle('Cost Analysis from YAML Data', fontsize=16, y=0.98)
    
    # Add text box with statistics
    stats_text = (
        f"Statistics:\n"
        f"Count: {len(costs_array)}\n"
        f"Mean: {mean_cost:.2f}\n"
        f"Median: {median_cost}\n"
        f"Min: {min_cost}\n"
        f"Max: {max_cost}\n"
        f"Q1: {q1}\n"
        f"Q3: {q3}\n"
        f"IQR: {q3 - q1}"
    )
    
    plt.figtext(0.92, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.8), 
               fontsize=10, ha='left', va='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return plt.gcf()

def create_alternative_visualizations(costs):
    """
    Create alternative visualizations that might be useful
    """
    costs_array = np.array(costs)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'Cost': costs_array})
    
    # Set up figure
    plt.figure(figsize=(15, 10))
    
    # 1. Box Plot
    plt.subplot(2, 2, 1)
    sns.boxplot(x=costs_array, color='lightblue')
    plt.title('Box Plot (Horizontal)')
    plt.xlabel('Cost')
    
    # 2. Violin Plot
    plt.subplot(2, 2, 2)
    sns.violinplot(x=costs_array, color='lightgreen')
    plt.title('Violin Plot')
    plt.xlabel('Cost')
    
    # 3. Strip Plot
    plt.subplot(2, 2, 3)
    sns.stripplot(x=costs_array, jitter=True, alpha=0.5, color='darkblue')
    plt.axvline(x=np.mean(costs_array), color='red', linestyle='-', label=f'Mean: {np.mean(costs_array):.2f}')
    plt.axvline(x=np.median(costs_array), color='green', linestyle='--', label=f'Median: {np.median(costs_array)}')
    plt.title('Strip Plot with Mean and Median')
    plt.xlabel('Cost')
    plt.legend()
    
    # 4. ECDF (Empirical Cumulative Distribution Function)
    plt.subplot(2, 2, 4)
    sns.ecdfplot(costs_array)
    plt.title('Empirical Cumulative Distribution Function')
    plt.xlabel('Cost')
    plt.ylabel('Proportion')
    
    plt.suptitle('Alternative Visualizations of Cost Data', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return plt.gcf()

# Main execution
if __name__ == "__main__":
    # Replace with your file path
    import os
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    file_path = ROOT_PATH + "/MiniGrid-LavaAdm_karan-v0DFA-game_QuantiativeRefinedAdmissible_random-human_random-sys_1.yaml"
    
    # Extract costs
    costs = extract_costs_from_yaml(file_path)
    
    # Create visualization
    fig = plot_cost_boxplot(costs)
    
    # Save the first visualization
    plt.savefig("cost_boxplot_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create alternative visualizations
    fig2 = create_alternative_visualizations(costs)
    
    # Save the alternative visualizations
    plt.savefig("cost_alternative_visualizations.png", dpi=300, bbox_inches='tight')
    
    # Display statistics in the console
    cost_array = np.array(costs)
    print(f"Number of runs: {len(costs)}")
    print(f"Mean cost: {np.mean(cost_array):.2f}")
    print(f"Median cost: {np.median(cost_array)}")
    print(f"Min cost: {min(costs)}")
    print(f"Max cost: {max(costs)}")
    print(f"Q1: {np.percentile(cost_array, 25)}")
    print(f"Q3: {np.percentile(cost_array, 75)}")
    print(f"Frequency of Cost = 51: {list(costs).count(51)}")
    print(f"Percentage of Costs = 51: {list(costs).count(51)/len(costs)*100:.2f}%")
    
    print("\nVisualization saved as 'cost_boxplot_analysis.png' and 'cost_alternative_visualizations.png'")