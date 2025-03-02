import os
import difflib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

VALID_MINIGRID_ENVS = ['MiniGrid-LavaAdm_karan-v0', 'MiniGrid-IntruderRobotRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0']#, 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0']
VALID_SYS_STR_TYP = ["QuantiativeRefinedAdmissible", "QuantitativeAdmMemorless"]
VALID_HUMAN_TYPE = ['epsilon-human', 'random-human', 'coop-human', 'mixed-human']
VALID_SYS_TYPE = ['random-sys']
EPSILON = 1 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CWD_DIRECTORY  = os.path.dirname(os.path.abspath(__file__)) # get current working directory
FILES = os.listdir(CWD_DIRECTORY) # List all files in the directory

SYS_ALIAS_DICT = {'random-sys': 'rnd',
                  'QuantiativeRefinedAdmissible': 'Ours',
                  'QuantitativeAdmMemorless': 'Adm-Mem'}


ENV_ALIAS_DICT = {VALID_HUMAN_TYPE[0]: 'Hrnd',
                  VALID_HUMAN_TYPE[1]: 'HAdv',
                  VALID_HUMAN_TYPE[2]: 'HCoop',
                  VALID_HUMAN_TYPE[3]: 'HAdv_Rnd'}

USE_ALIAS: bool = True



def find_closest_file(minigrid_env, sys_type, valid_human_type, sys_str_type):
    """
    Find the closest file name in the given directory based on the provided sys_type, valid_human_type, and sys_str_type.

    :param minigrid_env: Minigrid Env name.
    :param sys_type: The system type to match.
    :param valid_human_type: The valid human type to match.
    :param sys_str_type: The system strategy type to match.
    :return: The closest file name.
    """
    # target file_name patter - game._graph.name + "_" + strategy_type + "_" + human_type + "_" + sys_type + "_" + str(epsilon) + timestamp + ".yaml"
    # Construct the target file name pattern
    if sys_type != '':
        target_pattern = f"{minigrid_env}_DFA_game_{valid_human_type}_{sys_type}_{EPSILON}"
    else:    
        target_pattern = f"{minigrid_env}_DFA_game_{sys_str_type}_{valid_human_type}__{EPSILON}"

    # Find the closest match using difflib
    closest_match = difflib.get_close_matches(target_pattern, FILES, n=1)

    if not closest_match or len(closest_match) > 1:
        warnings.warn("[Error] Could not locate the closest file or return mroe than one yaml file names.")
        
    return closest_match[0]

def extract_costs_from_yaml(file_path):
    """
    Extract cost values from the YAML file.
    """
    # modify path to abs path
    file_path = ROOT_PATH + f"/{file_path}"
    with open(file_path, 'r') as stream:
        episdic_data: dict = yaml.load(stream, Loader=yaml.Loader)
    
    # Extract all Cost values using regex
    # cost_regex = r'Cost: (\d+)'
    costs = [int(data['Cost']) for runs, data in episdic_data.items()]
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



def plot_box_plot(costs, file_name: str, labels: str = [], fig_title: str = '') -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel('Cost')

    # data = np.array(costs)

    ax.boxplot(costs.T, labels=labels)
    ax.set_title('Default', fontsize=10)

    # plt.show(block=True)
    if fig_title != '':
        plt.title(fig_title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Main execution
if __name__ == "__main__":
    # minigrid_env = 'MiniGrid-LavaAdm_karan-v0'
    # sys_type = "random-sys"
    sys_type = ""
    valid_human_type = "coop-human"
    # sys_str_type = "QuantiativeRefinedAdmissible"
    sys_str_type = "QuantitativeAdmMemorless"
    

    # valid human types "manual", "no-human", "random-human", "epsilon-human", "coop-human", "mixed-human"
    # use "random-human" for Adv. Human
    # use "epsilon-human" with epsilon set to 1 for completely random human
    # use "coop-human" for cooperative human
    # use "mixed-human" for Adv. and Random Human

    # test extarcting file name
    # costs = []
    # yaml_files =[]
    # for env in VALID_MINIGRID_ENVS:
    #     yaml_files.append(find_closest_file(minigrid_env=env, sys_type=sys_type, valid_human_type=valid_human_type, sys_str_type=sys_str_type))

    #     # Extract costs
    #     one_env_cost = extract_costs_from_yaml(yaml_files[-1])
    #     costs.append(one_env_cost)
    #     # stack the array
    #     # costs = np.vstack((costs, one_env_cost))

    #     # Create alternative visualizations
    #     fig2 = create_alternative_visualizations(one_env_cost)
        
    #     # Save the alternative visualizations
    #     if sys_type == "":
    #         fig_name = f"cost_{valid_human_type}_{sys_str_type}_{env}.png"
    #     else:
    #         fig_name = f"cost_{valid_human_type}_{sys_type}_{env}.png"
        
    #     plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    # plot_box_plot(np.array(costs))


    # Try plotting the box plot for fixed Minigrid Env and Human type
    
    
    # for env in VALID_MINIGRID_ENVS:
    env = 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0'
    # valid_human_type = "epsilon-human"
    for human in VALID_HUMAN_TYPE:
        yaml_files =[]
        costs = []
        for st in VALID_SYS_TYPE + VALID_SYS_STR_TYP:
            yaml_files.append(find_closest_file(minigrid_env=env, sys_type=st, valid_human_type=human, sys_str_type=st))

            # Extract costs
            one_env_cost = extract_costs_from_yaml(yaml_files[-1])
            costs.append(one_env_cost)
            # stack the array
            # costs = np.vstack((costs, one_env_cost))

            # # Create alternative visualizations
            # fig2 = create_alternative_visualizations(one_env_cost)
            
            # # Save the alternative visualizations
            # if sys_type == "":
            #     fig_name = f"cost_{valid_human_type}_{sys_str_type}_{env}.png"
            # else:
            #     fig_name = f"cost_{valid_human_type}_{sys_type}_{env}.png"
            
            # plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            # if sys_type == "":
                # fig_name = f"cost_{valid_human_type}_{sys_str_type}_{env}.png"
            # else:
        if USE_ALIAS:
            human = ENV_ALIAS_DICT.get(human)
        fig_name = f"cost_{human}_{env}.png"
            
        if USE_ALIAS:
            labels = [SYS_ALIAS_DICT.get(sys_type) for sys_type in SYS_ALIAS_DICT]
            plot_box_plot(np.array(costs), file_name=fig_name, labels=labels, fig_title=env)
        else:
            plot_box_plot(np.array(costs), file_name=fig_name, labels=VALID_SYS_TYPE + VALID_SYS_STR_TYP,  fig_title=env)


    # plot_box_plot(costs)
    
    # # Create visualization
    # fig = plot_cost_boxplot(costs)
    
    # # Save the first visualization
    # plt.savefig("cost_boxplot_analysis.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)
    
    # # Create alternative visualizations
    # fig2 = create_alternative_visualizations(costs)
    
    # # Save the alternative visualizations
    # plt.savefig("cost_alternative_visualizations.png", dpi=300, bbox_inches='tight')
    
    # Display statistics in the console
    # cost_array = np.array(costs)
    # print(f"Number of runs: {len(costs)}")
    # print(f"Mean cost: {np.mean(cost_array):.2f}")
    # print(f"Median cost: {np.median(cost_array)}")
    # print(f"Min cost: {min(costs)}")
    # print(f"Max cost: {max(costs)}")
    # print(f"Q1: {np.percentile(cost_array, 25)}")
    # print(f"Q3: {np.percentile(cost_array, 75)}")
    # print(f"Frequency of Cost = 51: {list(costs).count(51)}")
    # print(f"Percentage of Costs = 51: {list(costs).count(51)/len(costs)*100:.2f}%")
    
    # print("\nVisualization saved as 'cost_boxplot_analysis.png' and 'cost_alternative_visualizations.png'")