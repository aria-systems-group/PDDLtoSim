import os
import copy
import difflib
import warnings

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict


VALID_MINIGRID_ENVS = ['MiniGrid-LavaAdm_karan-v0', 'MiniGrid-IntruderRobotRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0', \
                        'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0', 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0-NOT_CPLX']
VALID_SYS_STR_TYP = ["QuantiativeRefinedAdmissible", "QuantitativeAdmMemorless"]
VALID_HUMAN_TYPE = ['epsilon-human', 'random-human', 'coop-human', 'mixed-human']
VALID_SYS_TYPE = ['random-sys']
EPSILON = 1 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = ROOT_PATH + "/plots/"
CWD_DIRECTORY  = os.path.dirname(os.path.abspath(__file__)) # get current working directory
FILES = os.listdir(CWD_DIRECTORY) # List all files in the directory

SYS_ALIAS_DICT = {'random-sys': 'rnd',
                  'QuantiativeRefinedAdmissible': 'Ours',
                  'QuantitativeAdmMemorless': 'Adm-Mem'}


ENV_ALIAS_DICT = {VALID_HUMAN_TYPE[0]: 'Hrnd',
                  VALID_HUMAN_TYPE[1]: 'HAdv',
                  VALID_HUMAN_TYPE[2]: 'HCoop',
                  VALID_HUMAN_TYPE[3]: 'HAdv_Rnd'}

MINIGRID_NAME_ALIAS_DICT = {VALID_MINIGRID_ENVS[0]: 'IJCAI25-Lava',
                            VALID_MINIGRID_ENVS[1]: '1-Door',
                            VALID_MINIGRID_ENVS[2]: '3-Door',
                            VALID_MINIGRID_ENVS[3]: '4-Door - CPLX',
                            VALID_MINIGRID_ENVS[4]: '4-Door - NOT CPLX', 
                            }

USE_ALIAS: bool = True

DEBUG: bool = True



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
    name: str = copy.copy(minigrid_env)
    if 'NOT_CPLX' in name:
        name = name.replace('-NOT_CPLX', '')
        minigrid_game = f"{name}_DFA_GAME_NOT_CPLX"
    else:
        minigrid_game = f"{name}_DFA_GAME"

    if sys_type == 'random-sys':
        target_pattern = f"{minigrid_game}_{valid_human_type}_{sys_type}_{EPSILON}"
    else:    
        target_pattern = f"{minigrid_game}_{sys_str_type}_{valid_human_type}__{EPSILON}"

    # Find the closest match using difflib
    closest_match = difflib.get_close_matches(target_pattern, FILES, n=1)

    if not closest_match or len(closest_match) > 1:
        warnings.warn("[Error] Could not locate the closest file or return mroe than one yaml file names.")
        
    return closest_match[0]


def extract_costs_from_yaml(file_path, get_game_stats: bool = False):
    """
    Extract cost values from the YAML file.
    """
    # modify path to abs path
    file_path = ROOT_PATH + f"/{file_path}"
    with open(file_path, 'r') as stream:
        episdic_data: dict = yaml.load(stream, Loader=yaml.Loader)
    
    #print extracted data
    if get_game_stats:
        cwin = cpen = clos = 0
        for runs, data in episdic_data.items():
            if data['status'] == "Win":
                cwin += 1
            elif data['status'] == "pen":
                cpen += 1
            elif data['status'] == "los":
                clos += 1
        assert cwin + cpen + clos == len(episdic_data.keys()), \
        "[Error] Sum of win, pen, and los is not equal to total number of runs. There is an issue with either rolling out or dumping."
        print(f"Win: {cwin} | Pen: {cpen} | Los: {clos}")

    # Extract all Cost values using regex
    # cost_regex = r'Cost: (\d+)'
    costs: List[int] = []
    for runs, data in episdic_data.items():
        if data['status'] in ['Win', 'pen']:
            costs.append(int(data['Cost']))
        else:
            costs.append(51)
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


def print_stats(costs: List[int]) -> None:
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
    
    # print("\nVisualization saved as 'cost_boxplot_analysis.png' and 'cost_alternative_visualizations.png'")



def plot_box_plot(costs, file_name: str, labels: str = [], fig_title: str = '') -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel('Cost')

    # data = np.array(costs)

    bplot: Dict = ax.boxplot(costs.T, labels=labels, showmeans=True, patch_artist=True)
    # colors = sorted(mcolors.CSS4_COLORS.keys()) for full color palette.
    COLORS = ['lightblue', 'lightgreen', 'mistyrose']
    # fill with colors
    for patch, color in zip(bplot['boxes'], COLORS):
        patch.set_facecolor(color)

    # color the boxplots
    ax.set_title('Default', fontsize=10)

    # plt.show(block=True)
    if fig_title != '':
        plt.title(fig_title)
    plt.savefig(PLOTS_DIR + file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Main execution
if __name__ == "__main__":
    # Try plotting the box plot for fixed Minigrid Env and Human type
    # env = 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0'
    # valid_human_type = "epsilon-human"
    for env in VALID_MINIGRID_ENVS:
        # if env != 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0':
        #     continue
        for human in VALID_HUMAN_TYPE:
            yaml_files =[]
            costs = []
            for st in VALID_SYS_TYPE + VALID_SYS_STR_TYP:
                yaml_files.append(find_closest_file(minigrid_env=env, sys_type=st, valid_human_type=human, sys_str_type=st))

                # Extract costs
                one_env_cost = extract_costs_from_yaml(yaml_files[-1])
                costs.append(one_env_cost)

                if DEBUG:
                    print("***************************************************************************************************")
                    print(f"{env} - {human} - {st}")
                    print_stats(one_env_cost)
                    print("***************************************************************************************************")

            if USE_ALIAS:
                human = ENV_ALIAS_DICT.get(human)
            fig_name = f"cost_{human}_{env}.png"
                
            if USE_ALIAS:
                labels = [SYS_ALIAS_DICT.get(sys_type) for sys_type in SYS_ALIAS_DICT]
                env_alias = MINIGRID_NAME_ALIAS_DICT.get(env)
                plot_box_plot(np.array(costs), file_name=fig_name, labels=labels, fig_title=env_alias + "-" + human)
            else:
                plot_box_plot(np.array(costs), file_name=fig_name, labels=VALID_SYS_TYPE + VALID_SYS_STR_TYP,  fig_title=env)

    #### TESTING plotting for single file
    # file_name = "MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0_DFA_game_NOT_CPLX_QuantitativeAdmMemorless_epsilon-human__120250302_032641.yaml"
    # costs = extract_costs_from_yaml(file_name, get_game_stats=True)
    # plot_box_plot(np.array(costs), file_name='testing', labels=['Adm-Mem'], fig_title='4-Door')
    
    # Create visualization
    # fig = plot_cost_boxplot(costs)
    
    # Save the first visualization
    # plt.savefig("cost_boxplot_analysis.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)
    
    # Create alternative visualizations
    # fig2 = create_alternative_visualizations(costs)
    
    # Save the alternative visualizations
    # plt.savefig("cost_alternative_visualizations.png", dpi=300, bbox_inches='tight')
    
    