import os
import copy
import difflib
import warnings

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from collections import defaultdict


VALID_MINIGRID_ENVS = ['MiniGrid-LavaAdm_karan-v0', 'MiniGrid-IntruderRobotRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0', \
                        'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0']
                        # 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0-NOT_CPLX']
VALID_SYS_STR_TYP = ["QuantiativeRefinedAdmissible", "QuantitativeAdmMemorless"]
VALID_HUMAN_TYPE = ['epsilon-human', 'random-human', 'coop-human', 'mixed-human']
VALID_SYS_TYPE = ['random-sys']
EPSILON = 1 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH_WAIT = os.path.dirname(os.path.abspath(__file__)) + "/wait_gw_fixed/"
PLOTS_DIR = ROOT_PATH + "/plots/"
PLOTS_DIR_WAIT = ROOT_PATH + "/plots/with_waiting/"
CWD_DIRECTORY  = os.path.dirname(os.path.abspath(__file__)) # get current working directory
FILES = os.listdir(CWD_DIRECTORY) # List all files in the directory
FILES_WAIT = os.listdir(ROOT_PATH + "/wait_gw_fixed") # List all files in the WAIT directory

# SYS_ALIAS_DICT = {'random-sys': 'rnd',
#                   'QuantiativeRefinedAdmissible': 'Ours',
#                   'QuantitativeAdmMemorless': 'Adm-Memless'}
SYS_ALIAS_DICT = {'QuantiativeRefinedAdmissible': 'Ours',
                  'QuantitativeAdmMemorless': 'Adm-Memless'}


ENV_ALIAS_DICT = {VALID_HUMAN_TYPE[2]: 'HCoop',
                  VALID_HUMAN_TYPE[0]: 'Hrnd',
                  VALID_HUMAN_TYPE[3]: 'HAdv_Rnd',
                  VALID_HUMAN_TYPE[1]: 'HAdv',
                  }

MINIGRID_NAME_ALIAS_DICT = {VALID_MINIGRID_ENVS[0]: 'IJCAI25-Lava',
                            VALID_MINIGRID_ENVS[1]: '1-Door',
                            VALID_MINIGRID_ENVS[2]: '3-Door',
                            VALID_MINIGRID_ENVS[3]: '4-Door - CPLX',
                            # VALID_MINIGRID_ENVS[4]: '4-Door - NOT CPLX', 
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
    closest_match = difflib.get_close_matches(target_pattern, FILES_WAIT, n=1)

    if not closest_match or len(closest_match) > 1:
        warnings.warn("[Error] Could not locate the closest file or return mroe than one yaml file names.")
        
    return closest_match[0]


def extract_costs_from_yaml(file_path, get_game_stats: bool = False, wait_gw: bool = False):
    """
    Extract cost values from the YAML file.
    """
    # modify path to abs path
    if wait_gw:
        file_path = ROOT_PATH_WAIT + f"/{file_path}"
    else :
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
    # Append all costs
    # for runs, data in episdic_data.items():
    #     if data['status'] in ['Win', 'pen']:
    #         costs.append(int(data['Cost']))
    #     else:
    #         costs.append(51)
    for runs, data in episdic_data.items():
        if data['status'] == 'Win':
            costs.append(int(data['Cost']))
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


def print_stats(costs: List[int]) -> Dict[str, Union[int, float]]:
    # Display statistics in the console
    cost_array = np.array(costs)
    stats_dict = {
        "runs": len(costs),
        "mean": np.mean(cost_array),
        "median": np.median(cost_array),
        "min": min(costs),
        "max": max(costs),
        "q1": np.percentile(cost_array, 25),
        "q3": np.percentile(cost_array, 75),
        "freq_51": list(costs).count(51),
        "pct_51": (list(costs).count(51) / len(costs)) * 100
    }

    print(f"Number of runs: {len(costs)}")
    print(f"Mean cost: {np.mean(cost_array):.2f}")
    print(f"Median cost: {np.median(cost_array)}")
    print(f"Min cost: {min(costs)}")
    print(f"Max cost: {max(costs)}")
    print(f"Q1: {np.percentile(cost_array, 25)}")
    print(f"Q3: {np.percentile(cost_array, 75)}")
    print(f"Frequency of Cost = 51: {list(costs).count(51)}")
    print(f"Percentage of Costs = 51: {list(costs).count(51)/len(costs)*100:.2f}%")

    return stats_dict

def plot_mean_dict_with_bars(mean_dict):
    """
    Plot the mean_dict dictionary as bar charts.
    Each human type will have 8 bars (4 environments × 2 system types).
    Different colors will be used for different environments, and hatching patterns
    will distinguish between the two system strategies.
    """
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import AutoMinorLocator

    # Enable TeX rendering
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    # Both these dictionary are sued to matching the naming convention of the environments and system strategies in the paper.
    LEGENDS_DICT = {'IJCAI25-Lava': r'$\mathbb{E}_1$',
                    '1-Door': r'$\mathbb{E}_2$',
                    '3-Door': r'$\mathbb{E}_3$',
                    '4-Door - CPLX': r'$\mathbb{E}_4$',
                    '4-Door - NOT CPLX': r'$\mathbb{E}_5$'
                    }
    
    HUMAN_LABEL_DICT = {'HCoop': r'$\mathbf{Co-Op}$',
                        'Hrnd': r'$\mathbf{Rand}$' ,
                        'HAdv_Rnd': r'$\mathbf{WCO}-\mathbf{Rand}$',
                        'HAdv': r'$\mathbf{WCO}$'}
    
    SYS_LABEL_DICT = {'Ours': r'$\mathbf{Adm-Rat}$',
                      'Adm-Memless': r'$\mathbf{Adm}$- Memless'}
    
    human_types = list(mean_dict.keys())
    env_types = list(MINIGRID_NAME_ALIAS_DICT.values())
    sys_str_types = list(SYS_ALIAS_DICT.values())

    skip_human_types = ['HAdv']  # e.g., ['HAdv'] to skip the HAdv human type


    human_types = [h for h in list(mean_dict.keys()) if h not in skip_human_types]
    # env_types = [e for e in list(MINIGRID_NAME_ALIAS_DICT.values()) if e not in skip_env_types]
    # sys_str_types = [s for s in list(SYS_ALIAS_DICT.values()) if s not in skip_sys_types]
    
    # Colors for different environments (using a color-blind friendly palette)
    env_colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3']
    
    # Hatching patterns for different system strategies
    sys_hatches = ['', '///']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Width of each bar
    bar_width = 0.09
    
    # Calculate positions for bars
    group_width = len(env_types) * len(sys_str_types) * bar_width + bar_width  # Width of one human type group
    group_positions = np.arange(len(human_types)) * (group_width + 0.2)  # Add space between human type groups
    
    # For legend
    legend_handles_env = []
    legend_handles_sys = []
    
    # Keep track of tick positions for x-axis
    tick_positions = []
    
    # Plot bars
    for human_idx, human in enumerate(human_types):
        tick_positions.append(group_positions[human_idx] + group_width/2 - bar_width/2)
        
        for env_idx, env in enumerate(env_types):
            for sys_idx, sys_str in enumerate(sys_str_types):
                if human in mean_dict and env in mean_dict[human] and sys_str in mean_dict[human][env]:
                    # skip WCO 
                    # if human == 'HAdv':
                    #     continue
                    # Calculate bar position
                    bar_position = group_positions[human_idx] + (env_idx * len(sys_str_types) + sys_idx) * bar_width
                    
                    # Plot the bar
                    bar = ax.bar(bar_position, mean_dict[human][env][sys_str], 
                                 width=bar_width, 
                                 color=env_colors[env_idx], 
                                 hatch=sys_hatches[sys_idx],
                                 edgecolor='black',
                                 linewidth=0.5)
                    
                    # Create legend handles (only once)
                    if human_idx == 0:
                        if sys_idx == 0:
                            # Add environment to legend
                            legend_handles_env.append(mpatches.Patch(
                                color=env_colors[env_idx], 
                                # label=f'Env: {LEGENDS_DICT[env]}'
                                label=f'{LEGENDS_DICT[env]}'
                            ))
                        if env_idx == 0:
                            # Add system strategy to legend
                            legend_handles_sys.append(mpatches.Patch(
                                # facecolor='lightgray' if sys_idx == 0 else 'lightgray',
                                facecolor='white' if sys_idx == 0 else 'white',
                                hatch=sys_hatches[sys_idx],
                                edgecolor='black',
                                # label=f'System: {sys_str}'
                                label=f'{SYS_LABEL_DICT[sys_str]}'
                            ))
    
    # Set x-tick positions and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([HUMAN_LABEL_DICT[_human] for _human in human_types])  ### all labels
    # ax.set_xticklabels([HUMAN_LABEL_DICT[_human] for _human in human_types if _human != 'HAdv'])
    
    # Better labels and title
    # ax.set_xlabel('Human Type (Increasing Difficulty →)', fontsize=24, labelpad=10)
    ax.set_ylabel('Mean Payoff (Lower is better)', fontsize=24, labelpad=10)
    # ax.set_title('Performance Deterioration Across Human Types and Environments', fontsize=14, pad=20)
    
    # Increase tick label font sizes, with xticks larger than yticks
    ax.tick_params(axis='x', which='major', labelsize=20, pad=8)  # Larger x-tick labels
    ax.tick_params(axis='y', which='major', labelsize=20)

    # Add grid for better readability (only horizontal)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add minor ticks for y-axis for better readability
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Create a clearer legend with two parts
    # First legend for environments
    env_legend = ax.legend(handles=legend_handles_env, loc='upper left', 
                           bbox_to_anchor=(0.01, 0.99),
                        #    title='Environment Types',
                        #    title_fontsize=20,
                           fontsize=20)
    # Add the first legend manually
    ax.add_artist(env_legend)
    # Second legend for system strategies, placed below the first one
    ax.legend(handles=legend_handles_sys, loc='upper left', 
              bbox_to_anchor=(0.12, 0.99),
            #   title='System Strategies',
            #   title_fontsize=20,
              fontsize=20)
    
    # Add an annotation about performance deterioration
    # mid_x = (tick_positions[0] + tick_positions[-1]) / 2
    # arrow_y = ax.get_ylim()[1] * 0.8
    # ax.annotate('Performance Deterioration', 
    #             xy=(tick_positions[-1], arrow_y), xytext=(mid_x, arrow_y),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #             fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR_WAIT + 'mean_cost_bar_chart_no_Hadv', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# def plot_mean_dict_2():
def plot_mean_dict_2(mean_dict):
    """
    Plot the mean_dict dictionary with trend lines showing performance deterioration.
    Each human type will be on the x-axis, and the y-axis will represent the mean costs.
    Different markers will be used for different environments, and blue and red colors will be used to distinguish
    between the two system strategies for the same environment and human type.
    """
    human_types = list(mean_dict.keys())
    env_types = list(MINIGRID_NAME_ALIAS_DICT.values())
    sys_str_types = list(SYS_ALIAS_DICT.values())

    markers = ['o', '^', 's', 'D']  # Different markers for different environments
    colors = ['blue', 'red']  # Blue and red colors for the two system strategies
    
    # Create a bigger figure for better readability
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track points for trend lines
    trend_lines = {env: {sys_str: {'x': [], 'y': []} for sys_str in sys_str_types} for env in env_types}
    
    # Plot scatter points first
    for env_idx, env in enumerate(env_types):
        for sys_idx, sys_str in enumerate(sys_str_types):
            for human_idx, human in enumerate(human_types):
                if human in mean_dict and env in mean_dict[human] and sys_str in mean_dict[human][env]:
                    # Store points for trend lines
                    trend_lines[env][sys_str]['x'].append(human_idx)
                    trend_lines[env][sys_str]['y'].append(mean_dict[human][env][sys_str])
                    
                    # Plot scatter point
                    color = colors[sys_idx % 2]
                    marker = markers[env_idx % len(markers)]
                    label = f'{env} - {sys_str}' if human_idx == 0 else None  # Only add to legend once
                    ax.scatter(human_idx, mean_dict[human][env][sys_str], color=color, marker=marker, s=80, label=label)
    
    # Now add trend lines for each environment and system strategy
    for env_idx, env in enumerate(env_types):
        for sys_idx, sys_str in enumerate(sys_str_types):
            if trend_lines[env][sys_str]['x']:  # Check if we have points for this combination
                color = colors[sys_idx % 2]
                # Add trend line with lower opacity
                ax.plot(trend_lines[env][sys_str]['x'], trend_lines[env][sys_str]['y'], 
                        color=color, linestyle='--', alpha=0.4)
    
    # # Calculate and plot average trend lines
    # for sys_idx, sys_str in enumerate(sys_str_types):
    #     avg_y_by_human = []
    #     for human_idx, human in enumerate(human_types):
    #         values = []
    #         for env in env_types:
    #             if human in mean_dict and env in mean_dict[human] and sys_str in mean_dict[human][env]:
    #                 values.append(mean_dict[human][env][sys_str])
    #         if values:
    #             avg_y_by_human.append(sum(values) / len(values))
    #         else:
    #             avg_y_by_human.append(None)
        
    #     # Filter out None values for plotting
    #     valid_indices = [i for i, v in enumerate(avg_y_by_human) if v is not None]
    #     valid_values = [avg_y_by_human[i] for i in valid_indices]
        
    #     if valid_indices:  # Only plot if we have valid values
    #         color = colors[sys_idx % 2]
    #         ax.plot(valid_indices, valid_values, color=color, linewidth=2.5, alpha=0.7,
    #                 label=f'Avg. {sys_str} Trend')
    
    # Set x-tick positions and labels
    ax.set_xticks(range(len(human_types)))
    ax.set_xticklabels(human_types)
    
    # Better labels and title
    ax.set_xlabel('Human Type (Increasing Difficulty →)', fontsize=12, labelpad=10)
    ax.set_ylabel('Mean Cost (Higher = Worse Performance)', fontsize=12, labelpad=10)
    ax.set_title('Performance Deterioration Across Human Types and Environments', fontsize=14, pad=20)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Create a better legend with grouped items
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
              bbox_to_anchor=(1.01, 1), title='System & Environment')
    
    # Annotate to emphasize performance deterioration
    midpoint_x = len(human_types) // 2 - 0.5
    min_y = min([min(trend_lines[env][sys_str]['y']) 
                for env in env_types 
                for sys_str in sys_str_types 
                if trend_lines[env][sys_str]['y']], default=0)
    
    # # Add an arrow showing deterioration trend
    # arrow_y = min_y * 1.2
    # ax.annotate('Performance Deterioration', 
    #             xy=(midpoint_x, arrow_y), xytext=(midpoint_x, arrow_y - 10),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #             fontsize=12, ha='center')
    
    # Add explanation for environment markers
    for env_idx, env in enumerate(env_types):
        marker = markers[env_idx % len(markers)]
        ax.scatter([], [], color='gray', marker=marker, s=80, label=f'Env: {env}')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR_WAIT + 'mean_cost_dist', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_mean_dict(mean_dict):
    """
    Plot the mean_dict dictionary with the specified requirements.
    Each human type will be on the x-axis, and the y-axis will represent the mean costs.
    Different markers will be used for different environments, and blue and red colors will be used to distinguish
    between the two system strategies for the same environment and human type.
    """
    human_types = list(mean_dict.keys())
    env_types = list(MINIGRID_NAME_ALIAS_DICT.values())
    sys_str_types = list(SYS_ALIAS_DICT.values())

    markers = ['o', 's', 'D', '^']  # Different markers for different environments
    colors = ['blue', 'red']  # Blue and red colors for the two system strategies

    fig, ax = plt.subplots(figsize=(12, 8))

    for env_idx, env in enumerate(env_types):
        for sys_idx, sys_str in enumerate(sys_str_types):
            x = []
            y = []
            for human in human_types:
                if human in mean_dict and env in mean_dict[human] and sys_str in mean_dict[human][env]:
                    x.append(human)
                    y.append(mean_dict[human][env][sys_str])
                    color = colors[sys_idx % 2]
                    marker = markers[env_idx % len(markers)]
                    ax.scatter(human, mean_dict[human][env][sys_str], color=color, marker=marker, label=f'{env} - {sys_str}' if sys_idx == 0 else "")

    ax.set_xlabel('Human Type')
    ax.set_ylabel('Mean Cost')
    ax.set_title('Mean Costs for Different Human Types and Environments')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(PLOTS_DIR_WAIT + 'mean_cost_dist', dpi=300, bbox_inches='tight')
    # plt.savefig(PLOTS_DIR + file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_box_plot(costs, file_name: str, labels: str = [], fig_title: str = '') -> None:
    """
     Parent box plot method creates the figure handles and calls the draw_box_plot() method to plot the box plot on the same figure. 
     This modular approach allows or box plot to have differet sample and yet plot it on the same canvas.
    """
    fig, ax = plt.subplots()
    ax.set_ylabel('Cost')
    num_boxes: int = len(costs)

    # for idx, ax, cost in enumerate(zip(axs, costs)):
    #     bplot = draw_box_plot(ax=ax, samples=np.array(cost), label=labels[idx])
    bplot = plt.boxplot(positions=list(range(num_boxes)),
                        labels=labels,
                        x=[np.array(cost) for cost in costs],
                        showmeans=True,
                        patch_artist=True)
    
    # add # of samples on top of each box plot
    range(len(costs))
    upper_labels = [len(data) for data in costs]
    pos = np.arange(num_boxes)
    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        # k = tick % 2
        ax.text(pos[tick], 1, upper_labels[tick],
                transform=ax.get_xaxis_transform(),
                horizontalalignment='center', size='x-small')
                # weight=weights[k], color=box_colors[k])

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
    plt.savefig(PLOTS_DIR_WAIT + file_name, dpi=300, bbox_inches='tight')
    # plt.savefig(PLOTS_DIR + file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def convert_defaultdict_to_dict(d):
    """
    Recursively convert a defaultdict to a regular dictionary.
    """
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


# Main execution
if __name__ == "__main__":
    # Try plotting the box plot for fixed Minigrid Env and Human type
    # env = 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0'
    # valid_human_type = "epsilon-human"
    MAX_COST_VAL = 51
    mean_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    for htype in ENV_ALIAS_DICT.values():
        for env_gridworld in VALID_MINIGRID_ENVS:
            for stype in VALID_SYS_STR_TYP:
                mean_dict[htype][MINIGRID_NAME_ALIAS_DICT.get(env_gridworld)][SYS_ALIAS_DICT[stype]] = MAX_COST_VAL
    
    for env in VALID_MINIGRID_ENVS:
        # if env != 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0':
        #     continue
        for human in VALID_HUMAN_TYPE:
            yaml_files =[]
            costs = []
            # for st in VALID_SYS_TYPE + VALID_SYS_STR_TYP:
            for st in VALID_SYS_STR_TYP:
                yaml_files.append(find_closest_file(minigrid_env=env, sys_type=st, valid_human_type=human, sys_str_type=st))

                # Extract costs
                one_env_cost = extract_costs_from_yaml(yaml_files[-1], wait_gw=True)
                if len(one_env_cost) > 0:
                    costs.append(one_env_cost)

                    if DEBUG:
                        print("***************************************************************************************************")
                        print(f"{env} - {human} - {st}")
                        stats_dict = print_stats(one_env_cost)
                        print("***************************************************************************************************")
                        mean_dict[ENV_ALIAS_DICT.get(human)][MINIGRID_NAME_ALIAS_DICT.get(env)][SYS_ALIAS_DICT[st]] = float(stats_dict['mean'])

            if USE_ALIAS:
                human = ENV_ALIAS_DICT.get(human)
            fig_name = f"cost_{human}_{env}.png"

            if len(costs) > 0:    
                if USE_ALIAS:
                    labels = [SYS_ALIAS_DICT.get(sys_type) for sys_type in VALID_SYS_STR_TYP]
                    env_alias = MINIGRID_NAME_ALIAS_DICT.get(env)
                    plot_box_plot(costs, file_name=fig_name, labels=labels, fig_title=env_alias + "-" + human)
                else:
                    plot_box_plot(costs, file_name=fig_name, labels=VALID_SYS_TYPE + VALID_SYS_STR_TYP,  fig_title=env)

    # for Adv huam env 
    # dump the dictionary to a yaml file
    # mean_dict = convert_defaultdict_to_dict(mean_dict)
    # with open('mean_dict_gw_wait.yaml', 'w') as file:
    #     yaml.dump(mean_dict, file)
    
    
    # load yaml dictionary
    # with open('mean_dict_gw_wait.yaml', 'r') as file: 
    #     mean_dict = yaml.load(file, Loader=yaml.Loader) 

    # plot_mean_dict(mean_dict)
    # plot_mean_dict_2(mean_dict)
    # plot_mean_dict_with_bars(mean_dict)

    #### TESTING plotting for single file
    # file_name = "MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0_DFA_game_coop-human_random-sys_120250309_051833.yaml"
    # costs = extract_costs_from_yaml(file_name, get_game_stats=True, wait_gw=False)
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
    
    