import os
import copy
import math
import difflib
import warnings

import yaml
import numpy as np

from abc import ABC
from typing import List, Dict, Union, Optional, Tuple, Iterable
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator


VALID_MINIGRID_ENVS = ['MiniGrid-LavaAdm_karan-v0', 'MiniGrid-IntruderRobotRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0', \
                        'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0',
                        'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0_NOT_CPLX', #]
                        'ARCH_problem_arch', 'ARCH_problem_two_arch']
VALID_SYS_STR_TYP = ["QuantiativeRefinedAdmissible", "QuantitativeAdmMemorless"]
VALID_HUMAN_TYPE = ['epsilon-human', 'random-human', 'coop-human', 'mixed-human']
VALID_SYS_TYPE = ['random-sys']
EPSILON = 1 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH_WAIT = os.path.dirname(os.path.abspath(__file__)) + "/wait_gw_fixed/"
PLOTS_DIR = ROOT_PATH + "/plots/"
PLOTS_DIR_WAIT = ROOT_PATH + "/plots/with_waiting/"
CWD_DIRECTORY  = os.path.dirname(os.path.abspath(__file__)) # get current working directory
FILES = os.listdir(ROOT_PATH) # List all files in the directory
FILES_WAIT = os.listdir(ROOT_PATH + "/wait_gw_fixed") # List all files in the WAIT directory
FILES_SYNTH_WAIT = os.listdir(ROOT_PATH + "/construction_and_synthesis_data") # List all files in the WAIT directory

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
                            VALID_MINIGRID_ENVS[4]: '4-Door - NOT CPLX', 
                            VALID_MINIGRID_ENVS[3]: '4-Door - CPLX',
                            VALID_MINIGRID_ENVS[5]: 'Manipulator_single_arch', 
                            VALID_MINIGRID_ENVS[6]: 'Manipulator_two_arch', 
                            }

MAX_COST_VAL = 51

USE_ALIAS: bool = True

DEBUG: bool = True


# Abstract Base Class for Plotting
class BasePlotter(ABC):
    def __init__(self, data: List[List[int]]):
        self.data = data

    def plot(self):
        """
        Abstract method to plot the data.
        """
        raise NotImplementedError

    def save_plot(self, file_name, plt_handle: plt, fig: Figure):
        """
         Simple method to save figure given the figure handle.
        """
        plt_handle.savefig(file_name, dpi=300, bbox_inches='tight')
        plt_handle.close(fig)


class StackedBarPlotter(BasePlotter):
    def __init__(self, data):
        super().__init__(data)
    

    def plot(self, file_name: str, labels: str = [], fig_title: str = ''):
        """
        Create stacked bar plots, one for each environment, showing the contribution
        of different human types to the total cost.
        """
        
        env_types = list(MINIGRID_NAME_ALIAS_DICT.values())
        # sys_str = list(SYS_ALIAS_DICT.values())[0]  # Only use the first system strategy
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Width of each bar
        bar_width = 0.6
        
        # Calculate positions for bars
        positions = np.arange(len(env_types))
        
        # Calculate total value for each environment
        env_values = []
        
        for env_idx, env in enumerate(env_types):
            total = 0
            # for human in mean_dict:
            #     if env in mean_dict[human] and sys_str in mean_dict[human][env]:
            #         total += mean_dict[human][env][sys_str]
            for synth_times in self.data[env]:
                # if env in mean_dict[human] and sys_str in mean_dict[human][env]:
                # total += mean_dict[human][env][sys_str]
                env_values.append(synth_times)
        
        # Plot bars
        bars = ax.bar(positions, env_values, width=bar_width, 
                    color='steelblue', edgecolor='black', linewidth=0.5)
        
        # # Add value labels on top of bars
        # for i, v in enumerate(env_values):
        #     ax.text(positions[i], v + 0.5, f'{v:.1f}', 
        #         ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Set x-tick positions and labels
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Environment {i+1}' for i in range(len(env_types))])
        
        # Increase font size for axis labels
        ax.set_xlabel('Environment Type', fontsize=16, labelpad=10)
        ax.set_ylabel('Total Cost', fontsize=16, labelpad=10)
        ax.set_title('Total Cost by Environment', fontsize=16, pad=20)
        
        # Increase tick label font sizes
        ax.tick_params(axis='x', which='major', labelsize=14, pad=8)
        ax.tick_params(axis='y', which='major', labelsize=12)
        
        # Add grid for better readability (only horizontal)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)  # Put grid behind bars
        
        # Add minor ticks for y-axis for better readability
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR_WAIT + 'cost_by_env', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        # plt.show(block=True)
        if fig_title != '':
            plt.title(fig_title)
        self.save_plot(PLOTS_DIR + file_name, plt_handle=plt, fig=fig)


class BoxPlotter(BasePlotter):
    def __init__(self, data):
        super().__init__(data)
    

    def plot(self, file_name: str, labels: str = [], fig_title: str = '') -> None:
        """
        Box plot method creates the figure handles and calls the draw_box_plot() method to plot the box plot on the same figure. 
         This modular approach allows or box plot to have differet sample and yet plot it on the same canvas.
        """
        fig, ax = plt.subplots()
        ax.set_ylabel('Cost')
        num_boxes: int = len(self.data)

        bplot = plt.boxplot(positions=list(range(num_boxes)),
                            labels=labels,
                            x=[np.array(cost) for cost in self.data],
                            showmeans=True,
                            patch_artist=True)
        
        # add # of samples on top of each box plot
        ax.set_ylim([0, MAX_COST_VAL + 4])
        upper_labels = [len(data) for data in self.data]
        pos = np.arange(num_boxes)
        for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
            # k = tick % 2
            ax.text(pos[tick], 0.95, upper_labels[tick],
                    transform=ax.get_xaxis_transform(),
                    horizontalalignment='center', size='small')
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
        self.save_plot(PLOTS_DIR_WAIT + file_name, plt_handle=plt, fig=fig)
        # plt.savefig(PLOTS_DIR_WAIT + file_name, dpi=300, bbox_inches='tight')
        # plt.savefig(PLOTS_DIR + file_name, dpi=300, bbox_inches='tight')
        # plt.close(fig)
    

    def plot_box_and_swarm_plor(self, file_name: str):
        """
         This method plots the box plot with swarm plot on the same canvas using the seaborn package along with stats printed on the right.
        """
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError("Seaborn is required for this function. Please install it using 'pip install seaborn'.")
        # Convert to numpy array to avoid pandas indexing issues
        assert len(self.data) == 1, "[Error] This method is only for single data set."
        costs_array = np.array(self.data)
        
        # Calculate statistics
        mean_cost = np.mean(costs_array)
        median_cost = np.median(costs_array)
        min_cost = min(costs_array)
        max_cost = max(costs_array)
        q1 = np.percentile(costs_array, 25)
        q3 = np.percentile(costs_array, 75)
        
        # Set the style
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(12, 7))
        
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
        self.save_plot(PLOTS_DIR + file_name, plt_handle=plt, fig=fig)
        
        # return plt.gcf()
    

    def plot_mean_dict_with_bars_for_specific_human(self, stats_dict, file_name: str, labels: str = [], fig_title: str = '') -> None:
        """
        Plot box plot for a specific human type (HAdv_Rnd by default) for various Envs.
        All environments are plotted on one canvas with color intensity representing sample counts.
        """
        import matplotlib.colors as mcolors

        # Enable TeX rendering
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'
        meanprpos = {'marker': 'D',          # Diamond marker
                    'markerfacecolor': 'red',  # Red fill
                    'markeredgecolor': 'black', # Black outline
                    'markersize': 6}

        # setting up things
        skip_human_types = ['HAdv', 'HCoop', 'Hrnd']  # e.g., ['HAdv'] to skip the HAdv human type
        human_types = [h for h in list(stats_dict.keys()) if h not in skip_human_types]

        # human_types = list(stats_dict.keys())
        env_types = list(MINIGRID_NAME_ALIAS_DICT.values())
        sys_str_types = ['Ours', 'Adm-Memless']

        fig, ax = plt.subplots()
        ax.set_ylabel('Payoff', fontsize=16, labelpad=10)
        self.data = []
        
        sample_counts = []
        labels = []
        min_count = math.inf
        max_count = 0
        for human_idx, human in enumerate(human_types):
            for env_idx, env in enumerate(env_types):
                for sys_idx, sys_str in enumerate(sys_str_types):
                    if human in stats_dict and env in stats_dict[human] and sys_str in stats_dict[human][env]:
                        if stats_dict[human][env][sys_str] is not None:
                            self.data.append(stats_dict[human][env][sys_str]['data'])
                            labels.append(sys_str)
                            sample_counts.append(len(stats_dict[human][env][sys_str]['data']))
        
        num_boxes: int = len(self.data)
        bplot = plt.boxplot(positions=list(range(num_boxes)),
                            labels=labels,     ### Overide the labels with the Env labels later.
                            x=[np.array(cost) for cost in self.data],
                            showmeans=True,
                            meanprops=meanprpos,
                            patch_artist=True, 
                            showfliers=False)

        # Create a color normalization
        min_count = min(sample_counts)
        max_count = max(sample_counts)
        norm = mcolors.Normalize(vmin=min_count, vmax=max_count)

        # Create a colormap - using a sequential colormap
        cmap = plt.cm.viridis  # You can try other colormaps like 'plasma', 'inferno', 'magma', etc.
        # cmap = plt.cm.Greys  # Using the Greys colormap for grayscale
        # cmap = plt.cm.Greys_r  # Using the Inverted Greys colormap for grayscale
        # cmap = plt.cm.coolwarm
        
        # Add sample counts on top of each box plot
        ax.set_ylim([0, MAX_COST_VAL + 4])
        pos = np.arange(num_boxes)
        for tick, count in zip(range(num_boxes), sample_counts):
            ax.text(pos[tick], 0.95, count,
                    transform=ax.get_xaxis_transform(),
                    horizontalalignment='center', size='small')

        
        # set markcolor to red
        # for box_mean in bplot['means']:
        #     # box_mean._set_markercolor('red')
        #     box_mean._color = 'red'

        # Fill with heat map colors based on sample count
        for i, (patch, count) in enumerate(zip(bplot['boxes'], sample_counts)):
            color = cmap(norm(count))
            patch.set_facecolor(color)

            if i % 2 == 0:
                line_style = '-'  # solid line
            else:
                line_style = '--'  # dashed line
            
            patch.set_linestyle(line_style)
            
            # Also color the median, whiskers, caps, and fliers to match
            # ['medians', 'whiskers', 'caps', 'fliers'] - Org list
            for element in ['medians', 'whiskers', 'caps']:
                if element == 'whiskers' or element == 'caps':
                    # These elements come in pairs
                    # bplot[element][i*2].set_color(color)
                    # bplot[element][i*2+1].set_color(color) 
                    bplot[element][i*2].set_color('black')
                    bplot[element][i*2+1].set_color('black')
                    bplot[element][i*2].set_linestyle(line_style)
                    bplot[element][i*2+1].set_linestyle(line_style)
                elif element in bplot:
                    bplot[element][i].set_color('black')  # Keep median line black for contrast
                    
                    if element == 'fliers':  # Make outlier points darker for visibility
                        bplot[element][i].set_markerfacecolor('black')
                        bplot[element][i].set_markeredgecolor('black')
        
        custom_labels = [r'$\mathbb{E}_1$', r'$\mathbb{E}_2$', r'$\mathbb{E}_3$', r'$\mathbb{E}_4$', r'$\mathbb{E}_5$']
        # Calculate positions for the 5 labels (positioned between pairs of boxes)
        if num_boxes > 1:
            # step = (num_boxes - 1) / 4  # To get 5 positions across the range
            # label_positions = [i * step for i in range(5)]
            label_positions = [0.5, 2.5, 4.5, 6.5, 8.5]
            
            # Set custom tick positions and labels
            ax.set_xticks(label_positions)
            ax.set_xticklabels(custom_labels)
            
            # Add minor ticks where the actual boxes are to help with alignment
            # ax.set_xticks(range(num_boxes), minor=True)
            
            # Add vertical grid lines at the label positions if desired
            for pos in label_positions[:-1]:
                ax.axvline(x=pos + 1, color='gray', linestyle=':', alpha=1)
        else:
            # If there's only one box, just put the label there
            ax.set_xticks([0])
            ax.set_xticklabels([custom_labels[0]])

        ax.tick_params(axis='x', which='major', labelsize=12, pad=8)  # Larger x-tick labels
        ax.tick_params(axis='y', which='major', labelsize=12)


        # Add a colorbar to show the sample count scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r'$|\it{win}|$', fontsize=16)

        ax.grid(True, linestyle='--', alpha=0.7, axis='y')  # Only y-axis grid lines
        # Set the grid to appear behind the plot elements
        ax.set_axisbelow(True)

        # add a legend on the bottom right
        legend_handles = []
        line_style = ['-', '--']
        for sys_idx, sys_str in enumerate(sys_str_types):
            legend_handles.append(mpatches.Patch(
                                    facecolor='white',
                                    linestyle=line_style[sys_idx],
                                    edgecolor='black',
                                    label=f'{sys_str_types[sys_idx]}'
                                ))
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D(
            [0], [0],
            color='none',  # No line
            **meanprpos,
            label='Mean'  # Label for the legend
        ))
        
        ax.legend(handles=legend_handles, loc='lower right', 
            #   bbox_to_anchor=(0.12, 0.99),
            #   title='System Strategies',
            #   title_fontsize=20,
              fontsize=12)
        # Set title
        if fig_title != '':
            # ax.set_title('Default', fontsize=10)
            plt.title(fig_title)
            
        # Make sure the figure fits well with the colorbar
        plt.tight_layout()
        
        self.save_plot(PLOTS_DIR_WAIT + file_name, plt_handle=plt, fig=fig)
    

    def create_alternative_visualizations(self, file_name: str):
        """
        Create alternative visualizations that might be useful
        """
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError("Seaborn is required for this function. Please install it using 'pip install seaborn'.")
        
         # Convert to numpy array to avoid pandas indexing issues
        assert len(self.data) == 1, "[Error] This method is only for single data set."
        costs_array = np.array(self.data)
        
        # Set up figure
        fig = plt.figure(figsize=(15, 10))
        
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
        self.save_plot(PLOTS_DIR + file_name, plt_handle=plt, fig=fig)



### UTILITY FUNCTIONS
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


def find_closest_file(minigrid_env, sys_type, valid_human_type, sys_str_type, wait_gw: bool = False):
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
        name = name.replace('NOT_CPLX', '')
        minigrid_game = f"{name}_DFA_GAME_NOT_CPLX"
    else:
        minigrid_game = f"{name}_DFA_GAME"

    if sys_type == 'random-sys':
        target_pattern = f"{minigrid_game}_{valid_human_type}_{sys_type}_{EPSILON}"
    else:    
        target_pattern = f"{minigrid_game}_{sys_str_type}_{valid_human_type}__{EPSILON}"

    # Find the closest match using difflib
    if wait_gw:
        closest_match = difflib.get_close_matches(target_pattern, FILES_WAIT, n=1)
    else:
        closest_match = difflib.get_close_matches(target_pattern, FILES, n=1)

    if not closest_match or len(closest_match) > 1:
        warnings.warn("[Error] Could not locate the closest file or return mroe than one yaml file names.")
        
    return closest_match[0]


def convert_defaultdict_to_dict(d):
    """
    Recursively convert a defaultdict to a regular dictionary.
    """
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def print_stats(costs: List[int]) -> Dict[str, Union[int, float]]:
    # Display statistics in the console
    cost_array = np.array(costs)
    # stats_dict = {
    #     "data": costs,
    #     "runs": len(costs),
    #     "mean": np.mean(cost_array),
    #     "median": np.median(cost_array),
    #     "min": min(costs),
    #     "max": max(costs),
    #     "q1": np.percentile(cost_array, 25),
    #     "q3": np.percentile(cost_array, 75),
    #     "freq_51": list(costs).count(51),
    #     "pct_51": (list(costs).count(51) / len(costs)) * 100
    # }

    stats_dict = {
        "data": costs,
        "runs": len(costs),
        "mean": float(np.mean(cost_array)),
        "median": float(np.median(cost_array)),
        "min": float(min(costs)),
        "max": float(max(costs)),
        "q1": float(np.percentile(cost_array, 25)),
        "q3": float(np.percentile(cost_array, 75)),
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


def get_stats_synth_times(data: Dict[str, Optional[float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Initialize dictionaries to store values
    comp_time_values = {
        'coop_time': [],
        'hopeadm_time': [],
        'safe_coop_time': [],
        'safeadm_game_constr': [],
        'safeadm_time': [],
        'safety_time': [],
        'cumul_safe_coop_time': [],
        'cumul_safeadm_game_constr': [],
        'cumul_safety_time': [],
        'wco_time': [],
        'wcoop_time': []
    }

    abs_dict_values = {
        '2p_game_constr_time': [],
        'DFA_game_constr_time': [],
        'DFA_game_edges': [],
        'DFA_game_nodes': []
    }

    def check_if_list_of_lists(values: Iterable) -> bool:
        for var in values:
            if isinstance(var, list):
                return True
            elif isinstance(var, float) or isinstance(var, int):
                return False
        return False

    # Extract values from the data
    for _, run_data in data.items():
        # Extract comp_time values
        for key in comp_time_values.keys():
            value = run_data['comp_time'].get(key, None)
            if value is not None:  # Skip None values or Key the above does not exists (happens for cumul)
                comp_time_values[key].append(value)
        
        # Extract abs_dict values
        for key in abs_dict_values.keys():
            abs_dict_values[key].append(run_data['abs_dict'][key])

    # Calculate means
    for key, values in comp_time_values.items():
        if check_if_list_of_lists(values=values):
            new_key: str = "cumul_" + key
            comp_time_values[new_key] = [np.sum(val) for val in values]

    comp_time_means: Dict[str, float] = {}
    for key, values in comp_time_values.items():
        if not check_if_list_of_lists(values) and len(values) != 0:
            comp_time_means[key] = np.mean(values)

    abs_dict_means: Dict[str, float] = {key: np.mean(values) for key, values in abs_dict_values.items()}

    total_synthesis_time: float = comp_time_means['coop_time'] +  comp_time_means['wco_time'] +  comp_time_means['wcoop_time'] + comp_time_means['safeadm_time']
    total_abs_time: float = abs_dict_means['DFA_game_constr_time']

    if comp_time_means.get('hopeadm_time') is not None:
        total_synthesis_time += comp_time_means['hopeadm_time']

    print(f"\nMean values for Total synthesis times: {total_synthesis_time:.2f} s")

    percentage_coop = (comp_time_means['coop_time'] / total_synthesis_time) * 100
    percentage_wco = (comp_time_means['wco_time'] / total_synthesis_time) * 100
    percentage_wcoop = (comp_time_means['wcoop_time'] / total_synthesis_time) * 100
    percentage_safeadm = (comp_time_means['safeadm_time'] / total_synthesis_time) * 100
        

    # percentage_safe_coop = (comp_time_means['safe_coop_time']/comp_time_means['safeadm_time']) * 100
    # percentage_safety = (comp_time_means['safety_time']/comp_time_means['safeadm_time']) * 100

    percentage_safe_coop = (comp_time_means['cumul_safe_coop_time']/comp_time_means['safeadm_time']) * 100
    percentage_safety = (comp_time_means['cumul_safety_time']/comp_time_means['safeadm_time']) * 100
    percentage_safeadm_game_constr = (comp_time_means['cumul_safeadm_game_constr']/comp_time_means['safeadm_time']) * 100

    percentage_game_time = (abs_dict_means['2p_game_constr_time']/total_abs_time) * 100
    # percentage_daf_game_time = (abs_dict_means['safety_time']/total_abs_time) * 100

    comp_time_percentage = {"percentage_coop": percentage_coop,
                            "percentage_wco": percentage_wco,
                            "percentage_wcoop": percentage_wcoop,
                            "percentage_safeadm": percentage_safeadm}

    if comp_time_means.get('hopeadm_time'):
        percentage_hopeadm = (comp_time_means['hopeadm_time'] / total_synthesis_time) * 100
        comp_time_percentage["percentage_hopeadm"] = percentage_hopeadm

    

    safeadm_comp_time_percentage = {"percentage_safety": percentage_safety,
                                    "percentage_safe_coop": percentage_safe_coop}
    
    abs_constr_time_percentage = {"percentage_game": percentage_game_time}


    print("\nPercentages of total synthesis time:")
    print(f"  coop_time: {percentage_coop:.2f}%")
    print(f"  wco_time: {percentage_wco:.2f}%")
    print(f"  wcoop_time: {percentage_wcoop:.2f}%")
    print(f"  safeadm_time: {percentage_safeadm:.2f}%")

    if comp_time_percentage.get("percentage_hopeadm") is not None:
        print(f"  hopeadm_time: {percentage_hopeadm:.2f}%")

    print("\nPercentages of Safe Adm synthesis time: ")
    print(f" Safety Game time: {percentage_safety:.2f}%")
    print(f" Safe Coop Game time: {percentage_safe_coop:.2f}%")
    print(f" Safe Adm Game Construction time: {percentage_safeadm_game_constr:.2f}%")

    # print("\nAbs. Size and Construction time: ")
    # print(f" DFA Game Nodes {int(abs_dict_means['DFA_game_nodes'])}")
    # print(f" DFA Game Edges {int(abs_dict_means['DFA_game_edges'])}")
    # print(f" Game time: {abs_dict_means['2p_game_constr_time']:.2f} s")
    # print(f" DFA Game time: {abs_dict_means['DFA_game_constr_time']:.2f} s")
    
    return comp_time_percentage, safeadm_comp_time_percentage




def plot_mean_dict_with_bars(mean_dict):
    """
    Plot the mean_dict dictionary as bar charts.
    Each human type will have 8 bars (4 environments x 2 system types).
    Different colors will be used for different environments, and hatching patterns
    will distinguish between the two system strategies.
    """
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    

    # Enable TeX rendering
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    # Both these dictionary are sued to matching the naming convention of the environments and system strategies in the paper.
    LEGENDS_DICT = {'IJCAI25-Lava': r'$\mathbb{E}_1$',
                    '1-Door': r'$\mathbb{E}_2$',
                    '3-Door': r'$\mathbb{E}_3$',
                    '4-Door - NOT CPLX': r'$\mathbb{E}_4$',
                    '4-Door - CPLX': r'$\mathbb{E}_5$'
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
    # env_colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3']
    env_colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']
    
    # Hatching patterns for different system strategies
    sys_hatches = ['', '///']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    # fig, ax = plt.subplots(figsize=(14, 14))
    
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
    ax.grid(axis='y', linestyle='--', alpha=1.0)
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
    
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR_WAIT + 'mean_cost_bar_chart_no_Hadv_all_five', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


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




def main_plot_average_synthesis():
    """
     The main method to plot the average time to compute adm rat stratgeies as stacker bar plot.
    """
    import sys

    percent_synth_times = {}
    for env in VALID_MINIGRID_ENVS:
        yaml_files =[]
        costs = []
        xlabels = []
        # target_pattern = f"comp_time_{env}_WAIT"
        target_pattern = env
        closest_match = difflib.get_close_matches(target_pattern, FILES_SYNTH_WAIT, n=1)
        yaml_files.append(closest_match[0])

        # load data
        with open(ROOT_PATH + "/construction_and_synthesis_data/" + yaml_files[-1], 'r') as stream:
            synthesis_data: dict = yaml.load(stream, Loader=yaml.Loader)

        # process data
        print(f"*************************** {env} ***************************")
        percent_synth_times[MINIGRID_NAME_ALIAS_DICT[env]], _ = get_stats_synth_times(data=synthesis_data)

        # plotting
        # plotter = StackedBarPlotter(data=percent_synth_times)
        # plotter.plot()
    sys.exit(-1)
        



# Main execution
if __name__ == "__main__":
    main_plot_average_synthesis()

    # mean_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    # stats_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    # for htype in ENV_ALIAS_DICT.values():
    #     for env_gridworld in VALID_MINIGRID_ENVS:
    #         for stype in VALID_SYS_STR_TYP:
    #             mean_dict[htype][MINIGRID_NAME_ALIAS_DICT.get(env_gridworld)][SYS_ALIAS_DICT[stype]] = MAX_COST_VAL
    #             stats_dict[htype][MINIGRID_NAME_ALIAS_DICT.get(env_gridworld)][SYS_ALIAS_DICT[stype]] = None
    
    # for env in VALID_MINIGRID_ENVS:
    #     # if env != 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0':
    #     #     continue
    #     for human in VALID_HUMAN_TYPE:
    #         yaml_files =[]
    #         costs = []
    #         xlabels = []
    #         for st in VALID_SYS_TYPE + VALID_SYS_STR_TYP:
    #         # for st in VALID_SYS_STR_TYP:
    #             yaml_files.append(find_closest_file(minigrid_env=env, sys_type=st, valid_human_type=human, sys_str_type=st, wait_gw=True))

    #             # Extract costs
    #             one_env_cost = extract_costs_from_yaml(yaml_files[-1], wait_gw=True)
    #             if len(one_env_cost) > 0:
    #                 costs.append(one_env_cost)
    #                 xlabels.append(SYS_ALIAS_DICT.get(st))

    #                 if DEBUG:
    #                     print("***************************************************************************************************")
    #                     print(f"{env} - {human} - {st}")
    #                     one_env_stats_dict: dict = print_stats(one_env_cost)
    #                     print("***************************************************************************************************")
    #                     mean_dict[ENV_ALIAS_DICT.get(human)][MINIGRID_NAME_ALIAS_DICT.get(env)][SYS_ALIAS_DICT[st]] = float(one_env_stats_dict['mean'])
    #                     stats_dict[ENV_ALIAS_DICT.get(human)][MINIGRID_NAME_ALIAS_DICT.get(env)][SYS_ALIAS_DICT[st]] = one_env_stats_dict 

            # if USE_ALIAS:
            #     human = ENV_ALIAS_DICT.get(human)
            # fig_name = f"cost_{human}_{env}_waiting.png"

            # barplotter = BoxPlotter(data=costs)
            # if len(costs) > 0:    
            #     if USE_ALIAS:
            #         # labels = [SYS_ALIAS_DICT.get(sys_type) for sys_type in VALID_SYS_STR_TYP] + [SYS_ALIAS_DICT.get(sys_type) for sys_type in VALID_SYS_TYPE]
            #         env_alias = MINIGRID_NAME_ALIAS_DICT.get(env)
            #         barplotter.plot(file_name=fig_name, labels=xlabels, fig_title=env_alias + "-" + human)
            #     else:
            #         barplotter.plot(file_name=fig_name, labels=xlabels,  fig_title=env + "-" + human)

    # for Adv huam env 
    # dump the dictionary to a yaml file
    # mean_dict = convert_defaultdict_to_dict(mean_dict)
    # with open('mean_dict_gw_wait_all_five.yaml', 'w') as file:
    #     yaml.dump(mean_dict, file)

    # stats_dict = convert_defaultdict_to_dict(stats_dict)
    # with open('stats_dict_gw_wait_all_five.yaml', 'w') as file:
    #     yaml.dump(stats_dict, file)
    
    
    # load yaml dictionary
    # with open('ral25_benchmark_data/stats_dict_gw_wait_all_five.yaml', 'r') as file: 
    #     stats_dict = yaml.load(file, Loader=yaml.Loader) 

    # barplotter = BoxPlotter(data=None)
    # barplotter.plot_mean_dict_with_bars_for_specific_human(stats_dict=stats_dict,
    #                                                        file_name='box_plot_HAdv_Rnd',
    #                                                        labels=VALID_SYS_STR_TYP,
    #                                                     #    fig_title='Box plot for Different System Strategies for HAdv_Rnd', 
    #                                                        fig_title='')

    with open('ral25_benchmark_data/mean_dict_gw_wait_all_five.yaml', 'r') as file: 
        mean_dict = yaml.load(file, Loader=yaml.Loader) 

    # plot_mean_dict(mean_dict)
    # plot_mean_dict_2(mean_dict)
    plot_mean_dict_with_bars(mean_dict)

    #### TESTING plotting for single file
    # file_name = "only_four_door_wait/MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0_DFA_game_QuantitativeAdmMemorless_random-human__120250317_010139.yaml"
    # costs = extract_costs_from_yaml(file_name, get_game_stats=True, wait_gw=False)
    # plot_box_plot(np.array(costs), file_name='testing', labels=['Adm-Mem'], fig_title='4-Door')
    
    