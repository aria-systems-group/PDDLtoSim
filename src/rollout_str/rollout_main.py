import sys
import warnings

from typing import Union

from utls import timer_decorator

from src.rollout_str.rollout_adm import RefinedAdmStrategyRolloutProvider, AdmWinStrategyRolloutProvider, AdmStrategyRolloutProvider
from src.rollout_str.rollout_regret import BestEffortStrategyRolloutProvider, RegretStrategyRolloutProvider, AdvStrategyRolloutProvider

from regret_synthesis_toolbox.src.graph.product import ProductAutomaton
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantiativeRefinedAdmissible
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import RegretMinimizationStrategySynthesis 
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantitativeNaiveAdmissible, QuantitativeGoUAdmissible, QuantitativeGoUAdmissibleWinning

BestEffortClass = Union[QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn]
Strategy = Union[ValueIteration, RegretMinimizationStrategySynthesis, BestEffortClass]

VALID_ENV_STRINGS = ["manual", "no-human", "random-human", "epsilon-human"]

@timer_decorator
def rollout_strategy(strategy: Strategy,
                     game: ProductAutomaton,
                     debug: bool = False,
                     human_type: str = "random-human",
                     epsilon: float = 0.1,
                     max_iterations: int = 100):
    """
    A function that calls the appropriate rollout provide based on the strategy instance.

    TODO: Add cooperative human rollout for cooperative rollouts.
    Human_type: input
      "manual" for rollouts with user in the loop
      "no-human" for rollouts with no human intervention
      "random-human" for rollouts with random human intervention
      "epsilon-human" for rollouts with epsilon human intervention
    """
    if isinstance(strategy, RegretMinimizationStrategySynthesis):
        rhandle = RegretStrategyRolloutProvider(game=game,
                                                strategy_handle=strategy,
                                                debug=debug,
                                                max_steps=max_iterations)
    elif isinstance(strategy, ValueIteration):
        rhandle = AdvStrategyRolloutProvider(game=game,
                                             strategy_handle=strategy,
                                             debug=debug,
                                             max_steps=max_iterations)
    
    elif isinstance(strategy, QuantiativeRefinedAdmissible):
        rhandle = RefinedAdmStrategyRolloutProvider(game=game,
                                                    strategy_handle=strategy,
                                                    debug=debug,
                                                    max_steps=max_iterations)

    elif isinstance(strategy, QuantitativeGoUAdmissibleWinning):
        rhandle = AdmWinStrategyRolloutProvider(game=strategy.game,
                                                strategy_handle=strategy,
                                                debug=debug,
                                                max_steps=max_iterations)
    
    elif isinstance(strategy, (QuantitativeNaiveAdmissible, QuantitativeGoUAdmissible)):
        rhandle = AdmStrategyRolloutProvider(game=strategy.game,
                                             strategy_handle=strategy,
                                             debug=debug,
                                             max_steps=max_iterations)

    elif isinstance(strategy, (QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn)):
        rhandle = BestEffortStrategyRolloutProvider(game=game,
                                                    strategy_handle=strategy,
                                                    debug=debug,
                                                    max_steps=max_iterations)
    
    
    else:
        warnings.warn(f"[Error] We do not have rollout provder for strategy of type: {type(strategy)}")
        sys.exit(-1)
    
    if human_type == "manual":
        rhandle.manual_rollout()
    elif human_type == "no-human":
        rhandle.rollout_no_human_intervention()
    elif human_type == "random-human":
        rhandle.rollout_with_human_intervention()
    elif human_type == "epsilon-human":
        rhandle.rollout_with_epsilon_human_intervention(epsilon=epsilon)
    else:
        warnings.warn(f"[Error] Please enter a valid human type from:[ {', '.join(VALID_ENV_STRINGS)} ]")
        sys.exit(-1)

    return rhandle