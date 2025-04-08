from typing import List
from src.rollout_str.rollout_provider_if import RolloutProvider


class SafetyStrategyRolloutProvider(RolloutProvider):
    """
     This class implements the rollout provider for the safety strategy synthesis.
    """
    def __init__(self, game, strategy_handle, debug: bool = False, max_steps: int = 100, logger=None):
        super().__init__(game=game,
                         strategy_handle=strategy_handle,
                         debug=debug,
                         max_steps=max_steps,
                         logger=logger)
        
    
    def set_strategy(self):
        self._strategy = self._strategy__handle.sys_str
    
    def set_env_strategy(self):
        self._env_strategy = self._strategy_handle.env_str

    def set_target_states(self):
        self._target_states: List = self.game.get_accepting_states()

    def set_init_states(self):
        self._init_state = self.game.get_initial_states()[0][0]
    
    def set_absorbing_states(self):
        self._absorbing_states: List = self.game.get_absorbing_state()
        self._absorbing_states = set(self.absorbing_states).union(set(self.target_states))

    def set_sink_states(self):
        self._sink_states: List = self.game.get_trap_states()
    
    def set_state_values(self):
        pass


    def manual_rollout(self):
        """
         This method implements the manual rollout for the safety strategy synthesis.
        """
        pass