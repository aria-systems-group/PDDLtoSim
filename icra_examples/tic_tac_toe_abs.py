import warnings

from typing import List, Tuple

from icra_examples.tic_tac_toe import genGame, State, NUM_CELLS, print_labels, r_c_to_index

from regret_synthesis_toolbox.src.graph import graph_factory
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

class TicTacToe():
    """
     Construct an absatrct two player graph from out ICRA 24 paper.
    """

    def __init__(self) -> None:
        self._two_player_game: TwoPlayerGraph = None
        self._formula: str = None

    @property
    def two_player_game(self):
        if isinstance(self._two_player_game, type(None)):
            warnings.warn("The Two player game is of type of None. Please build the Tic-Tac-Toe game before accessing it")
        return self._two_player_game
    
    @property
    def formula(self):
        return self._formula
    

    @formula.setter
    def formula(self, formula: str):
        if not isinstance(formula, str):
            warnings.warn("Please make sure the input formula is of type string.")
        self._formula = formula
    

    def check_diagonals(self, cells: Tuple[int]) -> int:
        # diagonal and anti-diagonal
        for val in [1, 2]:
            if ((cells[0] == val and cells[4] == val and cells[8] == val) or
                (cells[2] == val and cells[4] == val and cells[6] == val)):
                return val
        return 0

    def check_verticals(self, cells: Tuple[int]) -> int:
        for val in [1, 2]:
            for i in range(3): 
                if cells[r_c_to_index(0, 1)] == val and cells[r_c_to_index(1, i)] == val and cells[r_c_to_index(2, i)] == val:
                    return val
        return 0

    def check_horizontals(self, cells: Tuple[int]) -> int:
        for val in [1, 2]:
            for i in range(3): 
                if cells[r_c_to_index(i, 0)] == val and cells[r_c_to_index(i, 1)] == val and cells[r_c_to_index(i, 2)] == val:
                    return val
        return 0


    def get_state_label(self, cells: Tuple[int]) -> str:
        """
         A funtion to parse the conf of the game and assign a label. 
         A state is winning if:
            three consecutive markers are placed vertically, horizontally or diagonally
        """
        # robot marker -1 and human marker - 2. Empty slots - 0
        if cells.count(1) < 2 or cells.count(2) < 2:
            return ''
        
        status: int = self.check_diagonals(cells)
        if status != 0:
            return 'win' if status == 1 else 'lose'
        status: int = self.check_verticals(cells)
        if status != 0:
            return 'win' if status == 1 else 'lose'
        status: int =self.check_horizontals(cells)
        if status != 0:
            return 'win' if status == 1 else 'lose'
        
        # game is still going, no-label
        if cells.count(0) == 0:
            return 'draw'
        else:
            return ''
        
        # return status
        # if cells.count(1) >= 3:
        #     # check if placec diagonally
        #     idx = [loc for _, loc in enumerate(cells)]


    def construct_graph(self):
        """
         Main function to build the game as per our ICRA code and then wrap construct a game out of it.
        """
        self._two_player_game = graph_factory.get("TwoPlayerGraph",
                                                  graph_name="tic_tac_toe_game",
                                                  config_yaml="/config/tic_tac_toe_game",
                                                  save_flag=True,
                                                  plot=False)
        initial_state = State()
        initial_state.cells = [0]*NUM_CELLS
        initial_state.robot_turn = True

        # generate the game state and all transition
        game: List[State] = genGame(initial_state)

        # add init state to our game
        self._two_player_game.add_state(tuple(initial_state.cells), **{'init': True, 'ap': '', 'player': "eve" if initial_state.robot_turn is True else "adam"})

        for state in game:
            stpl = tuple(state.cells)
            if not self._two_player_game._graph.has_node(stpl):
                self._two_player_game.add_state(stpl, **{'init': False, 'ap': self.get_state_label(cells=stpl), 'player': "eve" if state.robot_turn is True else "adam"})
                # self._two_player_game._graph.nodes[stpl]['init'] = False
                # self._two_player_game._graph.nodes[stpl]['ap'] = self.get_state_label(cells=stpl)
                # self._two_player_game._graph.nodes[stpl]['player'] = "eve" if state.robot_turn is True else "adam"

            for j in range(len(state.transitions)):
                tran = state.transitions[j]
                for s_prime, p in tran.prob_distr:
                    succ_tpl = tuple(s_prime.cells)
                    if not self._two_player_game._graph.has_node(succ_tpl):    
                        self._two_player_game.add_state(succ_tpl, **{'init': False, 'ap': self.get_state_label(cells=succ_tpl), 'player': "eve" if s_prime.robot_turn is True else "adam"})
                        # self._two_player_game._graph.nodes[succ_tpl]['init'] = False
                        # self._two_player_game._graph.nodes[succ_tpl]['ap'] = self.get_state_label(cells=succ_tpl)
                        # self._two_player_game._graph.nodes[succ_tpl]['player'] = "eve" if s_prime.robot_turn is True else "adam"

                    self._two_player_game.add_edge(stpl, succ_tpl, **{'weight': 1 if state.robot_turn else 0 , 'actions': tran.action})
        
        assert len(game) == len(self._two_player_game._graph.nodes()), "[Error] Mismatch in # of state in the Game Abs and the Tic-Tac-Toe code"


    def build_LTLf_automaton(self, formula: str, debug: bool = False, plot: bool = False):
        """
         A method to construct LTLf automata using the regret_synthesis_tool.
        """
        self.formula = formula

        _ltl_automaton = graph_factory.get('LTLfDFA',
                                           graph_name="pddl_ltlf",
                                           config_yaml="/config/pddl_ltlf",
                                           save_flag=True,
                                           ltlf=formula,
                                           plot=plot)

        if debug:
            print(f"The pddl formula is : {formula}")

        return _ltl_automaton


    def build_product(self, dfa, trans_sys, plot: bool = False):
        _product_automaton = graph_factory.get("ProductGraph",
                                               graph_name="pddl_product_graph",
                                               config_yaml="/config/pddl_product_graph",
                                               trans_sys=trans_sys,
                                               automaton=dfa,
                                               # dfa=dfa,
                                               save_flag=True,
                                               prune=False,
                                               debug=False,
                                               absorbing=True,
                                               finite=False,
                                               plot=plot)

        print("Done building the Product Automaton")

        # Add the accepting state "accept_all" in the product graph with player = "eve"
        # should technically be only one if absorbing is true
        # _states = _product_automaton.get_accepting_states()

        # for _s in _states:
        #     _product_automaton.add_state_attribute(_s,
        #                                            attribute_key="player",
        #                                            attribute_value="eve")

        return _product_automaton


if __name__ == "__main__":
    game_handle = TicTacToe()
    game_handle.construct_graph()


