import re
import pprint

from collections import deque
from typing import Union, Set, List
from src.graph_construction.two_player_game import TwoPlayerGame


def remove_non_reachable_states(game: TwoPlayerGame, debug: bool = False):
        """
        A helper method that removes all the states that are not reachable from the initial state. This method is
        called by the edge weighted are reg solver method to trim states and reduce the size of the graph

        :param game:
        :return:
        """
        print("Starting purging nodes")
        # get the initial state
        _init_state = game.get_initial_states()[0][0]
        _org_node_set: set = set(game._graph.nodes())

        stack = deque()
        path: set = set()

        stack.append(_init_state)
        while stack:
            vertex = stack.pop()
            if vertex in path:
                continue
            path.add(vertex)
            for _neighbour in game._graph.successors(vertex):
                stack.append(_neighbour)

        _valid_states = path

        _nodes_to_be_purged: set = _org_node_set - _valid_states
        game._graph.remove_nodes_from(_nodes_to_be_purged)
        if debug:
            print("Nodes Purged")
            pprint.pp(_nodes_to_be_purged)
        print(f"Done purging nodes: # Nodes Purged: {len(_nodes_to_be_purged)}")


def modify_abstraction(game: TwoPlayerGame,
                       all_human_loc: Set[str],
                       hopeless_human_loc: Union[Set[str], str],
                       human_only_loc: Union[Set[str], str],
                       debug: bool = False):
    """
     A helper function that modifies a game that indicates that robot will commit to safe game if a safe adm str exists.

     Setup: Give game G, human_only_loc - set of locations where the human can move object to and 
     hopeless_human_loc - the set of human locations from which the human can move an object to human_only_loc, we first:
     1. Remove all Sys edges to human_only_loc - includes - transit and transfer edges to this loc
     2. Remove all Human edges from locs not in hopeless_human_loc to human_only_loc.
     3. Finally, remove states that are not reachable. - helps in VI
    """
    loc_pattern = "[l|L][\d]+"
    hopeful_human_loc: Set[str] = all_human_loc.difference(hopeless_human_loc, human_only_loc)
    assert hopeful_human_loc.intersection(hopeless_human_loc) == set(), "[Error] Set of Hopeful human locs and Set of Hopeless human locs should be disjoint. Fix this!!!"
    assert all_human_loc == hopeful_human_loc.union(hopeless_human_loc, human_only_loc), "[Error] Set of Hopeful human locs, Hopeless human locs, Human Only Locs should be the set of all human locs. Fix this!!"
    edges_to_rm = set()
    
    for _e in game._graph.edges():
        _u: str = _e[0]
        _v: str = _e[1]
        edge_action: str = game._graph[_u][_v][0].get('actions')

        if game._graph.nodes[_u]['player'] == 'adam':
            if 'human-move' not in edge_action:
                continue
            loc_states: List[str] = re.findall(loc_pattern, edge_action)
            assert len(loc_states) == 2, "[Error] From and To loc missing from transit or transfer action. Fix this!!!"
            _from_loc = loc_states[0]
            _to_loc = loc_states[1]

            if _from_loc in hopeful_human_loc and _to_loc in human_only_loc:
                edges_to_rm.add(_e)
                # print(edge_action)
            elif _from_loc in hopeful_human_loc and _to_loc in hopeless_human_loc:
                edges_to_rm.add(_e)
                # print(edge_action)
            elif _from_loc in hopeless_human_loc and _to_loc in hopeful_human_loc:
                edges_to_rm.add(_e)
                # print(edge_action)
            elif _from_loc in human_only_loc and _to_loc in hopeless_human_loc.union(hopeful_human_loc):
                edges_to_rm.add(_e)
                # print(edge_action)
        
        else:
            assert game._graph.nodes[_u]['player'] == 'eve', "[Error] Encountered a state that is not eve. Fix this!!!"
            if 'grasp' in edge_action or 'release' in edge_action:
                continue
            
            if 'else' in edge_action:
                continue
            
            # get the from and to loc
            loc_states: List[str] = re.findall(loc_pattern, edge_action)
            assert len(loc_states) == 2, "[Error] From and To loc missing from transit or transfer action. Fix this!!!"
            _to_loc = loc_states[1]

            assert 'transit'in edge_action or 'transfer' in edge_action, "[Error] Deleting incorrect action. Fix this!!!"
            if _to_loc in human_only_loc:
                edges_to_rm.add(_e)
    
    if debug:
        print(f"# of Edges to remove: {len(edges_to_rm)}")
        pprint.pp(edges_to_rm)
    
    # remove edges
    game._graph.remove_edges_from(edges_to_rm)

    # remove unreachable nodes
    remove_non_reachable_states(game=game, debug=debug)
    