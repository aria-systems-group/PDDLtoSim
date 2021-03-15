import os
import time
import math
import warnings
import pybullet as pb
import numpy as np

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

from src.pddl_env_simualtor.envs.panda_sim import PandaSim

def _pre_loaded_pick_and_place_action(pos):
    _wait_pos_left = [-0.2, 0.0, 0.9, math.pi, 0, math.pi]
    _wait_pos_right = [0.2, 0.0, 0.9, math.pi, 0, math.pi]
    
    # give pose and orientation
    panda.apply_high_level_action("openEE", [])

    panda.apply_high_level_action("transit", _wait_pos_left, vel=0.5)

    # panda.apply_action(panda._home_hand_pose)
    # # panda.apply_action(action=_pos, max_vel=0.5)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)

    _pos = [pos[0], pos[1], pos[2] + 0.3, math.pi, 0, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=2)
    # panda.pre_grasp(max_velocity=2)
    #
    # for _ in range(800):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 2. go down towards the object
    _pos = [pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=0.5)
    #
    # for _ in range(600):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # panda.grasp(obj_id=3)
    panda.apply_high_level_action("closeEE", [], vel=0.5)
    # for _ in range(300):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 3. grab the object
    _pos = [pos[0], pos[1], pos[2] + 0.3,  math.pi, 0, math.pi]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=0.5)
    # panda.grasp(obj_id=3)
    #
    # for _ in range(400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

    _pos = [-pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)

    panda.apply_high_level_action("openEE", [], vel=0.5)

    # _pre_pose = [0.2, 0.0, 0.9,
    #             min(math.pi, max(-math.pi, math.pi)),
    #             min(math.pi, max(-math.pi, 0)),
    #             min(math.pi, max(-math.pi, 0))]
    #
    # panda.apply_action(_pre_pose)
    # panda.grasp(obj_id=3)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 4. pick and place it somewhere else
    # _pos = [-pos[0], pos[1], pos[2] + 0.05]
    # panda.apply_action(action=_pos, max_vel=0.5)
    # panda.grasp(obj_id=3)
    #
    # for _ in range(800):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # release
    # panda.pre_grasp()
    # for _ in range(300):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # panda.apply_action(panda._home_hand_pose)
    # # panda.apply_action(action=_pos, max_vel=0.5)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)


def load_pre_built_loc_info(exp_name: str):
    if exp_name == "diag":
        loc_dict = {
            0: np.array([-0.7, -0.2, 0.17 / 2]),
            2: np.array([-0.4, 0.2, 0.17/2]),
            1: np.array([-0.4, -0.2, 0.17/2]),
            # 2: np.array([-0.7, -0.2, 0.17/2]),
            3: np.array([0.6, -0.2, 0.17/2]),
            4: np.array([0.45, 0.0, 0.17/2])
        }
    elif exp_name == "arch":
        pass
    else:
        warnings.warn("PLease enter a valid experiment name")

    return loc_dict

if __name__ == "__main__":
    record = False

    # build the product automaton
    # _project_root = os.path.dirname(os.path.abspath(__file__))
    #
    # # Experimental stage - lets try calling the function within pyperplan
    # domain_file_path = _project_root + "/pddl_files/blocks_world/domain.pddl"
    # problem_file_ath = _project_root + "/pddl_files/blocks_world/problem.pddl"
    #
    # causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=False)
    #
    # causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)
    #
    # transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    # transition_system_instance.build_transition_system(plot=False)
    #
    # two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    # two_player_instance.build_two_player_game(human_intervention=5, plot_two_player_game=False)
    # two_player_instance.set_appropriate_ap_attribute_name()
    # two_player_instance.modify_ap_w_object_types()
    #
    # dfa = two_player_instance.build_LTL_automaton(formula="F(p01 & ((p11 & p22) || (p12 & p21)))")
    # product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)
    # relabelled_graph = two_player_instance.internal_node_mapping(product_graph)
    # # relabelled_graph.plot_graph()
    #
    # # print some details about the product graph
    # print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
    # print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    # build the simulator
    if record:
        physics_client = pb.connect(pb.GUI,
                                    options="--minGraphicsUpdateTimeMs=0 --mp4=\"experiment.mp4\" --mp4fps=240")
    else:
        physics_client = pb.connect(pb.GUI)

    panda = PandaSim(physics_client, use_IK=1)
    _loc_dict = load_pre_built_loc_info("diag")
    _obj_loc = []
    for i in range(3):
        _obj_loc.append(_loc_dict.get(i))
        panda.world.load_object(urdf_name="red_box",
                                obj_name=f"red_box_{i}",
                                obj_init_position=_loc_dict.get(i),
                                obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

    # for _ in range(1500):
    for i, loc in enumerate(_obj_loc):
        # if i != 0:
        #     panda.apply_action(panda._home_hand_pose)
        #     for _ in range(100):
        #         pb.stepSimulation()
        #         time.sleep(0.01)
        # for _ in range(2):
        _pre_loaded_pick_and_place_action(loc)