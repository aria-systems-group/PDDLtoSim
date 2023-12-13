import math

LTLF_EXPECTED_DFA_OP = {
    "F(p01 | p17)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2
        },
    "F(p01)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2
        },
    "F(p00)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2
        }
    }


LTLF_EXPECTED_CAUSAL_GRAPH_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 55,
        "num_of_edges": 204
        },
    "F(p01)": {
        "num_of_nodes": 55,
        "num_of_edges": 204
        },
    "F(p00)": {
        "num_of_nodes": 55,
        "num_of_edges": 204
        }
    }


LTLF_EXPECTED_TRANS_SYS_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 241,
        "num_of_edges": 542
        },
    "F(p01)": {
        "num_of_nodes": 241,
        "num_of_edges": 542
        },
    "F(p00)": {
        "num_of_nodes": 241,
        "num_of_edges": 542
        }
    }


LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP = {
    "F(p01 | p17)": {
        "sys_count": 295,
        "env_count": 668,
        "num_of_nodes": 963,
        "num_of_edges": 2180
        },
    "F(p01)": {
        "sys_count": 295,
        "env_count": 668,
        "num_of_nodes": 963,
        "num_of_edges": 2180
        },
    "F(p00)": {
        "sys_count": 295,
        "env_count": 668,
        "num_of_nodes": 963,
        "num_of_edges": 2180
        }
    }


LTLF_EXPECTED_PRODUCT_GRAPH_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 964,
        "num_of_edges": 2033
        },
    "F(p01)": {
        "num_of_nodes": 964,
        "num_of_edges": 2091
        },
    "F(p00)": {
        "num_of_nodes": 964,
        "num_of_edges": 2086
        }
    }


LTLF_EXPECTED_MIN_MAX_VALUE_OP = {
    "F(p01 | p17)": {
        "min_max value": 4,
        "is_winning": True,
        },
    "F(p01)": {
        "min_max value": 4,
        "is_winning": True,
        },
    "F(p00)": {
        "min_max value": 1,
        "is_winning": True,
        }
    }

LTLF_EXPECTED_GoU_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 281,
        "num_of_edges": 550,
        "num_of_leaf_nodes": 6,
        "leaf_nodes": set({('q2', 5), ('q2', 2), 'vT', ('q2', 1), ('q2', 4), ('q2', 3)}),
        "edge_weights": set({1, 2, 3, 4, 5}),
        },
    "F(p01)": {
        "num_of_nodes": 457,
        "num_of_edges": 923,
        "num_of_leaf_nodes": 3,
        "leaf_nodes": set({'vT', ('q2', 4), ('q2', 5)}),
        "edge_weights": set({4, 5}),
        },
    "F(p00)": {
        "num_of_nodes": 2,
        "num_of_edges": 2,
        "num_of_leaf_nodes": 1,
        "leaf_nodes": set({('q2', 1)}),
        "edge_weights": set({1}),
        }
}


LTLF_EXPECTED_CVAL_OP = {
    "F(p01 | p17)": {
        "cval": set({1, 2, 3, 4, 5, math.inf}),
        "iters": 7
        },
    "F(p01)": {
        "cval": set({4, 5, math.inf}),
        "iters": 10
        },
    "F(p00)": {
        "cval": set({math.inf}),
        "iters": 2
        }
}


LTLF_EXPECTED_GoAlt_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 281,
        "num_of_edges": 550,
        "num_of_leaf_nodes": 6,
        "leaf_nodes": set({(('q2', 4), 1), (('q2', 3), 1), (('q2', 5), 1), ('vT', 1), (('q2', 2), 1), (('q2', 1), 1)}),
        },
    "F(p01)": {
        "num_of_nodes": 537,
        "num_of_edges": 1063,
        "num_of_leaf_nodes": 5,
        "leaf_nodes": set({(('q2', 5), 4), ('vT', 4), (('q2', 5), 5), (('q2', 4), 5), ('vT', 5)}),
        },
    "F(p00)": {
        "num_of_nodes": 2,
        "num_of_edges": 2,
        "num_of_leaf_nodes": 1,
        "leaf_nodes": set({(('q2', 1), math.inf)}),
        }
}

LTLF_EXPECTED_REG_STR_OP = {
    "F(p01 | p17)": {
        "rval": 3,
        "iters": 10
        },
    "F(p01)": {
        "rval": 0,
        "iters": 9
        },
    "F(p00)": {
        "rval": 0,
        "iters": 2
        }
}