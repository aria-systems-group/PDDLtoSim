################################### Formulas #########################################
# Formulas follow Spot's syntax. Please see https://spot.lre.epita.fr/app/ Help section for more info.

# Pick and Place
FORMULA_2B_2L_OR = 'F(p01 || p17)'
FORMULA_2B_2L_AND = 'F(p01 & p17)'
# FORMULA_2B_2L_OR_W_SAFETY = 'F(p01 || p17) & G(!p18)'
FORMULA_2B_2L_OR_W_SAFETY = 'F(p01) & G(!p18)'
FORMULA_2B_2L_AND_W_SAFETY = 'F(p01)  & F(p17) & G(!p18)'
FORMULA_2B_2L_AND_W_TRAP = "!p18 U (p17 & p01)"

FORMULA_1B_3L_AND_W_TRAP = "G(!p08) U F(p07 & F(p06))"


# Arch Formulas 
ARCH_FORMULA = "F((l8 & l9 & l0) || (l3 & l2 & l1))"


# Diag Formulas
DIAG_FORMULA = "F((p22 & p14 & p03) || (p05 & p19 & p26))"



################################### Minigrid #########################################
# It is dictionary of environment names their corresponding formulas

minigrid_env_formulas = {'MiniGrid-FloodingLava-v0': ['F(floor_green_open)'],
                        'MiniGrid-CorridorLava-v0': ['F(floor_green_open)'],
                        'MiniGrid-ToyCorridorLava-v0': ['F(floor_green_open)'], 
                        'MiniGrid-FishAndShipwreckAvoidAgent-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgentIn4Square-v0': ['F(floor_green_open) & G!(agent_blue_right)',
                                                              '!(agent_blue_right) U (floor_green_open) ',
                                                              'GF(floor_green_open) & G!(agent_blue_right)',
                                                              'G!(agent_blue_right) & G!(floor_green_open)'],
                        'MiniGrid-FourGrids-v0': 'F(floor_green_open)'[],
                        'MiniGrid-ChasingAgent-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgentInSquare4by4-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgentInSquare3by3-v0': ['F(floor_green_open)']

                        }