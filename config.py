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

# Adm Formula - start  in winning region..
FORMULA_ADM = '!(p09) U (p01 | (p06 & p17))'
FORMULA_ADM_HOPELESS = '!(p09) U ((p06 & p17))'
FORMULA_SAFE_ADM = 'F(p01) | F(p17)'   # very optimistic beh.
FORMULA_SAFE_ADM_TEST = 'F(p07 & p16) | F(p01 & X F(p02))'   # very optimistic beh.
FORMULA_SAFE_ADM_TEST_2 = 'F((p18 | p12) & X F((p20 & p11) | (p16 & p07)))'   # very optimistic beh.

# Benchmark formulas
FORMULA_ADM_B1 = 'F(p06)'
FORMULA_ADM_B2 = 'F(p06 & F(p07))'
FORMULA_ADM_B3 = 'F(p06 & F(p07 & F(p08)))'
FORMULA_ADM_B4 = 'F(p06 & F(p07 & F(p08 & F(p09))))'


## ADM -Tic-Tac-Toe 
FORMULA_ADM_TIC_TAC_TOE = 'F(win | draw)'
FORMULA_ADM_TIC_TAC_TOE_2 = 'F(win)'
FORMULA_ADM_TIC_TAC_TOE_3 = '(!(lose) U win)'

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
                        'MiniGrid-FourGrids-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgent-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgentInSquare4by4-v0': ['F(floor_green_open)'],
                        'MiniGrid-ChasingAgentInSquare3by3-v0': ['F(floor_green_open)']

                        }