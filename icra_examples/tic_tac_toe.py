from copy import deepcopy

NUM_CELL_OPTIONS = 3
NUM_CELLS = 9
NUM_ROWS = 3

IMPORTABLE = True
HUMAN_TERM_PROB = 0.05

ROBOT_PROB_NEIGHBOR_CELL = 0.00
HUMAN_PROB_NEIGHBOR_CELL = 0.00

EMPTY_CELL = 0
ROBOT_MARKER = 1
HUMAN_MARKER = 2


class Transition:
    action = ""
    prob_distr = []  # list of <new state, probability>

    def __init__(self, tpl_list):
        self.action = ""
        self.prob_distr = tpl_list


class State:
    robot_turn = True
    cells = [0]*NUM_CELLS
    transitions = []  # list of transitions

    def __init__(self):
        self.robot_turn = True
        self.cells = [0]*NUM_CELLS
        self.transitions = []

    def toInt(self):
        r = 0
        power = 1
        for i in range(len(self.cells)):
            r = r + self.cells[i] * power
            power *= NUM_CELL_OPTIONS
        if self.robot_turn:
            r += power
        return r

    def toPrismStr(self, primed):
        str = ""
        if self.robot_turn:
            if primed:
                str = "(rturn'=1) & "
            else:
                str = "(rturn=1) & "
        else:
            if primed:
                str = "(rturn'=0) & "
            else:
                str = "(rturn=0) & "

        if primed:
            for i in range(NUM_CELLS):
                str += "(o{}'={}) & ".format(i, self.cells[i])
            return str[:-2]
        else:
            for i in range(NUM_CELLS):
                str += "(o{}={}) & ".format(i, self.cells[i])
            return str[:-2]

    def toTplStr(self, primed):
        my_str = "("
        if self.robot_turn:
            my_str = "(1,"
        else:
            my_str = "(0,"

        for i in range(NUM_CELLS):
            my_str = my_str+str(self.cells[i])+","
        my_str = my_str[:-1]+")"
        return my_str


def index_to_r_c(index):
    r = int(index / NUM_ROWS)
    c = index % NUM_ROWS
    return (r, c)


def r_c_to_index(r, c):
    return NUM_ROWS * r + c

def game_finished(state):
    for r in range(NUM_ROWS):
        row_matches = True
        if state.cells[r_c_to_index(r,0)] == EMPTY_CELL:
            continue
        for c in range(1,NUM_ROWS):
            if state.cells[r_c_to_index(r,c)] != state.cells[r_c_to_index(r,0)]:
                row_matches = False
                break
        if row_matches:
            return True
    
    for c in range(NUM_ROWS):
        col_matches = True
        if state.cells[r_c_to_index(0,c)] == EMPTY_CELL:
            continue
        for r in range(1,NUM_ROWS):
            if state.cells[r_c_to_index(r,c)] != state.cells[r_c_to_index(0,c)]:
                col_matches = False
                break
        if col_matches:
            return True
    
    if state.cells[0] == state.cells[4] and state.cells[0] == state.cells[8] and state.cells[4] != EMPTY_CELL:
        return True
    
    if state.cells[2] == state.cells[4] and state.cells[2] == state.cells[6] and state.cells[4] != EMPTY_CELL:
        return True

    return False

def get_available_moves(state):
    if game_finished(state):
        return []
    moves = []
    for i in range(len(state.cells)):
        if state.cells[i] == EMPTY_CELL:
            moves.append(i)
    return moves

def is_open(state, index):
    return state.cells[index] == EMPTY_CELL

def get_neighbor_cells(state, index):
    r, c = index_to_r_c(index)
    ret = []
    if r > 0:
        ret.append(r_c_to_index(r-1, c))
    if r < NUM_ROWS-1:
        ret.append(r_c_to_index(r+1, c))
    if c > 0:
        ret.append(r_c_to_index(r, c-1))
    if c < NUM_ROWS-1:
        ret.append(r_c_to_index(r, c+1))

    filtered_ret = []
    for cell in ret:
        if is_open(state, cell):
            filtered_ret.append(cell)
    return filtered_ret

def genNeighbors(state):
    ret = []
    potential_moves = get_available_moves(state)
    for p_mv in potential_moves:
        neigh_cells = get_neighbor_cells(state, p_mv)
        if state.robot_turn:
            prob_desired = 1 - ROBOT_PROB_NEIGHBOR_CELL*len(neigh_cells)
        else:
            prob_desired = 1 - HUMAN_PROB_NEIGHBOR_CELL*len(neigh_cells)
        s_prime = State()
        s_prime.robot_turn = not state.robot_turn
        s_prime.cells = state.cells.copy()
        if state.robot_turn:
            s_prime.cells[p_mv] = ROBOT_MARKER
        else:
            s_prime.cells[p_mv] = HUMAN_MARKER
        ret.append(deepcopy(s_prime))
        transition_list = [(deepcopy(s_prime), prob_desired)]
        if prob_desired < 1:
            for l_n_c in neigh_cells:
                s_p = State()
                s_p.robot_turn = not state.robot_turn
                s_p.cells = state.cells.copy()
                if state.robot_turn:
                    s_p.cells[l_n_c] = ROBOT_MARKER
                    ret.append(deepcopy(s_p))
                    transition_list.append(
                        (deepcopy(s_p), ROBOT_PROB_NEIGHBOR_CELL))
                else:
                    s_p.cells[l_n_c] = HUMAN_MARKER
                    ret.append(deepcopy(s_p))
                    transition_list.append(
                        (deepcopy(s_p), HUMAN_PROB_NEIGHBOR_CELL))
        state.transitions.append(Transition(transition_list))
        r, c = index_to_r_c(p_mv)
        if state.robot_turn:
            state.transitions[-1].action = "robot_place_"+str(r)+"_"+str(c)
        else:
            state.transitions[-1].action = "human_place_"+str(r)+"_"+str(c)

    for n in ret:
        if state.robot_turn == n.robot_turn:
            print("ERROR, the turn didn't alternate=================================================================================")
    for t in state.transitions:
        for s_prime, prob in t.prob_distr:
            if state.robot_turn == s_prime.robot_turn:
                print("ERROR, the turn didn't alternate=================================================================================")
                print(len(state.transitions))

    return ret


def genGame(initial_state):
    s = initial_state

    visited_states = {}
    curr_frontier = []
    all_states = []

    curr_frontier.append(s)

    while(len(curr_frontier) > 0):
        # remove state from frontier and add to visited states
        s = curr_frontier[-1]
        curr_frontier.pop()
        my_tpl = {s.toInt(): s}

        if s.toInt() in visited_states:
            continue

        visited_states.update(my_tpl)
        all_states.append(s)
        # check all neighbors and add new ones to frontier
        neighbors = genNeighbors(s)
        for n in neighbors:
            if s.robot_turn == n.robot_turn:
                print("ERROR, the turn didn't alternate=================================================================================")

            if n.toInt() in visited_states:
                pass
            else:
                curr_frontier.append(n)
    return all_states


def print_front_matter():
    print("smg")

    print("player r1")
    robot_moves = ""
    for r in range(NUM_ROWS):
        for c in range(NUM_ROWS):
            robot_moves = robot_moves + "[robot_place_"+str(r)+"_"+str(c)+"], "
    print("  robot, "+robot_moves[:-2])
    print("endplayer")

    print("player h1")
    human_moves = ""
    for r in range(NUM_ROWS):
        for c in range(NUM_ROWS):
            human_moves = human_moves + "[human_place_"+str(r)+"_"+str(c)+"], "
    print("  human, "+human_moves[:-2])
    print("endplayer")


def print_global_vars():
    print("global rturn: [0..1] init 1;")
    for i in range(NUM_CELLS):
        print("global o{}: [0..{}] init 0;".format(i, HUMAN_MARKER))


def print_robot_module(game, state_to_int_map):
    print("module robot")
    for s in game:
        if s.robot_turn:
            for t in s.transitions:
                string = ""
                for s_prime, prob in t.prob_distr:
                    string += " {}: {} +".format(prob,
                                                 s_prime.toPrismStr(True))
                print("    [{}] ".format(t.action) +
                      s.toPrismStr(False)+" -> "+string[:-2]+";")
    print("endmodule")


def print_human_module(game, state_to_int_map):
    print("module human")
    for s in game:
        if not s.robot_turn:
            for t in s.transitions:
                string = ""
                for s_prime, prob in t.prob_distr:
                    string += " {}: {} +".format(prob,
                                                 s_prime.toPrismStr(True))
                print("    [{}] ".format(t.action) +
                      s.toPrismStr(False)+" -> "+string[:-2]+";")
    print("endmodule")

def print_labels():
    str = "("
    for r in range(NUM_ROWS):
        str = str+"("
        for c in range(NUM_ROWS):
            str = str + "(o{}=2) & ".format(r_c_to_index(r,c))
        str = str[:-3]+") | "

    for c in range(NUM_ROWS):
        str = str+"("
        for r in range(NUM_ROWS):
            str = str + "(o{}=2) & ".format(r_c_to_index(r,c))
        str = str[:-3]+") | "

    str = str + " ((o0=2) & (o4=2) & (o8=2)) | ((o2=2) & (o4=2) & (o6=2)));"
    print("label \"humanwin\" = "+str)
    str = str.replace("=2", "=1") 
    print("label \"robotwin\" = "+str)


def print_rewards():
    print("rewards")
    print("    true : 1;")
    print("endrewards")


def write_tra_file(game, state_to_int_map, og_state_to_int_map, int_to_state_map, num_choices, num_transitions, filename):
    with open(filename, "w") as f:
        f.write(str(len(state_to_int_map))+" " +
                str(num_choices) + " "+str(num_transitions)+"\n")
        for s, i in state_to_int_map.items():
            for j in range(len(s.transitions)):
                t = s.transitions[j]
                for s_prime, p in t.prob_distr:
                    f.write(str(i) + " " + str(j) + " " + (
                        str(og_state_to_int_map[s_prime.toInt()]) + " " + str(p) + " "+t.action+"\n"))


def write_sta_file(game, state_to_int_map, filename):
    with open(filename, "w") as f:
        my_str = ""
        for i in range(NUM_CELLS):
            my_str = my_str+",o"+str(i)
        f.write("(rloc,hloc,rturn"+my_str+")\n")
        for s, i in state_to_int_map.items():
            f.write(str(i)+":"+s.toTplStr(False)+"\n")


def write_lab_file(game, state_to_int_map, filename):
    with open(filename, "w") as f:
        f.write("0=\"init\" 1=\"deadlock\" 2=\"goalterm\" 3=\"goalcleaned\"\n")
        for s, i in state_to_int_map.items():
            my_str = ""
            if i == 0:
                my_str = my_str+" 0"
            # if len(s.transitions) == 0:
            #     my_str=my_str+" 1"
            # if s.robot_loc == TERM_LOC and s.human_loc == TERM_LOC:
            #     my_str=my_str+" 2"
            # if sat_goal(s.cells):
            #     my_str=my_str+" 3"
            # if len(my_str) > 0:
            #     f.write(str(i)+":"+my_str+"\n")


def write_pla_file(game, state_to_int_map, filename):
    with open(filename, "w") as f:
        f.write(str(len(state_to_int_map))+"\n")
        for s, i in state_to_int_map.items():
            p = 1
            if not s.robot_turn:
                p = 2
            f.write(str(i)+":"+str(p)+"\n")


def write_rew_file(game, state_to_int_map, filename):
    with open(filename, "w") as f:
        f.write(str(len(state_to_int_map))+" "+str(len(state_to_int_map))+"\n")
        for s, i in state_to_int_map.items():
            f.write(str(i)+" 1\n")


if __name__ == "__main__":
    
    initial_state = State()
    initial_state.cells = [0]*NUM_CELLS
    initial_state.robot_turn = True
    # initial_state.cells = [2,0,1,1,1,0,2,0,0]
    # initial_state.robot_turn = False

    game = genGame(initial_state)

    state_to_int_map = {initial_state.toInt(): 0}
    real_state_to_int_map = {initial_state: 0}
    int_to_state_map = {0: initial_state}
    counter = 1
    num_prism_choices = 0
    num_prism_transitions = 0
    for s in game:
        if not (s.toInt() in state_to_int_map):
            tpl = {s.toInt(): counter}
            state_to_int_map.update(tpl)
            real_state_to_int_map.update({s: counter})
            int_to_state_map.update({counter: s})
            counter += 1
            # for j in range(len(s.transitions)):
            #     num_prism_choices += 1
            #     t = s.transitions[j]
            #     for s_prime, p in t.prob_distr:
            #         num_prism_transitions+=1


    if(IMPORTABLE):
        # TODO: why does this need to be here?????
        for s, i in real_state_to_int_map.items():
            for j in range(len(s.transitions)):
                num_prism_choices += 1
                t = s.transitions[j]
                for s_prime, p in t.prob_distr:
                    num_prism_transitions += 1

        write_tra_file(game, real_state_to_int_map, state_to_int_map,
                    int_to_state_map, num_prism_choices, num_prism_transitions, "model.tra")
        write_sta_file(game, real_state_to_int_map, "model.sta")
        write_lab_file(game, real_state_to_int_map, "model.lab")
        write_pla_file(game, real_state_to_int_map, "model.pla")
        write_rew_file(game, real_state_to_int_map, "model.rew")
    else:
        print_front_matter()
        print_global_vars()
        print_robot_module(game, state_to_int_map)
        print_human_module(game, state_to_int_map)
        print_labels()
        print_rewards()