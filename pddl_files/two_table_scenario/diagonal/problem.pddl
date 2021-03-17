(define (problem diag_3_obj_2_tables) (:domain two_table_diagonal_objects_problem)
(:objects
    franka - robot
    else - robo_loc

    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc
    l3 - box_loc
    l4 - box_loc
;	l5 - box_loc
;	l6 - box_loc
;	l7 - box_loc
;	l8 - box_loc
;	l9 - box_loc
	

    b0 - box
    b1 - box
;    b2 - box
)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l0)
    (on b1 l1)
;    (on b2 l1)
)

;NOTE: Because of the way my code works, you need to include all the locations that are of type "box_loc" in the goal condition to make them "Relevant"
;ISSUE: My PDDL graph construction module relies on Pyperplan package. While returning operators relevant to a given task, they remove states that are sinks. From "hbox_loc" type states ; you do have "human-move" action but, "box_loc" type states are sinks because of the way I have wirtten my domain file. 

;todo: put the goal condition here
(:goal (and
; add box_loc to compensate for above problem
;    (on b0 l6)
;    (on b0 l5)

    (on b0 l3)
    (on b2 l4)
))

)