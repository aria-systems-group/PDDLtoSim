(define (problem arch_2_tables) (:domain two_table_arch_objects_problem)
(:objects
    franka - robot
    else - robo_loc

    l0 - top_loc
    l1 - top_loc
    l2 - hbox_loc
    l3 - hbox_loc
    l4 - hbox_loc
;	l5 - box_loc
;	l6 - box_loc
	l7 - box_loc
	l8 - box_loc
	l9 - box_loc
	

    b0 - box
    b1 - box
    b2 - box
)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l7)
    (on b1 l2)
    (on b2 l4)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l9)
    (on b1 l3)
    (on b2 l2)
))

)