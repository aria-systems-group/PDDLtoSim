(define (problem diag_3_obj_2_tables) (:domain two_table_diagonal_objects_problem)
(:objects
    franka - robot
    else - robo_loc

;    l0 - hbox_loc
;    l1 - hbox_loc
;    l2 - hbox_loc
;    l3 - hbox_loc
;    l4 - hbox_loc
;	l5 - box_loc
;	l6 - box_loc
;	l7 - box_loc
;	l8 - box_loc
	l9 - box_loc
	

    b0 - box
;    b1 - box
;    b2 - box
)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l9)
;    (on b1 l6)
;    (on b2 l5)
)

;todo: put the goal condition here
(:goal (and
; add box_loc to compensate for above problem

    (on b0 l9)
;    (on b1 l8)
;    (on b2 l9)
))

)