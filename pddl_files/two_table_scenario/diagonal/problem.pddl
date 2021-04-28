(define (problem diag_3_obj_2_tables) (:domain two_table_diagonal_objects_problem)
(:objects
    franka - robot
    else - robo_loc

    l0 - hbox_loc
;    l1 - hbox_loc
    l2 - hbox_loc
;    l3 - hbox_loc
;    l4 - hbox_loc
	l5 - box_loc
	l6 - box_loc
	l7 - box_loc
;	l8 - box_loc
	l9 - box_loc
	

    b0 - box
    b1 - box
    b2 - box
;    b3 - box
;    b4 - box
)

; b0/b2: black_box
; b1/b3: grey_box
; b2/b4: white_box Invalid

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l0)
    (on b1 l7)
    (on b2 l6)
;    (on b3 l2)
;    (on b4 l7)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l0)
    (on b1 l2)
    (on b2 l5)
;    (on b3 l5)
;    (on b4 l8)
))

)