(define (problem diag_3_obj_2_tables) (:domain two_table_diagonal_objects_problem)
(:objects
    franka - robot
    else - robo_loc

    l0 - box_loc
    l1 - box_loc
    ;l2 - box_loc
    ;l3 - box_loc
    
	
    l6 - hbox_loc
	l7 - hbox_loc
	l8 - hbox_loc
    ;l9 - hbox_loc
    ;l10 - hbox_loc
    ;l11 - hbox_loc
    ;l12 - hbox_loc
    ;l13 - hbox_loc
    ;l14 - hbox_loc

	

    b0 - box
    b1 - box
    ;b2 - box
    ;b3 - box
    ;b4 - box
)

; b0/b3: black_box
; b1/b4: grey_box
; b2: white_box 

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l0)
    (on b1 l6)
    ;(on b2 l1)
    ;(on b3 l8)
    ;(on b4 l2)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l6)
    (on b1 l7)
    ;(on b2 l8)
    ;(on b3 l7)
    ;(on b4 l9)
))

)