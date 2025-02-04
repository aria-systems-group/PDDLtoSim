(define (problem franka_adm_human_undo_problem) (:domain franka_unrealizable_world)
(:objects
    franka - robot
    else - robo_loc
    ;;; Problem file for implementing IJCAI 25 manipulator exmaple with human undo moves only 
    
    ;;; Region from which human can move within it.
    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc
    l3 - hbox_loc
    
    ;; arch location
    l4 - hbox_loc
    l5 - hbox_loc
    l6 - hbox_loc
	
    b0 - box
    b1 - box
    b2 - box
    b3 - box
)


;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    (on b0 l0)
    (on b1 l1)
    (on b2 l2)
    (on b3 l3)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l2)
    (on b1 l3)
    (on b2 l1)
    (on b3 l1)
))

)
