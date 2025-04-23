(define (problem franka_adm_arch_problem) (:domain franka_adm_arch_unrealizable_world)
(:objects
    franka - robot
    else - robo_loc
    ;;; Problem file for implementing TRO 25 manipulator exmaple with human undo moves only 
    
    l0 - box_loc
    l1 - box_loc
    
    l2 - hbox_loc
    ;l3 - hbox_loc
    
    ;; arch location
    l4 - hbox_loc
    l5 - hbox_loc
    l6 - top_loc

    ;; human placeholder locations
    ;l2 - top_loc
    l7 - hbox_loc
    l8 - hbox_loc
	
    b0 - box
    b1 - box
    b2 - box
    ;b3 - box
    ;b4 - box
)


;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    (on b0 l5)
    (on b1 l6)
    (on b2 l4)
    ;(on b3 l4)
    ;(on b4 l8)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l2)
    (on b1 l7)
    (on b2 l4)
    ;(on b3 l8)
    ;(on b4 l8)
))

)
