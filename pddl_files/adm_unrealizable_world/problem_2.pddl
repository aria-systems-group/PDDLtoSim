(define (problem franka_adm_problem) (:domain franka_unrealizable_world)
(:objects
    franka - robot
    else - robo_loc

    ;;; Region from which human can move within it.
    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc
    
    ;;; Region from which human can place an obj in l9.
    l6 - hbox_loc
    l7 - hbox_loc
    l8 - hbox_loc

    ;; Region where robot can not reach
    l9 - hbox_loc
	
    b0 - box
    b1 - box
)


;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l0)
    (on b1 l6)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l6)
    (on b1 l7)
))

)