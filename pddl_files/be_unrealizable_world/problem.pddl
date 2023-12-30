(define (problem franka_sinmple_problem) (:domain franka_unrealizable_world)
(:objects
    franka - robot
    else - robo_loc

    l0 - box_loc
    l1 - hbox_loc
    l2 - hbox_loc
    l3 - hbox_loc
	

    b0 - box
    b1 - box
)


;todo: put the initial state's facts and numeric values here
(:init
    (ready else)

    (on b0 l0)
    (on b1 l2)
)

;todo: put the goal condition here
(:goal (and
; always add the config of the hbox loc here. It's fine not to include any of the box_loc config

    (on b0 l1)
    (on b1 l3)
))

)