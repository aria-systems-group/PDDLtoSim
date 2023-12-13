(define (problem dynamic_regret_franka_world) (:domain dynamic_regret_franka_box_world)
(:objects

    franka - robot
    ;;;;; Locs where only the robot can operate ;;;;;

    else - robo_loc

    l0 - box_loc
    l1 - box_loc
    l2 - box_loc
	
    ;;;;; Locs where the robot & human can operate ;;;;;
    l6 - hbox_loc
	l7 - hbox_loc
	l8 - hbox_loc


    b0 - box
    b1 - box
)


(:init
    (ready else)
    (on b0 l0)
    (on b1 l6)
    ;(on b2 l1)
    ;(on b3 l8)
    ;(on b4 l2)
)


(:goal (and
    ;;;;;always add the config of the hbox loc here. It's fine not to include any of the box_loc config;;;

    (on b0 l6)
    (on b1 l7)
    ;(on b2 l8)
    ;(on b3 l7)
    ;(on b4 l9)
))

)