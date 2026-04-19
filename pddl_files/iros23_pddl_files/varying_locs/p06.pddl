(define (problem dynamic_only_franka_world) (:domain franka_unrealizable_world)
(:objects
    franka - robot
    else - robo_loc

    ;;;;; Locs where only the robot can operate ;;;;;
    
    l0 - box_loc
    l1 - hbox_loc
    l2 - hbox_loc
    l3 - hbox_loc

    ;;;;; Locs where the robot & human can operate ;;;;;
    ; NOTE: The way pyperplan parses the PDDL file, you need atleast two human locs to construct `human-move` action

    l6 - hbox_loc
    l7 - hbox_loc
    l8 - hbox_loc
    l9 - hbox_loc
    l10 - hbox_loc
    l11 - hbox_loc
    l12 - hbox_loc
    l13 - hbox_loc
    l14 - hbox_loc


    b0 - box
    b1 - box
    b2 - box
)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    
    (on b0 l2)
    (on b1 l6)
    (on b2 l1)
)

(:goal 
(and
    (on b0 l0)
))

)