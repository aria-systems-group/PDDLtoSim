;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Two-player Franka Blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain dynamic_franka_box_world)

(:requirements :strips :typing)

(:types
    box - object
    location - object 
    general_loc - location
    ee_loc - location
    box_loc - general_loc
    hbox_loc - box_loc
)

; Indicates box is in end effector
(:constants else - general_loc ee - ee_loc) 

(:predicates
    (holding ?l - general_loc)
    (ready ?l - general_loc)
    (to-obj ?b - box)
    (on ?b - box ?l - location)
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define actions here
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(:action transit
    :parameters (?b - box ?l1 - general_loc ?l2 - general_loc)
    :precondition (and 
        (ready ?l1)
        (on ?b ?l2)
    )
    :effect (and 
        (to-obj ?b)
        (not (ready ?l1))   
    )
)


(:action grasp
    :parameters (?b - box ?l - general_loc)
    :precondition (and 
        (to-obj ?b)
        (on ?b ?l)
    )
    :effect (and 
        (holding ?l)
        (on ?b ee)
        (not (to-obj ?b))
        (not (on ?b ?l))
    )
)

(:action transfer
    :parameters (?b - box ?l1 - general_loc ?l2 - box_loc)
    :precondition (and 
        (holding ?l1)
        (on ?b ee)
    )
    :effect (and 
        (holding ?l2)
        (on ?b ee)
        (not (holding ?l1))
    )
)

(:action release
    :parameters (?b - box ?l - box_loc)
    :precondition (and
        (holding ?l)
        (on ?b ee)
    )
    :effect (and
        (ready ?l)
        (on ?b ?l)
        (not (holding ?l))
        (not (on ?b ee))
    )
)


(:action human-move
    :parameters (?b - box ?l1 - hbox_loc ?l2 - hbox_loc)
    :precondition (and
        (on ?b ?l1)
    )
    :effect (and
        (on ?b ?l2)
        (not (on ?b ?l1))
    )
)


)