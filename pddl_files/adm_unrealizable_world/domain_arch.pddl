;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Arch Construction Domain File
;;; In this file, we have additional Top location and I use the "either" keyword
;;; in the human-move action to allow human to move from either hbox or top locations to either hbox or top locations
;;; See: https://planning.wiki/ref/pddl/domain#either for more info.  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain franka_adm_arch_unrealizable_world)

(:requirements :strips :typing)

(:types
    robot - object
    box - object
    location - object
    robo_loc - location
    box_loc - location
    hbox_loc - box_loc
    top_loc - location
)

(:predicates
    (holding ?b - box ?l - box_loc)
    (ready ?l - location)

    (to-obj ?b - box ?l - box_loc)
    (to-loc ?b - box ?l - box_loc)

    (on ?b - box ?l - box_loc)
)


(:action transit
    :parameters (?b - box ?l1 - location ?l2 - (either box_loc top_loc))
    :precondition (and 
        (ready ?l1)
    )
    :effect (and 
        (not (ready ?l1))
        (to-obj ?b ?l2)
        (not (on ?b ?l2))
    )
)


(:action grasp
    :parameters (?b - box ?l - (either box_loc top_loc))
    :precondition (and 
        (to-obj ?b ?l)
    )
    :effect (and 
        (holding ?b ?l)
        (not (to-obj ?b ?l))
    )
)


(:action transfer
    :parameters (?b - box ?l1 - (either box_loc top_loc) ?l2 - (either box_loc top_loc))
    :precondition (and 
        (holding ?b ?l1)
    )
    :effect (and 
        (to-loc ?b ?l2)
    )
)


(:action release
    :parameters (?b - box ?l - (either box_loc top_loc))
    :precondition (and
        (to-loc ?b ?l)
    )
    :effect (and
        (ready ?l)
        (not (holding ?b ?l))
        (on ?b ?l)
        (not (to-loc ?b ?l))
    )
)


(:action human-move
	:parameters (?b - box ?l1 -(either hbox_loc top_loc) ?l2 - (either hbox_loc top_loc))
    :precondition (and
		(on ?b ?l1)
	)
	:effect (and
		(on ?b ?l2)
	)

)

)