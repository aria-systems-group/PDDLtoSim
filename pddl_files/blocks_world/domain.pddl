;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 3 blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;Header and description

(define (domain diagonal_objects_problem)

;remove requirements that are not needed
(:requirements :strips :typing)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    robot - object
    box - object
    location - object
    robo_loc - location
    box_loc - location
)

;(:constants
;	else - robo_loc
;)

(:predicates ;todo: define predicates here
    (holding ?b - box ?l - box_loc)
    (ready ?l - location)

    (to-obj ?b - box ?l - box_loc)
    (to-loc ?b - box ?l - box_loc)

    (on ?b - box ?l - box_loc)
)


; (:functions ;todo: define numeric functions here

; )


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define actions here
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; (ready to move) Move to Grasp position (moved to location) -> 
;;; (not ready to move) Grasp (holding True and ready to move) ->
;;; (hold is true and ready to move) Tranfer with object in hand (move to location) ->
;;; (not read to move) Release (holding false and ready to move)

;;;;;;;;;; Move to a Grab-from-Top position action - Transit without Object;;;;;;;;;;;;;;;;;;;;;; 
; Parameters: This action takes in three parameters each of type box, from and to location respectively
; Precondition: The robot 'r' should be free and ready to move initially and the box 'b' should be at location 'l'  
; Effect: The roboy 'r' should not be ready to move (as it assumed grasp position) and has already moved to object's location 'l' and the box is not at location 'l'
(:action transit
    :parameters (?b - box ?l1 - location ?l2 - box_loc)
    :precondition (and 
        (ready ?l1)
    )
    :effect (and 
        (not (ready ?l1))
        (to-obj ?b ?l2)
        (not (on ?b ?l2))
    )
)

;;;; Perform the Grasp Action;;;;;;;;;;;;;;;;;;;;;;
; Parameters: Takes in the Robot and box type parameters  
; Precondition: Intially the robot 'r' should free and already moved to the box 'b'
; Effect: The robot 'r' should not be free and holding the box 'b' in its hands and it should be ready to move

(:action grasp
    :parameters (?b - box ?l - box_loc)
    :precondition (and 
        (to-obj ?b ?l)
    )
    :effect (and 
        (holding ?b ?l)
        (not (to-obj ?b ?l))
    )
)

;;;;; Perform Transfer motion with object in hand - Transit with object;;;;;;;;;;;;;;;;;
; Parameter: Takes in three parameters box, form and to location respectively
; Precondition: Initally the robot 'r' is holding the object 'b' and is ready to move
; Effect: The robot 'r' moved to the location 'l'  with object 'b' and is not ready not move and is ready to release the object

(:action transfer
    :parameters (?b - box ?l1 - box_loc ?l2 - box_loc)
    :precondition (and 
        (holding ?b ?l1)
    )
    :effect (and 
        (to-loc ?b ?l2)
    )
)

;;;;; Perform Release action;;;;;;;;;;;;;
; Parameter: Takes in the three parameters of type robot, box and location
; Precondition: The robot 'r' has already moved to the location 'l' with box 'b'
; Effect: The robot 'r' is free, not holding the box 'b', the box 'b' is on lokcation 'l' and the robot is free and not at the location 'l' anymore

(:action release
    :parameters (?b - box ?l - box_loc)
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


;;; Perform Human Intervention;;;;;
; A human can move objects when the box in on location l - A box is floating in the world when the robot is in grasp/release or move-to-location mode
;
;

(:action human-move
	:parameters (?b - box ?l1 - box_loc ?l2 - box_loc)
	:precondition (and
		(on ?b ?l1)
	)
	:effect (and
		(on ?b ?l2)
	)

)

)