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
    red_box - box
    black_box - box
    location
)

; un-comment following line if constants are needed
(:constants 
    rb - red_box
    bb_1 - black_box
    bb_2 - black_box
)

(:predicates ;todo: define predicates here
    (free ?r - robot)
    (holding ?r - robot ?b - box)
    (ready-to-move ?r - robot)

    (moved-to-object ?r - robot ?b - box)
    (moved-to-location ?r - robot ?b - box ?l - location)

    (on ?b - box ?l - location)
)


; (:functions ;todo: define numeric functions here

; )


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define actions here
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; (ready to move) Move to Grasp position (moved to location) -> 
;;; (not ready to move) Grasp (ready to move) ->
;;; (hold is true and ready to move) Tranfer with object in hand (move to location) ->
;;; (not read to move) Release (holding false and ready to move)

;;;;;;;;;; Move to a Grab-from-Top position action - Transit without Object;;;;;;;;;;;;;;;;;;;;;; 
; Parameters: This action takes in three parameters each of type robot, box and location
; Precondition: The robot 'r' should be free and ready to move initially and the box 'b' should be at location 'l'  
; Effect: The roboy 'r' should not be ready to move (as it assumed grasp position) and has already moved to object's location 'l' and the box is not at location 'l'

(:action move-to-object-top
    :parameters (?r - robot ?b - box ?l - location)
    :precondition (and 
        (free ?r)
        (ready-to-move ?r)
        (on ?b ?l)
    )
    :effect (and 
        (not (ready-to-move ?r))
        (moved-to-object ?r ?b)
        (not (on ?b ?l))
    )
)

;;;;; Move to a Grab-from-side Position action - Transit without Object;;;;;;;;;;;;;;
; Parameters: Same as above
; Precondition: Same as above 
; Effect: Same as above. The only difference is in the actual execution.

(:action move-to-object-side
    :parameters (?r - robot ?b - box ?l - location)
    :precondition (and 
        (free ?r)
        (ready-to-move ?r)
        (on ?b ?l)
    )
    :effect (and 
        (not (ready-to-move ?r))
        (moved-to-object ?r ?b)
        (not (on ?b ?l))
    )
)

;;;; Perform the Grasp Action;;;;;;;;;;;;;;;;;;;;;;
; Parameters: Takes in the Robot and box type parameters  
; Precondition: Intially the robot 'r' should free and already moved to the box 'b'
; Effect: The robot 'r' should not be free and holding the box 'b' in its hands and it should be ready to move

(:action grasp
    :parameters (?r - robot ?b - box)
    :precondition (and 
        (free ?r)
        (moved-to-object ?r ?b)
    )
    :effect (and 
        (not (free ?r))
        (holding ?r ?b)
        (not (moved-to-object ?r ?b))
        (ready-to-move ?r)
    )
)

;;;;; Perform Transfer motion with object in hand - Transit with object;;;;;;;;;;;;;;;;;
; Parameter: Takes in three parameters of type robot, box and location respectively
; Precondition: Initally the robot 'r' is holding the object 'b' and is ready to move
; Effect: The robot 'r' moved to the location 'l'  with object 'b' and is not ready not move  

(:action move-to-location
    :parameters (?r - robot ?b - box ?l - location)
    :precondition (and 
        (holding ?r ?b)
        (ready-to-move ?r)
    )
    :effect (and 
        (moved-to-location ?r ?b ?l)
        (not (ready-to-move ?r))
    )
)

;;;;; Perform Release action;;;;;;;;;;;;;
; Parameter:Takes in the three parameters of type robot, box and location
; Precondition: The robot 'r' has already moved to the location 'l' with box 'b'
; Effect: The robot 'r' is free, not holding the box 'b', the box 'b' is on lokcation 'l' and the robot is free and not at the location 'l' anymore

(:action release
    :parameters (?r - robot ?b - box ?l - location)
    :precondition (and
        (moved-to-location ?r ?b ?l)
    )
    :effect (and
        (free ?r)
        (ready-to-move ?r)
        (not (holding ?r ?b))
        (on ?b ?l)
        (not (moved-to-location ?r ?b ?l))
    )
)


)