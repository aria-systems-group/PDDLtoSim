(define (problem 5loc_problem) (:domain diagonal_objects_problem)
(:objects
    franka - robot

    l_0 - location
    l_1 - location
    l_2 - location
    l_3 - location
    l_4 - location
)

;todo: put the initial state's facts and numeric values here
(:init
    (free franka)
    (ready-to-move franka)

    (on rb l_3)
    (on bb_1 l_4)
    (on bb_2 l_1)
)

;todo: put the goal condition here
(:goal (and
    (on rb l_2)
    (on bb_1 l_0)
    (on bb_2 l_3)
))

)