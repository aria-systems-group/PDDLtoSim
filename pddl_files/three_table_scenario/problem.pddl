(define (problem 3obj_problem) (:domain stationary_object_sorting)
(:objects
    iiwa - robot

    table_0 - init_table
    table_1 - goal_table
    table_2 - free_table
)

(:init
    ;todo: put the initial state's facts and numeric values here
    (free iiwa)
    (ready-to-move iiwa)

    (on box_0 table_0)
    (on box_1 table_0)
    (on box_2 table_0)
    (on box_3 table_0)
    (on box_4 table_0)

    (unobstructed box_0 box_0)
    ; (unobstructed box_0 box_1)
    ; (unobstructed box_0 box_2)
    (unobstructed box_0 box_3)
    (unobstructed box_0 box_4)

    (unobstructed box_1 box_0)
    (unobstructed box_1 box_1)
    (unobstructed box_1 box_2)
    ; (unobstructed box_1 box_3)
    (unobstructed box_1 box_4)

    (unobstructed box_2 box_0)
    (unobstructed box_2 box_1)
    (unobstructed box_2 box_2)
    ; (unobstructed box_2 box_3)
    (unobstructed box_2 box_4)

    (unobstructed box_3 box_0)
    (unobstructed box_3 box_1)
    (unobstructed box_3 box_2)
    (unobstructed box_3 box_3)
    (unobstructed box_3 box_4)

    (unobstructed box_4 box_0)
    (unobstructed box_4 box_1)
    (unobstructed box_4 box_2)
    (unobstructed box_4 box_3)
    (unobstructed box_4 box_4)

)

(:goal (and
    ;todo: put the goal condition here
    (on box_0 table_1)
    (on box_1 table_2)
))

;un-comment the following line if metric is needed
;(:metric minimize (???))
)