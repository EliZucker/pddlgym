(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	bear-0 - moveable
	bear-1 - moveable
	bear-2 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
    )

    (:init
    
	(IsBear bear-0)
	(IsBear bear-1)
	(IsBear bear-2)
	(IsRobot robot)
	(At bear-0 loc-0-0)
	(At bear-1 loc-1-0)
	(At bear-2 loc-1-2)
	(At robot loc-1-3)
	(Handsfree robot)

    ; Action literals
    
	(Pick bear-0)
	(Place bear-0)
	(Pick bear-1)
	(Place bear-1)
	(Pick bear-2)
	(Place bear-2)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-0-3)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-1-3)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-2-3)
    )

    (:goal (and  (At bear-2 loc-0-0)  (Holding bear-0) ))
)
    