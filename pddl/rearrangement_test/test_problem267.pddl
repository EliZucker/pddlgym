(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	bear-0 - moveable
	bear-1 - moveable
	monkey-2 - moveable
	bear-3 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-0-4 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-1-4 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-2-4 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
	loc-3-4 - static
    )

    (:init
    
	(IsBear bear-0)
	(IsBear bear-1)
	(IsMonkey monkey-2)
	(IsBear bear-3)
	(IsRobot robot)
	(At bear-0 loc-1-2)
	(At bear-1 loc-0-3)
	(At monkey-2 loc-2-3)
	(At bear-3 loc-2-2)
	(At robot loc-3-0)
	(Handsfree robot)

    ; Action literals
    
	(Pick bear-0)
	(Place bear-0)
	(Pick bear-1)
	(Place bear-1)
	(Pick monkey-2)
	(Place monkey-2)
	(Pick bear-3)
	(Place bear-3)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-0-3)
	(MoveTo loc-0-4)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-1-3)
	(MoveTo loc-1-4)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-2-3)
	(MoveTo loc-2-4)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
	(MoveTo loc-3-3)
	(MoveTo loc-3-4)
    )

    (:goal (and  (At monkey-2 loc-1-2) ))
)
    