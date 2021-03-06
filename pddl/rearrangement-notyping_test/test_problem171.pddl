(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	bear-0
	monkey-1
	bear-2
	bear-3
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-0-3
	loc-1-0
	loc-1-1
	loc-1-2
	loc-1-3
	loc-2-0
	loc-2-1
	loc-2-2
	loc-2-3
	loc-3-0
	loc-3-1
	loc-3-2
	loc-3-3
    )

    (:init
    
	(IsBear bear-0)
	(IsMonkey monkey-1)
	(IsBear bear-2)
	(IsBear bear-3)
	(IsRobot robot)
	(At bear-0 loc-1-2)
	(At monkey-1 loc-1-0)
	(At bear-2 loc-0-1)
	(At bear-3 loc-0-2)
	(At robot loc-2-3)
	(Handsfree robot)

    ; Action literals
    
	(Pick bear-0)
	(Place bear-0)
	(Pick monkey-1)
	(Place monkey-1)
	(Pick bear-2)
	(Place bear-2)
	(Pick bear-3)
	(Place bear-3)
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
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
	(MoveTo loc-3-3)
    )

    (:goal (and  (At bear-2 loc-3-2) ))
)
    