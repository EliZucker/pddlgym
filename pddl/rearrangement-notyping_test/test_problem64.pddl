(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	monkey-0
	pawn-1
	monkey-2
	bear-3
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-1-0
	loc-1-1
	loc-1-2
	loc-2-0
	loc-2-1
	loc-2-2
	loc-3-0
	loc-3-1
	loc-3-2
	loc-4-0
	loc-4-1
	loc-4-2
    )

    (:init
    
	(IsMonkey monkey-0)
	(IsPawn pawn-1)
	(IsMonkey monkey-2)
	(IsBear bear-3)
	(IsRobot robot)
	(At monkey-0 loc-2-0)
	(At pawn-1 loc-1-0)
	(At monkey-2 loc-4-0)
	(At bear-3 loc-0-1)
	(At robot loc-2-2)
	(Handsfree robot)

    ; Action literals
    
	(Pick monkey-0)
	(Place monkey-0)
	(Pick pawn-1)
	(Place pawn-1)
	(Pick monkey-2)
	(Place monkey-2)
	(Pick bear-3)
	(Place bear-3)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
	(MoveTo loc-4-0)
	(MoveTo loc-4-1)
	(MoveTo loc-4-2)
    )

    (:goal (and  (At pawn-1 loc-4-2)  (At monkey-0 loc-2-1) ))
)
    