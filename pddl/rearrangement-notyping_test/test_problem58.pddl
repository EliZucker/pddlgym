(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	pawn-0
	monkey-1
	monkey-2
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-0-3
	loc-0-4
	loc-1-0
	loc-1-1
	loc-1-2
	loc-1-3
	loc-1-4
	loc-2-0
	loc-2-1
	loc-2-2
	loc-2-3
	loc-2-4
    )

    (:init
    
	(IsPawn pawn-0)
	(IsMonkey monkey-1)
	(IsMonkey monkey-2)
	(IsRobot robot)
	(At pawn-0 loc-1-3)
	(At monkey-1 loc-1-2)
	(At monkey-2 loc-0-1)
	(At robot loc-1-3)
	(Handsfree robot)

    ; Action literals
    
	(Pick pawn-0)
	(Place pawn-0)
	(Pick monkey-1)
	(Place monkey-1)
	(Pick monkey-2)
	(Place monkey-2)
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
    )

    (:goal (and  (At pawn-0 loc-0-3)  (At monkey-1 loc-0-3) ))
)
    