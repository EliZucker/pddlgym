(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	bear-0
	pawn-1
	pawn-2
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
    )

    (:init
    
	(IsBear bear-0)
	(IsPawn pawn-1)
	(IsPawn pawn-2)
	(IsRobot robot)
	(At bear-0 loc-1-2)
	(At pawn-1 loc-1-1)
	(At pawn-2 loc-0-2)
	(At robot loc-2-1)
	(Handsfree robot)

    ; Action literals
    
	(Pick bear-0)
	(Place bear-0)
	(Pick pawn-1)
	(Place pawn-1)
	(Pick pawn-2)
	(Place pawn-2)
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
    )

    (:goal (and  (At pawn-1 loc-0-1) ))
)
    