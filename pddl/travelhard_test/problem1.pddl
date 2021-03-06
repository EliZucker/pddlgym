
(define (problem travel) (:domain travel)
  (:objects
        ca - state
	car-0 - car
	ky - state
	nj - state
	og - state
	pe - state
	plane-0 - plane
	wv - state
  )
  (:goal (and
	(visited ca)
	(visited pe)))
  (:init 
	(Drive ca ca car-0)
	(Drive ca ky car-0)
	(Drive ca nj car-0)
	(Drive ca og car-0)
	(Drive ca pe car-0)
	(Drive ca wv car-0)
	(Drive ky ca car-0)
	(Drive ky ky car-0)
	(Drive ky nj car-0)
	(Drive ky og car-0)
	(Drive ky pe car-0)
	(Drive ky wv car-0)
	(Drive nj ca car-0)
	(Drive nj ky car-0)
	(Drive nj nj car-0)
	(Drive nj og car-0)
	(Drive nj pe car-0)
	(Drive nj wv car-0)
	(Drive og ca car-0)
	(Drive og ky car-0)
	(Drive og nj car-0)
	(Drive og og car-0)
	(Drive og pe car-0)
	(Drive og wv car-0)
	(Drive pe ca car-0)
	(Drive pe ky car-0)
	(Drive pe nj car-0)
	(Drive pe og car-0)
	(Drive pe pe car-0)
	(Drive pe wv car-0)
	(Drive wv ca car-0)
	(Drive wv ky car-0)
	(Drive wv nj car-0)
	(Drive wv og car-0)
	(Drive wv pe car-0)
	(Drive wv wv car-0)
	(Fly ca plane-0)
	(Fly ky plane-0)
	(Fly nj plane-0)
	(Fly og plane-0)
	(Fly pe plane-0)
	(Fly wv plane-0)
	(Walk ca)
	(Walk ky)
	(Walk nj)
	(Walk og)
	(Walk pe)
	(Walk wv)
	(adjacent ca og)
	(adjacent ky wv)
	(adjacent nj pe)
	(adjacent og ca)
	(adjacent pe nj)
	(adjacent pe wv)
	(adjacent wv ky)
	(adjacent wv pe)
	(at nj)
	(caravailable car-0)
	(isbluestate ca)
	(isredplane plane-0)
	(isredstate ky)
	(isredstate og)
	(planeavailable plane-0)
))
        