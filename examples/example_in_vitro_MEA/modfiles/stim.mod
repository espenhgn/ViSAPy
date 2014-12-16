COMMENT
Since this is an synapse current, positive values of i depolarize the cell
and is a transmembrane current.
ENDCOMMENT

NEURON {
	POINT_PROCESS ISyn
	RANGE del, dur, amp, i
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	del (ms)
	dur (ms)	<0,1e9>
	amp (nA)
}
ASSIGNED { i (nA) }

INITIAL {
	i = 0
}

BREAKPOINT {
	at_time(del)
	at_time(del+dur)
	if (t < del + dur && t >= del) {
		i = amp
	}else{
		i = 0
	}
}
