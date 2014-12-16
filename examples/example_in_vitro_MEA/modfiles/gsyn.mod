TITLE another leaky passive membrane channel used to emulate gradient synapse input

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX gsyn
	NONSPECIFIC_CURRENT i
	RANGE g, e
}

PARAMETER {
	g = .000	(S/cm2)	<0,1e9>
	e = 0	(mV)
}

ASSIGNED {v (mV)  i (mA/cm2)}

BREAKPOINT {
	i = g*(v - e)
}
