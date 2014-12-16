TITLE decay of submembrane calcium concentration
:
: Internal calcium concentration due to calcium currents and pump.
: Differential equations.
:
: This file contains two mechanisms:
:
: 1. Simple model of ATPase pump with 3 kinetic constants (Destexhe 1992)
:
:       Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
:
:   A Michaelis-Menten approximation is assumed, which reduces the complexity
:   of the system to 2 parameters: 
:       kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
:	kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE
:   The values of these parameters are chosen assuming a high affinity of 
:   the pump to calcium and a low transport capacity (cfr. Blaustein, 
:   TINS, 11: 438, 1988, and references therein).  
:
:   For further information about this this mechanism, see Destexhe, A. 
:   Babloyantz, A. and Sejnowski, TJ.  Ionic mechanisms for intrinsic slow 
:   oscillations in thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993.
:
:
: 2. Simple first-order decay or buffering:
:
:       Cai + B <-> ...
:
:   which can be written as:
:
:       dCai/dt = (cainf - Cai) / taur
:
:   where cainf is the equilibrium intracellular calcium value (usually
:   in the range of 200-300 nM) and taur is the time constant of calcium 
:   removal.  The dynamics of submembranal calcium is usually thought to
:   be relatively fast, in the 1-10 millisecond range (see Blaustein, 
:   TINS, 11: 438, 1988).
:
: All variables are range variables
:
: Written by Alain Destexhe, Salk Institute, Nov 12, 1992
: Modified by TJ Velte, Univ of Minnesota, March 17, 1995

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX cad
	USEION ca READ ica, cai WRITE cai
	RANGE depth,kt,kd,cainf,taur
}

UNITS {
	(molar) = (1/liter)			: moles do not appear in units
	(mM)	= (millimolar)
	(um)	= (micron)
	(mA)	= (milliamp)
	(msM)	= (ms mM)
}

CONSTANT {
	FARADAY = 96489		(coul)		: moles do not appear in units
}

PARAMETER {
	depth	= .1	(um)		: depth of shell
	taur	= 1.5	(ms)		: remove first-order decay
	cainf	= 0.0001 (mM)
	kt	= 1e-5	(mM/ms)
	kd	= 0.0001	(mM)
}

STATE {
	cai		(mM) 
}

INITIAL {
	cai = kd
}

ASSIGNED {
	ica		(mA/cm2)
	drive_channel	(mM/ms)
	drive_pump	(mM/ms)
}
	
BREAKPOINT {
	SOLVE state METHOD derivimplicit
}

DERIVATIVE state { 

	drive_channel =  - (10000) * ica / (2 * FARADAY * depth)

	if (drive_channel <= 0.) { drive_channel = 0. }	: cannot pump below resting level

	drive_pump = -kt * cai / (cai + kd )		: Michaelis-Menten

	if (ica <= kd ) { drive_pump = 0. }

	cai' = drive_channel + (cainf-cai)/taur
}

