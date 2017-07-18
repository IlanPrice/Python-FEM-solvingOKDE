# Python-FEM-solvingOKDE

This folder contains the code used to produce the results presented the accompanying special topic report, which was produced for the 'Python for Scientific Comptuing' elective as part of the MSc in Mathematical Modelling and Scientific Computing at the University of Oxford.

Most functions were written into the OKLibrary.py file, which is imported by the RunScript.py file, which contains sections for running most simulations.

The three functions are:
 - OKuniform - which is used to simulate the results in Section 5 among others
 - OKadaptive - which implements the adaptive time stepping algorithm
 - OKuniformIMX - which implements to alternative implicit-explicit splitting scheme.

The RunScript file has sections with the backbones in place for running each of the Section’s simulations, though not all parameter variation has been automated, some was done by hand.

The only sections whose code is not written as functions in that library (though their code is mostly identical), are the method of manufactured solutions scripts. There is one for the fully implicit method, and one for the implicit explicit scheme. 
