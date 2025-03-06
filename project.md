### Creating project which have the custom backward engine in c++ with memory efficient 

## Task1 
    - fisrt create cmake project.
    - create tensor namespace and it should have maember like data , gradient,set of child which point toward from which to tensor it is derived.
    - Then do operator overloading for matrix multiplication * , dot product by ^ and addition by -.
    - create .h file and implement all it's functnality in the .cpp file.
    - last setup main.cpp in which we test the all operation.

    -gradient for matrix multiplication

        $A = B*C$
        $B = (3*2)$
        C = (2x2)
        dL/dA (3x2)
        dL/dB = dL/dA * C^T
        dL/dC = B^T * dL/dA

    -gradient for the addition 
        A = B + C
        dL/dB = dL/dA
        dL/dC = dL/dA


