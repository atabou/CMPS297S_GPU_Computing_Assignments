
## CMPS 297S/396AA: GPU COMPUTING

### ASSIGNMENT 1

In this assignment, you will write a CUDA program that computes the elementwise maximum of two vectors of double precision values. That is, you will compute a vector c from two vectors a and b such that c[i] is the maximum of a[i] and b[i] for all i.

**Instructions**
  
  
1. Place the files provided with this assignment in a single directory. The files are:

    - main.cu: contains setup and sequential code

    - kernel.cu: where you will implement your code (you should only modify this file)
    - common.h: for shared declarations across main.cu and kernel.cu
    - timer.h: to assist with timing
    - Makefile: used for compilation
  
  
2. Edit kernel.cu where TODO is indicated to implement the following:

    - Allocate device memory

    - Copy data from the host to the device
    - Configure and invoke the CUDA kernel
    - Copy the results from the device to the host
    - Free device memory
    - Perform the computation in the kernel
  
  
3. Compile your code by running: make
  
  
4. Test your code by running: ./vecmax

    - If you are using the university cluster, do not forget to use the submission system.  
   Do not run on the head node!
   
    - For testing on different input sizes, you can provide your own value for the number of vector elements: ./vecmax <M>  
   (example: ./vecmax 1000000 )

**Submission**
Submit your modified kernel.cu file via Moodle by the due date. Do not submit any other files or
compressed folders.


