all: brouchet
	
brouchet : brouchet.c
	mpicc -fopenmp -Wall -o brouchet brouchet.c