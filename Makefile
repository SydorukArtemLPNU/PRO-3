.PHONY: all clean

CC ?= cc
MPICC ?= mpicc
CFLAGS ?= -O2 -std=c11 -Walmpi_gridl -Wextra -Wpedantic

all: sequential mpi_grid

sequential: src/sequential.c src/matrix_utils.c src/matrix_utils.h
	$(CC) $(CFLAGS) -o $@ src/sequential.c src/matrix_utils.c

mpi_grid: src/mpi_grid.c src/matrix_utils.c src/matrix_utils.h
	$(MPICC) $(CFLAGS) -o $@ src/mpi_grid.c src/matrix_utils.c

clean:
	rm -f sequential 
	rm -rf out/*