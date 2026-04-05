#include "matrix_utils.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void compute_block_sizes(int n, int blocks, int* sizes, int* displs) {
  const int base = n / blocks;
  const int rem = n % blocks;
  int off = 0;
  for (int i = 0; i < blocks; i++) {
    sizes[i] = base + (i < rem ? 1 : 0);
    displs[i] = off;
    off += sizes[i];
  }
}

static void write_rank_file(int rank, const char* label, Matrix m) {
  char path[256];
  snprintf(path, sizeof(path), "out/Results_%d_%s.tsv", rank, label);
  mat_write_tsv(path, label, m);
}

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s [scale] [verbose]              → матриці 87×scale × 319×scale × 213×scale\n"
          "  %s <n1> <n2> <n3> [verbose]       → довільні розміри A[n1][n2], B[n2][n3]\n"
          "  verbose: 1 = виводити обмін (default), 0 = лише час\n",
          prog, prog);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  const int P = 4;
  const int Q = 2;
  const int workers = P * Q;
  if (world != workers + 1) {
    if (rank == 0) {
      fprintf(stderr, "This program requires %d processes (1 master + %d workers)\n", workers + 1, workers);
      fprintf(stderr, "Run: mpirun -np %d ./mpi_grid [args]\n", workers + 1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const int n1_base = 87;
  const int n2_base = 319;
  const int n3_base = 213;

  int n1, n2, n3, verbose, write_files;
  if (rank == 0) {
    if (argc == 1) {
      n1 = n1_base;
      n2 = n2_base;
      n3 = n3_base;
      verbose = 1;
      write_files = 1;
    } else if (argc == 2) {
      int scale = atoi(argv[1]);
      if (scale < 1) {
        fprintf(stderr, "scale має бути >= 1\n");
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      n1 = n1_base * scale;
      n2 = n2_base * scale;
      n3 = n3_base * scale;
      verbose = 1;
      write_files = (scale == 1);
    } else if (argc == 3) {
      int scale = atoi(argv[1]);
      verbose = atoi(argv[2]);
      if (scale < 1) {
        fprintf(stderr, "scale має бути >= 1\n");
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      n1 = n1_base * scale;
      n2 = n2_base * scale;
      n3 = n3_base * scale;
      write_files = (scale == 1);
    } else if (argc >= 4) {
      n1 = atoi(argv[1]);
      n2 = atoi(argv[2]);
      n3 = atoi(argv[3]);
      verbose = (argc >= 5) ? atoi(argv[4]) : 1;
      if (n1 < 1 || n2 < 1 || n3 < 1) {
        fprintf(stderr, "n1, n2, n3 мають бути >= 1\n");
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      write_files = (n1 <= n1_base && n2 <= n2_base && n3 <= n3_base);
    } else {
      usage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int params[4];
  if (rank == 0) {
    params[0] = n1;
    params[1] = n2;
    params[2] = n3;
    params[3] = verbose;
  }
  MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);
  n1 = params[0];
  n2 = params[1];
  n3 = params[2];
  verbose = params[3];
  MPI_Bcast(&write_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int a_sizes[P], a_displs[P];
  int b_sizes[Q], b_displs[Q];
  compute_block_sizes(n1, P, a_sizes, a_displs);
  compute_block_sizes(n3, Q, b_sizes, b_displs);

  // Master holds full matrices.
  Matrix A_full = {0}, B_full = {0}, C_full = {0};

  double t0 = 0.0, t1 = 0.0;
  if (rank == 0) {
    A_full = mat_alloc(n1, n2);
    B_full = mat_alloc(n2, n3);
    C_full = mat_alloc(n1, n3);
    mat_fill_random(A_full, 12345u, 100);
    mat_fill_random(B_full, 67890u, 100);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    // Send stripes.
    for (int i = 0; i < P; i++) {
      for (int j = 0; j < Q; j++) {
        const int wrank = 1 + i * Q + j;
        const int a_rows = a_sizes[i];
        const int b_cols = b_sizes[j];
        int meta[4] = {i, j, a_rows, b_cols};
        MPI_Send(meta, 4, MPI_INT, wrank, 10, MPI_COMM_WORLD);

        if (verbose) {
          printf("I am 0 process. Send A stripe i=%d (%d rows) to %d\n", i, a_rows, wrank);
          printf("I am 0 process. Send B stripe j=%d (%d cols) to %d\n", j, b_cols, wrank);
          fflush(stdout);
        }

        // Send A_block (a_rows x n2) contiguous.
        const double* a_ptr = &A_full.data[(size_t)a_displs[i] * (size_t)n2];
        MPI_Send((void*)a_ptr, a_rows * n2, MPI_DOUBLE, wrank, 11, MPI_COMM_WORLD);

        // Send B_block (n2 x b_cols) as packed contiguous buffer.
        double* b_pack = (double*)malloc((size_t)n2 * (size_t)b_cols * sizeof(double));
        if (!b_pack) {
          fprintf(stderr, "malloc failed for b_pack\n");
          MPI_Abort(MPI_COMM_WORLD, 2);
        }
        for (int r = 0; r < n2; r++) {
          memcpy(&b_pack[(size_t)r * (size_t)b_cols],
                 &B_full.data[(size_t)r * (size_t)n3 + (size_t)b_displs[j]],
                 (size_t)b_cols * sizeof(double));
        }
        MPI_Send(b_pack, n2 * b_cols, MPI_DOUBLE, wrank, 12, MPI_COMM_WORLD);
        free(b_pack);
      }
    }

    // Receive C blocks back and assemble.
    for (int k = 0; k < workers; k++) {
      int meta[4];
      MPI_Recv(meta, 4, MPI_INT, MPI_ANY_SOURCE, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      const int i = meta[0], j = meta[1], a_rows = meta[2], b_cols = meta[3];
      const int src = 1 + i * Q + j;

      if (verbose) {
        printf("I am 0 process. Received C block C%d%d from %d\n", i, j, src);
        fflush(stdout);
      }

      double* c_pack = (double*)malloc((size_t)a_rows * (size_t)b_cols * sizeof(double));
      if (!c_pack) {
        fprintf(stderr, "malloc failed for c_pack\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      MPI_Recv(c_pack, a_rows * b_cols, MPI_DOUBLE, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int r = 0; r < a_rows; r++) {
        memcpy(&C_full.data[(size_t)(a_displs[i] + r) * (size_t)n3 + (size_t)b_displs[j]],
               &c_pack[(size_t)r * (size_t)b_cols],
               (size_t)b_cols * sizeof(double));
      }
      free(c_pack);
    }

    t1 = MPI_Wtime();
    printf("MPI_GRID_4x2 n1=%d n2=%d n3=%d time_ms=%.3f\n", n1, n2, n3, (t1 - t0) * 1000.0);

    if (write_files) {
      mat_write_tsv("out/mpi_A.tsv", "Matrix A", A_full);
      mat_write_tsv("out/mpi_B.tsv", "Matrix B", B_full);
      mat_write_tsv("out/mpi_C.tsv", "Matrix C", C_full);
    }

    mat_free(&A_full);
    mat_free(&B_full);
    mat_free(&C_full);
  } else {
    // Worker.
    MPI_Barrier(MPI_COMM_WORLD);
    int meta[4];
    MPI_Recv(meta, 4, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int I = meta[0];
    const int J = meta[1];
    const int a_rows = meta[2];
    const int b_cols = meta[3];

    if (verbose) {
      printf("I am %d process. Received meta I=%d J=%d\n", rank, I, J);
      fflush(stdout);
    }

    Matrix A_blk = mat_alloc(a_rows, n2);
    Matrix B_blk = mat_alloc(n2, b_cols);
    Matrix C_blk = mat_alloc(a_rows, b_cols);

    MPI_Recv(A_blk.data, a_rows * n2, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (verbose) {
      printf("I am %d process. Received A stripe A%d from 0 process\n", rank, I);
      fflush(stdout);
    }
    MPI_Recv(B_blk.data, n2 * b_cols, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (verbose) {
      printf("I am %d process. Received B stripe B%d from 0 process\n", rank, J);
      fflush(stdout);
    }

    mat_mul(C_blk, A_blk, B_blk);
    if (verbose) {
      printf("I am %d process. Multiplied A%d x B%d = C%d%d\n", rank, I, J, I, J);
      fflush(stdout);
    }

    if (write_files) {
      write_rank_file(rank, "A", A_blk);
      write_rank_file(rank, "B", B_blk);
      write_rank_file(rank, "C", C_blk);
    }

    int out_meta[4] = {I, J, a_rows, b_cols};
    MPI_Send(out_meta, 4, MPI_INT, 0, 20, MPI_COMM_WORLD);
    MPI_Send(C_blk.data, a_rows * b_cols, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
    if (verbose) {
      printf("I am %d process. Sent C%d%d to 0 process\n", rank, I, J);
      fflush(stdout);
    }

    mat_free(&A_blk);
    mat_free(&B_blk);
    mat_free(&C_blk);
  }

  MPI_Finalize();
  return 0;
}

