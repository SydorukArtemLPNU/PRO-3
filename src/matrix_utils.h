#pragma once

#include <stddef.h>

typedef struct {
  int rows;
  int cols;
  double* data; // row-major, rows*cols
} Matrix;

Matrix mat_alloc(int rows, int cols);
void mat_free(Matrix* m);
void mat_fill_random(Matrix m, unsigned int seed, int max_val_inclusive);
void mat_fill_ones(Matrix m);

static inline double mat_get(Matrix m, int r, int c) { return m.data[(size_t)r * (size_t)m.cols + (size_t)c]; }
static inline void mat_set(Matrix m, int r, int c, double v) { m.data[(size_t)r * (size_t)m.cols + (size_t)c] = v; }

void mat_mul(Matrix C, Matrix A, Matrix B); // C = A*B
void mat_write_tsv(const char* path, const char* title, Matrix m);

