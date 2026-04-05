#include "matrix_utils.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix mat_alloc(int rows, int cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.data = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
  if (!m.data) {
    fprintf(stderr, "calloc failed for %dx%d\n", rows, cols);
    exit(1);
  }
  return m;
}

void mat_free(Matrix* m) {
  if (!m) return;
  free(m->data);
  m->data = NULL;
  m->rows = 0;
  m->cols = 0;
}

static unsigned int lcg_next(unsigned int* state) {
  // Simple LCG, deterministic across platforms.
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

void mat_fill_random(Matrix m, unsigned int seed, int max_val_inclusive) {
  unsigned int st = seed ? seed : 1u;
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      unsigned int r = lcg_next(&st);
      double v = (double)(r % (unsigned int)(max_val_inclusive + 1));
      mat_set(m, i, j, v);
    }
  }
}

void mat_fill_ones(Matrix m) {
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      mat_set(m, i, j, 1.0);
    }
  }
}

void mat_mul(Matrix C, Matrix A, Matrix B) {
  // Dimensions must be compatible; no checks for speed in lab.
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < B.cols; j++) {
      double sum = 0.0;
      const double* arow = &A.data[(size_t)i * (size_t)A.cols];
      for (int k = 0; k < A.cols; k++) {
        sum += arow[k] * B.data[(size_t)k * (size_t)B.cols + (size_t)j];
      }
      C.data[(size_t)i * (size_t)C.cols + (size_t)j] = sum;
    }
  }
}

void mat_write_tsv(const char* path, const char* title, Matrix m) {
  FILE* f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Failed to open '%s': %s\n", path, strerror(errno));
    exit(1);
  }
  if (title && title[0] != '\0') fprintf(f, "%s\n", title);
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      fprintf(f, "%.0f", mat_get(m, i, j));
      if (j + 1 < m.cols) fputc('\t', f);
    }
    fputc('\n', f);
  }
  fclose(f);
}

