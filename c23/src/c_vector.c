/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/vector.h"
#include "lightlm/matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

lightlm_vector_t* lightlm_vector_new(int64_t size) {
  lightlm_vector_t* vec = (lightlm_vector_t*)malloc(sizeof(lightlm_vector_t));
  if (vec == NULL) {
    return NULL;
  }
  vec->size = size;
  vec->data = (lightlm_real*)malloc(size * sizeof(lightlm_real));
  if (vec->data == NULL) {
    free(vec);
    return NULL;
  }
  return vec;
}

void lightlm_vector_free(lightlm_vector_t* vec) {
  if (vec == NULL) {
    return;
  }
  free(vec->data);
  free(vec);
}

void lightlm_vector_zero(lightlm_vector_t* vec) {
  memset(vec->data, 0, vec->size * sizeof(lightlm_real));
}

void lightlm_vector_mul(lightlm_vector_t* vec, lightlm_real a) {
  for (int64_t i = 0; i < vec->size; i++) {
    vec->data[i] *= a;
  }
}

lightlm_real lightlm_vector_norm(const lightlm_vector_t* vec) {
  lightlm_real sum = 0;
  for (int64_t i = 0; i < vec->size; i++) {
    sum += vec->data[i] * vec->data[i];
  }
  return sqrt(sum);
}

void lightlm_vector_add_vector(lightlm_vector_t* vec, const lightlm_vector_t* source) {
  assert(vec->size == source->size);
  for (int64_t i = 0; i < vec->size; i++) {
    vec->data[i] += source->data[i];
  }
}

void lightlm_vector_add_vector_scaled(lightlm_vector_t* vec, const lightlm_vector_t* source, lightlm_real s) {
  assert(vec->size == source->size);
  for (int64_t i = 0; i < vec->size; i++) {
    vec->data[i] += s * source->data[i];
  }
}

int64_t lightlm_vector_argmax(const lightlm_vector_t* vec) {
  lightlm_real max = vec->data[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < vec->size; i++) {
    if (vec->data[i] > max) {
      max = vec->data[i];
      argmax = i;
    }
  }
  return argmax;
}

void lightlm_vector_add_row(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, int64_t i) {
    assert(i >= 0);
    assert(i < A->m_);
    assert(vec->size == A->n_);
    A->add_row_to_vector(A, vec, i);
}

void lightlm_vector_add_row_scaled(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, int64_t i, lightlm_real a) {
    assert(i >= 0);
    assert(i < A->m_);
    assert(vec->size == A->n_);
    A->add_row_to_vector_scaled(A, vec, i, a);
}

void lightlm_vector_mul_matrix_vector(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, const lightlm_vector_t* source) {
    assert(A->m_ == vec->size);
    assert(A->n_ == source->size);
    for (int64_t i = 0; i < vec->size; i++) {
        vec->data[i] = A->dot_row(A, source, i);
    }
}
