/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include "lightlm/real.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct lightlm_matrix_s;

typedef struct lightlm_vector_s {
  int64_t size;
  lightlm_real* data;
} lightlm_vector_t;

lightlm_vector_t* lightlm_vector_new(int64_t size);
void lightlm_vector_free(lightlm_vector_t* vec);
void lightlm_vector_zero(lightlm_vector_t* vec);
void lightlm_vector_mul(lightlm_vector_t* vec, lightlm_real a);
lightlm_real lightlm_vector_norm(const lightlm_vector_t* vec);
void lightlm_vector_add_vector(lightlm_vector_t* vec, const lightlm_vector_t* source);
void lightlm_vector_add_vector_scaled(lightlm_vector_t* vec, const lightlm_vector_t* source, lightlm_real s);
void lightlm_vector_add_row(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, int64_t i);
void lightlm_vector_add_row_scaled(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, int64_t i, lightlm_real a);
void lightlm_vector_mul_matrix_vector(lightlm_vector_t* vec, const struct lightlm_matrix_s* A, const lightlm_vector_t* source);
int64_t lightlm_vector_argmax(const lightlm_vector_t* vec);

#ifdef __cplusplus
}
#endif
