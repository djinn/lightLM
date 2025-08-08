/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "lightlm/matrix.h"
#include "lightlm/real.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lightlm_dense_matrix_s {
  lightlm_real* data;
} lightlm_dense_matrix_t;

lightlm_matrix_t* lightlm_dense_matrix_new(int64_t m, int64_t n);
void lightlm_dense_matrix_uniform(lightlm_matrix_t* mat, lightlm_real a, unsigned int thread, int32_t seed);
void lightlm_dense_matrix_zero(lightlm_matrix_t* mat);

#ifdef __cplusplus
}
#endif
