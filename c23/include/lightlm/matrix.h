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

// Forward declarations
typedef struct lightlm_vector_s lightlm_vector_t;
typedef struct lightlm_matrix_s lightlm_matrix_t;

// Function pointer types for the matrix operations
typedef lightlm_real (*lightlm_matrix_dot_row_t)(const lightlm_matrix_t*, const lightlm_vector_t*, int64_t);
typedef void (*lightlm_matrix_add_vector_to_row_t)(const lightlm_matrix_t*, const lightlm_vector_t*, int64_t, lightlm_real);
typedef void (*lightlm_matrix_add_row_to_vector_t)(const lightlm_matrix_t*, lightlm_vector_t*, int32_t);
typedef void (*lightlm_matrix_add_row_to_vector_scaled_t)(const lightlm_matrix_t*, lightlm_vector_t*, int32_t, lightlm_real);
typedef void (*lightlm_matrix_save_t)(const lightlm_matrix_t*, void*); // void* for FILE* or ostream
typedef void (*lightlm_matrix_load_t)(lightlm_matrix_t*, void*); // void* for FILE* or istream
typedef void (*lightlm_matrix_dump_t)(const lightlm_matrix_t*, void*); // void* for FILE* or ostream
typedef void (*lightlm_matrix_free_t)(lightlm_matrix_t*);

struct lightlm_matrix_s {
    int64_t m_;
    int64_t n_;
    void* data; // Pointer to the actual matrix data (e.g., dense or quant)

    lightlm_matrix_dot_row_t dot_row;
    lightlm_matrix_add_vector_to_row_t add_vector_to_row;
    lightlm_matrix_add_row_to_vector_t add_row_to_vector;
    lightlm_matrix_add_row_to_vector_scaled_t add_row_to_vector_scaled;
    lightlm_matrix_save_t save;
    lightlm_matrix_load_t load;
    lightlm_matrix_dump_t dump;
    lightlm_matrix_free_t free;
};

#ifdef __cplusplus
}
#endif
