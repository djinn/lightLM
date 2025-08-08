/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/densematrix.h"
#include "lightlm/vector.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // For FILE

// ==========================================================================
// Dense Matrix private implementation
// ==========================================================================

static lightlm_real dense_matrix_dot_row(const lightlm_matrix_t* mat, const lightlm_vector_t* vec, int64_t i) {
    assert(i >= 0);
    assert(i < mat->m_);
    assert(vec->size == mat->n_);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    lightlm_real d = 0.0;
    for (int64_t j = 0; j < mat->n_; j++) {
        d += dm->data[i * mat->n_ + j] * vec->data[j];
    }
    // TODO: NaN check
    return d;
}

static void dense_matrix_add_vector_to_row(const lightlm_matrix_t* mat, const lightlm_vector_t* vec, int64_t i, lightlm_real a) {
    assert(i >= 0);
    assert(i < mat->m_);
    assert(vec->size == mat->n_);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    for (int64_t j = 0; j < mat->n_; j++) {
        dm->data[i * mat->n_ + j] += a * vec->data[j];
    }
}

static void dense_matrix_add_row_to_vector(const lightlm_matrix_t* mat, lightlm_vector_t* x, int32_t i) {
    assert(i >= 0);
    assert(i < mat->m_);
    assert(x->size == mat->n_);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    for (int64_t j = 0; j < mat->n_; j++) {
        x->data[j] += dm->data[i * mat->n_ + j];
    }
}

static void dense_matrix_add_row_to_vector_scaled(const lightlm_matrix_t* mat, lightlm_vector_t* x, int32_t i, lightlm_real a) {
    assert(i >= 0);
    assert(i < mat->m_);
    assert(x->size == mat->n_);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    for (int64_t j = 0; j < mat->n_; j++) {
        x->data[j] += a * dm->data[i * mat->n_ + j];
    }
}

static void dense_matrix_save(const lightlm_matrix_t* mat, void* out_stream) {
    FILE* out = (FILE*)out_stream;
    fwrite(&mat->m_, sizeof(int64_t), 1, out);
    fwrite(&mat->n_, sizeof(int64_t), 1, out);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    fwrite(dm->data, sizeof(lightlm_real), mat->m_ * mat->n_, out);
}

static void dense_matrix_load(lightlm_matrix_t* mat, void* in_stream) {
    FILE* in = (FILE*)in_stream;
    fread(&mat->m_, sizeof(int64_t), 1, in);
    fread(&mat->n_, sizeof(int64_t), 1, in);
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    dm->data = (lightlm_real*)realloc(dm->data, mat->m_ * mat->n_ * sizeof(lightlm_real));
    fread(dm->data, sizeof(lightlm_real), mat->m_ * mat->n_, in);
}

static void dense_matrix_dump(const lightlm_matrix_t* mat, void* out_stream) {
    // Not implemented yet
}

static void dense_matrix_free(lightlm_matrix_t* mat) {
    if (mat == NULL) {
        return;
    }
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    free(dm->data);
    free(dm);
    free(mat);
}

void lightlm_dense_matrix_zero(lightlm_matrix_t* mat) {
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;
    memset(dm->data, 0, mat->m_ * mat->n_ * sizeof(lightlm_real));
}


// ==========================================================================
// Dense Matrix public API
// ==========================================================================

lightlm_matrix_t* lightlm_dense_matrix_new(int64_t m, int64_t n) {
    lightlm_matrix_t* mat = (lightlm_matrix_t*)malloc(sizeof(lightlm_matrix_t));
    if (mat == NULL) {
        return NULL;
    }

    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)malloc(sizeof(lightlm_dense_matrix_t));
    if (dm == NULL) {
        free(mat);
        return NULL;
    }

    dm->data = (lightlm_real*)malloc(m * n * sizeof(lightlm_real));
    if (dm->data == NULL) {
        free(dm);
        free(mat);
        return NULL;
    }

    mat->m_ = m;
    mat->n_ = n;
    mat->data = dm;

    mat->dot_row = dense_matrix_dot_row;
    mat->add_vector_to_row = dense_matrix_add_vector_to_row;
    mat->add_row_to_vector = dense_matrix_add_row_to_vector;
    mat->add_row_to_vector_scaled = dense_matrix_add_row_to_vector_scaled;
    mat->save = dense_matrix_save;
    mat->load = dense_matrix_load;
    mat->dump = dense_matrix_dump;
    mat->free = dense_matrix_free;

    return mat;
}
