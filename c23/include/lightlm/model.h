/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "lightlm/vector.h"
#include <stdint.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct lightlm_model_s lightlm_model_t;
struct lightlm_loss_s;
struct lightlm_matrix_s;

typedef struct {
    lightlm_vector_t* hidden;
    lightlm_vector_t* output;
    lightlm_vector_t* grad;
    // TODO: add rng
    double lossValue_;
    int64_t nexamples_;
} lightlm_model_state_t;

lightlm_model_state_t* lightlm_model_state_new(int32_t hiddenSize, int32_t outputSize, int32_t seed);
void lightlm_model_state_free(lightlm_model_state_t* state);
double lightlm_model_state_get_loss(const lightlm_model_state_t* state);
void lightlm_model_state_increment_n_examples(lightlm_model_state_t* state, double loss);


typedef struct lightlm_model_s {
    struct lightlm_matrix_s* wi_;
    struct lightlm_matrix_s* wo_;
    struct lightlm_loss_s* loss_;
    int normalizeGradient_;
} lightlm_model_t;

lightlm_model_t* lightlm_model_new(
    struct lightlm_matrix_s* wi,
    struct lightlm_matrix_s* wo,
    struct lightlm_loss_s* loss,
    int normalizeGradient);
void lightlm_model_free(lightlm_model_t* model);

void lightlm_model_update(
    lightlm_model_t* model,
    const int32_t* input,
    int input_size,
    const int32_t* targets,
    int targets_size,
    int32_t targetIndex,
    lightlm_real lr,
    lightlm_model_state_t* state);


#ifdef __cplusplus
}
#endif
