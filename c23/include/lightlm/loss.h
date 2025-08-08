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

// Forward declaration
struct lightlm_model_state_s;

typedef struct lightlm_loss_s lightlm_loss_t;

// Function pointer types for the loss operations
typedef lightlm_real (*lightlm_loss_forward_t)(
    lightlm_loss_t* loss,
    const int32_t* targets,
    int targets_size,
    int32_t targetIndex,
    struct lightlm_model_state_s* state,
    lightlm_real lr,
    int backprop);

typedef void (*lightlm_loss_compute_output_t)(lightlm_loss_t* loss, struct lightlm_model_state_s* state);
typedef void (*lightlm_loss_free_t)(lightlm_loss_t* loss);

struct lightlm_loss_s {
    lightlm_matrix_t* wo_; // output matrix
    void* data; // Pointer to the actual loss data (e.g., for HS or NS)

    lightlm_loss_forward_t forward;
    lightlm_loss_compute_output_t compute_output;
    lightlm_loss_free_t free;
};

// Declarations for the concrete loss functions
lightlm_loss_t* lightlm_softmax_loss_new(lightlm_matrix_t* wo);
lightlm_loss_t* lightlm_ns_loss_new(lightlm_matrix_t* wo, int neg, const int64_t* targetCounts, int ntargets);
// ... and so on for other loss types

#ifdef __cplusplus
}
#endif
