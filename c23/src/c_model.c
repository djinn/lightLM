/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/model.h"
#include "lightlm/loss.h"
#include "lightlm/vector.h"
#include "lightlm/matrix.h"
#include <assert.h>
#include <stdlib.h>

// ==========================================================================
// Model::State
// ==========================================================================

lightlm_model_state_t* lightlm_model_state_new(int32_t hiddenSize, int32_t outputSize, int32_t seed) {
    lightlm_model_state_t* state = (lightlm_model_state_t*)malloc(sizeof(lightlm_model_state_t));
    if (state == NULL) return NULL;

    state->hidden = lightlm_vector_new(hiddenSize);
    state->output = lightlm_vector_new(outputSize);
    state->grad = lightlm_vector_new(hiddenSize);
    state->lossValue_ = 0.0;
    state->nexamples_ = 0;
    // TODO: init rng
    return state;
}

void lightlm_model_state_free(lightlm_model_state_t* state) {
    if (state == NULL) return;
    lightlm_vector_free(state->hidden);
    lightlm_vector_free(state->output);
    lightlm_vector_free(state->grad);
    free(state);
}

double lightlm_model_state_get_loss(const lightlm_model_state_t* state) {
    return state->lossValue_ / state->nexamples_;
}

void lightlm_model_state_increment_n_examples(lightlm_model_state_t* state, double loss) {
    state->lossValue_ += loss;
    state->nexamples_++;
}


// ==========================================================================
// Model
// ==========================================================================

lightlm_model_t* lightlm_model_new(
    lightlm_matrix_t* wi,
    lightlm_matrix_t* wo,
    lightlm_loss_t* loss,
    int normalizeGradient
) {
    lightlm_model_t* model = (lightlm_model_t*)malloc(sizeof(lightlm_model_t));
    if (model == NULL) return NULL;

    model->wi_ = wi;
    model->wo_ = wo;
    model->loss_ = loss;
    model->normalizeGradient_ = normalizeGradient;
    return model;
}

void lightlm_model_free(lightlm_model_t* model) {
    free(model);
}

static void lightlm_model_compute_hidden(
    const lightlm_model_t* model,
    const int32_t* input,
    int input_size,
    lightlm_model_state_t* state
) {
    lightlm_vector_t* hidden = state->hidden;
    lightlm_vector_zero(hidden);
    for (int i = 0; i < input_size; i++) {
        lightlm_vector_add_row(hidden, model->wi_, input[i]);
    }
    lightlm_vector_mul(hidden, 1.0 / input_size);
}

void lightlm_model_update(
    lightlm_model_t* model,
    const int32_t* input,
    int input_size,
    const int32_t* targets,
    int targets_size,
    int32_t targetIndex,
    lightlm_real lr,
    lightlm_model_state_t* state
) {
    if (input_size == 0) {
        return;
    }
    lightlm_model_compute_hidden(model, input, input_size, state);

    lightlm_vector_t* grad = state->grad;
    lightlm_vector_zero(grad);
    double lossValue = model->loss_->forward(model->loss_, targets, targets_size, targetIndex, state, lr, 1);
    lightlm_model_state_increment_n_examples(state, lossValue);

    if (model->normalizeGradient_) {
        lightlm_vector_mul(grad, 1.0 / input_size);
    }
    for (int i = 0; i < input_size; i++) {
        model->wi_->add_vector_to_row(model->wi_, grad, input[i], 1.0);
    }
}
