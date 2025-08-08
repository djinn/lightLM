/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/loss.h"
#include "lightlm/model.h"
#include "lightlm/vector.h"
#include "lightlm/predictions.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 512

typedef struct {
    lightlm_real* t_sigmoid_;
    lightlm_real* t_log_;
} lightlm_loss_base_t;

static void init_loss_tables(lightlm_loss_base_t* base) {
    base->t_sigmoid_ = (lightlm_real*)malloc((SIGMOID_TABLE_SIZE + 1) * sizeof(lightlm_real));
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
        lightlm_real x = (lightlm_real)(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        base->t_sigmoid_[i] = 1.0 / (1.0 + exp(-x));
    }

    base->t_log_ = (lightlm_real*)malloc((LOG_TABLE_SIZE + 1) * sizeof(lightlm_real));
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
        lightlm_real x = ((lightlm_real)(i) + 1e-5) / LOG_TABLE_SIZE;
        base->t_log_[i] = log(x);
    }
}

static void free_loss_tables(lightlm_loss_base_t* base) {
    free(base->t_sigmoid_);
    free(base->t_log_);
}

static lightlm_real loss_log(const lightlm_loss_base_t* base, lightlm_real x) {
    if (x > 1.0) {
        return 0.0;
    }
    int64_t i = (int64_t)(x * LOG_TABLE_SIZE);
    return base->t_log_[i];
}

static lightlm_real loss_sigmoid(const lightlm_loss_base_t* base, lightlm_real x) {
    if (x < -MAX_SIGMOID) {
        return 0.0;
    } else if (x > MAX_SIGMOID) {
        return 1.0;
    } else {
        int64_t i = (int64_t)((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
        return base->t_sigmoid_[i];
    }
}

static void findKBest(lightlm_loss_t* loss, int32_t k, lightlm_real threshold, predictions_t* heap, const lightlm_vector_t* output) {
    for (int32_t i = 0; i < output->size; i++) {
        if (output->data[i] < threshold) {
            continue;
        }
        if (heap->size == k && output->data[i] < heap->data[0].score) {
            continue;
        }
        predictions_push(heap, output->data[i], i);
        qsort(heap->data, heap->size, sizeof(prediction_t), compare_predictions);
        if (heap->size > k) {
            heap->size--;
        }
    }
}


// ==========================================================================
// SoftmaxLoss
// ==========================================================================

typedef struct {
    lightlm_loss_base_t base;
} lightlm_softmax_loss_t;

static void softmax_loss_compute_output(lightlm_loss_t* loss, struct lightlm_model_state_s* state) {
    lightlm_vector_t* output = ((lightlm_model_state_t*)state)->output;
    lightlm_vector_mul_matrix_vector(output, loss->wo_, ((lightlm_model_state_t*)state)->hidden);

    lightlm_real max = output->data[0];
    lightlm_real z = 0.0;
    for (int32_t i = 0; i < output->size; i++) {
        if (output->data[i] > max) {
            max = output->data[i];
        }
    }
    for (int32_t i = 0; i < output->size; i++) {
        output->data[i] = exp(output->data[i] - max);
        z += output->data[i];
    }
    for (int32_t i = 0; i < output->size; i++) {
        output->data[i] /= z;
    }
}

static lightlm_real softmax_loss_forward(
    lightlm_loss_t* loss,
    const int32_t* targets,
    int targets_size,
    int32_t targetIndex,
    struct lightlm_model_state_s* state,
    lightlm_real lr,
    int backprop
) {
    softmax_loss_compute_output(loss, state);

    assert(targetIndex >= 0);
    assert(targetIndex < targets_size);
    int32_t target = targets[targetIndex];

    if (backprop) {
        int32_t osz = loss->wo_->m_;
        lightlm_model_state_t* state_ptr = (lightlm_model_state_t*)state;
        for (int32_t i = 0; i < osz; i++) {
            lightlm_real label = (i == target) ? 1.0 : 0.0;
            lightlm_real alpha = lr * (label - state_ptr->output->data[i]);
            lightlm_vector_add_row_scaled(state_ptr->grad, loss->wo_, i, alpha);
            loss->wo_->add_vector_to_row(loss->wo_, state_ptr->hidden, i, alpha);
        }
    }
    lightlm_softmax_loss_t* sl = (lightlm_softmax_loss_t*)loss->data;
    return -loss_log(&sl->base, ((lightlm_model_state_t*)state)->output->data[target]);
}

static void softmax_loss_predict(lightlm_loss_t* loss, int32_t k, lightlm_real threshold, void* predictions, struct lightlm_model_state_s* state) {
    predictions_t* heap = (predictions_t*)predictions;
    softmax_loss_compute_output(loss, state);
    findKBest(loss, k, threshold, heap, ((lightlm_model_state_t*)state)->output);
    qsort(heap->data, heap->size, sizeof(prediction_t), compare_predictions);
}


static void softmax_loss_free(lightlm_loss_t* loss) {
    if (loss == NULL) return;
    lightlm_softmax_loss_t* sl = (lightlm_softmax_loss_t*)loss->data;
    free_loss_tables(&sl->base);
    free(sl);
    free(loss);
}

lightlm_loss_t* lightlm_softmax_loss_new(lightlm_matrix_t* wo) {
    lightlm_loss_t* loss = (lightlm_loss_t*)malloc(sizeof(lightlm_loss_t));
    if (loss == NULL) return NULL;

    lightlm_softmax_loss_t* sl = (lightlm_softmax_loss_t*)malloc(sizeof(lightlm_softmax_loss_t));
    if (sl == NULL) {
        free(loss);
        return NULL;
    }

    init_loss_tables(&sl->base);
    loss->wo_ = wo;
    loss->data = sl;
    loss->forward = softmax_loss_forward;
    loss->compute_output = softmax_loss_compute_output;
    loss->predict = softmax_loss_predict;
    loss->free = softmax_loss_free;

    return loss;
}
