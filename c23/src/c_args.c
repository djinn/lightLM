/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/args.h"

#include <stdlib.h>
#include <string.h>

lightlm_args_t* lightlm_args_new() {
  lightlm_args_t* args = (lightlm_args_t*)malloc(sizeof(lightlm_args_t));
  if (args == NULL) {
    return NULL;
  }

    args->input = NULL;
  args->output = NULL;
  args->lr = 0.05;
  args->lrUpdateRate = 100;
  args->dim = 100;
  args->ws = 5;
  args->epoch = 5;
    args->minCount = 5;
    args->minCountLabel = 0;
  args->neg = 5;
  args->wordNgrams = 1;
  args->loss = lightlm_loss_name_ns;
  args->model = lightlm_model_name_sg;
  args->bucket = 2000000;
  args->minn = 3;
  args->maxn = 6;
  args->thread = 12;
  args->t = 1e-4;
  args->label = strdup("__label__");
  args->verbose = 2;
  args->pretrainedVectors = strdup("");
  args->saveOutput = 0;
  args->seed = 0;

  args->qout = 0;
  args->retrain = 0;
  args->qnorm = 0;
  args->cutoff = 0;
  args->dsub = 2;

  args->autotuneValidationFile = strdup("");
  args->autotuneMetric = strdup("f1");
  args->autotunePredictions = 1;
  args->autotuneDuration = 60 * 5; // 5 minutes
  args->autotuneModelSize = strdup("");

  return args;
}

void lightlm_args_free(lightlm_args_t* args) {
  if (args == NULL) {
    return;
  }
  free(args->input);
  free(args->output);
  free(args->label);
  free(args->pretrainedVectors);
  free(args->autotuneValidationFile);
  free(args->autotuneMetric);
  free(args->autotuneModelSize);
  free(args);
}

lightlm_args_t* lightlm_args_copy(const lightlm_args_t* src) {
    lightlm_args_t* dest = (lightlm_args_t*)malloc(sizeof(lightlm_args_t));
    if (dest == NULL) return NULL;
    *dest = *src; // shallow copy
    // deep copy the strings
    if (src->input) dest->input = strdup(src->input);
    if (src->output) dest->output = strdup(src->output);
    if (src->label) dest->label = strdup(src->label);
    if (src->pretrainedVectors) dest->pretrainedVectors = strdup(src->pretrainedVectors);
    if (src->autotuneValidationFile) dest->autotuneValidationFile = strdup(src->autotuneValidationFile);
    if (src->autotuneMetric) dest->autotuneMetric = strdup(src->autotuneMetric);
    if (src->autotuneModelSize) dest->autotuneModelSize = strdup(src->autotuneModelSize);
    return dest;
}
