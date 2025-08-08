/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  lightlm_model_name_cbow = 1,
  lightlm_model_name_sg,
  lightlm_model_name_sup
} lightlm_model_name;

typedef enum {
  lightlm_loss_name_hs = 1,
  lightlm_loss_name_ns,
  lightlm_loss_name_softmax,
  lightlm_loss_name_ova
} lightlm_loss_name;

typedef enum {
  lightlm_metric_name_f1score = 1,
  lightlm_metric_name_f1scoreLabel,
  lightlm_metric_name_precisionAtRecall,
  lightlm_metric_name_precisionAtRecallLabel,
  lightlm_metric_name_recallAtPrecision,
  lightlm_metric_name_recallAtPrecisionLabel
} lightlm_metric_name;

typedef struct lightlm_args_s {
  char* input;
  char* output;
  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  lightlm_loss_name loss;
  lightlm_model_name model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  char* label;
  int verbose;
  char* pretrainedVectors;
  int saveOutput;
  int seed;

  int qout;
  int retrain;
  int qnorm;
  size_t cutoff;
  size_t dsub;

  char* autotuneValidationFile;
  char* autotuneMetric;
  int autotunePredictions;
  int autotuneDuration;
  char* autotuneModelSize;
} lightlm_args_t;

lightlm_args_t* lightlm_args_new();
void lightlm_args_free(lightlm_args_t* args);
lightlm_args_t* lightlm_args_copy(const lightlm_args_t* src);

#ifdef __cplusplus
}
#endif
