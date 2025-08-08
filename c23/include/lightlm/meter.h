#pragma once

#include <stdint.h>
#include "lightlm/real.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t gold;
    uint64_t predicted;
    uint64_t predictedGold;
} lightlm_metrics_t;

typedef struct {
    int32_t label_id;
    lightlm_metrics_t metrics;
} lightlm_label_metrics_t;

typedef struct {
    lightlm_metrics_t metrics_;
    uint64_t nexamples_;
    lightlm_label_metrics_t* labelMetrics_;
    int32_t nlabels_;
    int falseNegativeLabels_;
} lightlm_meter_t;

lightlm_meter_t* lightlm_meter_new(int falseNegativeLabels, int32_t nlabels);
void lightlm_meter_free(lightlm_meter_t* meter);
void lightlm_meter_log(lightlm_meter_t* meter, const int32_t* labels, int nlabels, const void* predictions, int npredictions);
double lightlm_meter_precision(const lightlm_meter_t* meter, int32_t labelId);
double lightlm_meter_recall(const lightlm_meter_t* meter, int32_t labelId);
double lightlm_meter_f1_score(const lightlm_meter_t* meter, int32_t labelId);

#ifdef __cplusplus
}
#endif
