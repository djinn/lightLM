#pragma once

#include "lightlm/real.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    lightlm_real score;
    int32_t label;
} prediction_t;

typedef struct {
    prediction_t* data;
    int size;
    int capacity;
} predictions_t;

void predictions_init(predictions_t* p, int capacity);
void predictions_push(predictions_t* p, lightlm_real score, int32_t label);
void predictions_free(predictions_t* p);
int compare_predictions(const void* a, const void* b);

#ifdef __cplusplus
}
#endif
