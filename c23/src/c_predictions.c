#include "lightlm/predictions.h"
#include <stdlib.h>

void predictions_init(predictions_t* p, int capacity) {
    p->data = (prediction_t*)malloc(capacity * sizeof(prediction_t));
    p->size = 0;
    p->capacity = capacity;
}

void predictions_push(predictions_t* p, lightlm_real score, int32_t label) {
    if (p->size == p->capacity) {
        p->capacity *= 2;
        p->data = (prediction_t*)realloc(p->data, p->capacity * sizeof(prediction_t));
    }
    p->data[p->size].score = score;
    p->data[p->size].label = label;
    p->size++;
}

void predictions_free(predictions_t* p) {
    free(p->data);
}

int compare_predictions(const void* a, const void* b) {
    const prediction_t* p1 = (const prediction_t*)a;
    const prediction_t* p2 = (const prediction_t*)b;
    if (p1->score > p2->score) return -1;
    if (p1->score < p2->score) return 1;
    return 0;
}
