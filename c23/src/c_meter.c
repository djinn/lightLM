#include "lightlm/meter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Assuming Predictions is an array of pairs of (real, int32_t)
typedef struct {
    lightlm_real first;
    int32_t second;
} lightlm_prediction_t;


lightlm_meter_t* lightlm_meter_new(int falseNegativeLabels, int32_t nlabels) {
    lightlm_meter_t* meter = (lightlm_meter_t*)malloc(sizeof(lightlm_meter_t));
    if (meter == NULL) return NULL;

    meter->metrics_.gold = 0;
    meter->metrics_.predicted = 0;
    meter->metrics_.predictedGold = 0;
    meter->nexamples_ = 0;
    meter->nlabels_ = nlabels;
    meter->falseNegativeLabels_ = falseNegativeLabels;

    meter->labelMetrics_ = (lightlm_label_metrics_t*)calloc(nlabels, sizeof(lightlm_label_metrics_t));
    if (meter->labelMetrics_ == NULL) {
        free(meter);
        return NULL;
    }
    for (int i = 0; i < nlabels; i++) {
        meter->labelMetrics_[i].label_id = i;
    }

    return meter;
}

void lightlm_meter_free(lightlm_meter_t* meter) {
    if (meter == NULL) return;
    free(meter->labelMetrics_);
    free(meter);
}

static int contains(const int32_t* array, int size, int32_t value) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) {
            return 1;
        }
    }
    return 0;
}

void lightlm_meter_log(lightlm_meter_t* meter, const int32_t* labels, int nlabels, const void* predictions, int npredictions) {
    meter->nexamples_++;
    meter->metrics_.gold += nlabels;
    meter->metrics_.predicted += npredictions;

    const lightlm_prediction_t* preds = (const lightlm_prediction_t*)predictions;

    for (int i = 0; i < npredictions; i++) {
        int32_t label_id = preds[i].second;
        if (label_id >= 0 && label_id < meter->nlabels_) {
            meter->labelMetrics_[label_id].metrics.predicted++;
            if (contains(labels, nlabels, label_id)) {
                meter->labelMetrics_[label_id].metrics.predictedGold++;
                meter->metrics_.predictedGold++;
            }
        }
    }

    if (meter->falseNegativeLabels_) {
        for (int i = 0; i < nlabels; i++) {
            int32_t label_id = labels[i];
            if (label_id >= 0 && label_id < meter->nlabels_) {
                 meter->labelMetrics_[label_id].metrics.gold++;
            }
        }
    }
}

static double precision(const lightlm_metrics_t* m) {
    if (m->predicted == 0) return NAN;
    return (double)m->predictedGold / m->predicted;
}

static double recall(const lightlm_metrics_t* m) {
    if (m->gold == 0) return NAN;
    return (double)m->predictedGold / m->gold;
}

static double f1_score(const lightlm_metrics_t* m) {
    if (m->predicted + m->gold == 0) return NAN;
    double p = precision(m);
    double r = recall(m);
    if (isnan(p) || isnan(r)) return NAN;
    return 2 * p * r / (p + r);
}


double lightlm_meter_precision(const lightlm_meter_t* meter, int32_t labelId) {
    if (labelId < 0 || labelId >= meter->nlabels_) return NAN;
    return precision(&meter->labelMetrics_[labelId].metrics);
}

double lightlm_meter_recall(const lightlm_meter_t* meter, int32_t labelId) {
    if (labelId < 0 || labelId >= meter->nlabels_) return NAN;
    return recall(&meter->labelMetrics_[labelId].metrics);
}

double lightlm_meter_f1_score(const lightlm_meter_t* meter, int32_t labelId) {
    if (labelId < 0 || labelId >= meter->nlabels_) return NAN;
    return f1_score(&meter->labelMetrics_[labelId].metrics);
}
