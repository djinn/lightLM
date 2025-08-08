#include "lightlm/lightlm.h"
#include "lightlm/densematrix.h"
#include "lightlm/loss.h"
#include "lightlm/dynamic_array.h"
#include "lightlm/meter.h"
#include "lightlm/quantmatrix.h"
#include "lightlm/predictions.h"
#include <stdio.h>
#include <stdlib.h>

static struct lightlm_matrix_s* create_random_matrix(lightlm_dictionary_t* dict, lightlm_args_t* args) {
    int64_t m = dict->nwords_ + args->bucket;
    int64_t n = args->dim;
    lightlm_matrix_t* mat = lightlm_dense_matrix_new(m, n);
    // In the C++ code, this is where uniform initialization happens.
    // I will add this later. For now, it's a zeroed matrix.
    lightlm_dense_matrix_zero(mat);
    return (struct lightlm_matrix_s*)mat;
}

static struct lightlm_matrix_s* create_train_output_matrix(lightlm_dictionary_t* dict, lightlm_args_t* args) {
    int64_t m = (args->model == lightlm_model_name_sup) ? dict->nlabels_ : dict->nwords_;
    int64_t n = args->dim;
    lightlm_matrix_t* mat = lightlm_dense_matrix_new(m, n);
    lightlm_dense_matrix_zero(mat);
    return (struct lightlm_matrix_s*)mat;
}

static struct lightlm_loss_s* create_loss(struct lightlm_matrix_s* output, lightlm_dictionary_t* dict, lightlm_args_t* args) {
    // For now, only softmax loss is supported
    if (args->loss == lightlm_loss_name_softmax) {
        return (struct lightlm_loss_s*)lightlm_softmax_loss_new((lightlm_matrix_t*)output);
    }
    // TODO: add other loss functions
    return NULL;
}

static void train_thread(lightlm_t* h, int32_t threadId) {
    FILE* ifs = fopen(h->args->input, "r");
    // TODO: seek to the correct position for the thread

    lightlm_model_state_t* state = lightlm_model_state_new(h->args->dim, ((lightlm_matrix_t*)h->output)->m_, threadId + h->args->seed);

    int64_t ntokens = h->dict->ntokens_;
    int64_t localTokenCount = 0;

    dynamic_array_t line, labels;
    dynamic_array_init(&line, 1024);
    dynamic_array_init(&labels, 10);

    for (int i = 0; i < h->args->epoch; i++) {
        // TODO: add progress and loss reporting
        fseek(ifs, 0, SEEK_SET); // Reset file for each epoch
        while (lightlm_dictionary_get_line(h->dict, ifs, (void*)&line, (void*)&labels) > 0) {
            localTokenCount += line.size + labels.size;
            if (h->args->model == lightlm_model_name_sup) {
                if (labels.size > 0 && line.size > 0) {
                    // For now, just train on the first label
                    lightlm_model_update(h->model, line.data, line.size, labels.data, labels.size, 0, h->args->lr, state);
                }
            }
            // TODO: add cbow and skipgram
        }
    }

    dynamic_array_free(&line);
    dynamic_array_free(&labels);
    lightlm_model_state_free(state);
    fclose(ifs);
}


void lightlm_train(lightlm_t* h, lightlm_args_t* args) {
    h->args = lightlm_args_copy(args);
    h->dict = lightlm_dictionary_new(h->args);

    FILE* ifs = fopen(args->input, "r");
    if (ifs == NULL) {
        // TODO: error handling
        return;
    }
    lightlm_dictionary_read_from_file(h->dict, args->input);
    fclose(ifs);

    h->input = create_random_matrix(h->dict, args);
    h->output = create_train_output_matrix(h->dict, args);
    h->loss = create_loss(h->output, h->dict, args);
    h->model = lightlm_model_new((lightlm_matrix_t*)h->input, (lightlm_matrix_t*)h->output, (lightlm_loss_t*)h->loss, args->model == lightlm_model_name_sup);

    train_thread(h, 0); // Single-threaded for now
}

lightlm_t* lightlm_new() {
    lightlm_t* h = (lightlm_t*)malloc(sizeof(lightlm_t));
    if (h == NULL) return NULL;
    h->args = NULL;
    h->dict = NULL;
    h->input = NULL;
    h->output = NULL;
    h->model = NULL;
    return h;
}

void lightlm_free(lightlm_t* h) {
    if (h == NULL) return;
    lightlm_args_free(h->args);
    lightlm_dictionary_free(h->dict);
    if (h->input) h->input->free(h->input);
    if (h->output) h->output->free(h->output);
    if (h->loss) h->loss->free(h->loss);
    lightlm_model_free(h->model);
    free(h);
}

void lightlm_test(lightlm_t* h, const char* filename, int32_t k, lightlm_real threshold) {
    FILE* ifs = fopen(filename, "r");
    if (ifs == NULL) {
        // TODO: error handling
        return;
    }

    lightlm_meter_t* meter = lightlm_meter_new(1, h->dict->nlabels_);
    lightlm_model_state_t* state = lightlm_model_state_new(h->args->dim, h->dict->nlabels_, 0);

    dynamic_array_t line, labels;
    dynamic_array_init(&line, 1024);
    dynamic_array_init(&labels, 10);

    predictions_t predictions;
    predictions_init(&predictions, k);

    while (lightlm_dictionary_get_line(h->dict, ifs, (void*)&line, (void*)&labels) > 0) {
        if (labels.size > 0 && line.size > 0) {
            predictions.size = 0;
            lightlm_model_predict(h->model, line.data, line.size, k, threshold, (void*)&predictions, state);
            lightlm_meter_log(meter, labels.data, labels.size, (void*)predictions.data, predictions.size);
        }
    }

    // TODO: print metrics
    printf("Precision: %f\n", lightlm_meter_precision(meter, -1));
    printf("Recall: %f\n", lightlm_meter_recall(meter, -1));
    printf("F1-Score: %f\n", lightlm_meter_f1_score(meter, -1));


    predictions_free(&predictions);
    dynamic_array_free(&line);
    dynamic_array_free(&labels);
    lightlm_model_state_free(state);
    lightlm_meter_free(meter);
    fclose(ifs);
}

void lightlm_quantize(lightlm_t* h, lightlm_args_t* qargs) {
    if (h->args->model != lightlm_model_name_sup) {
        // For now we only support quantization of supervised models
        return;
    }

    // For now, we only support quantizing the input matrix
    lightlm_matrix_t* qmat = lightlm_quant_matrix_new((lightlm_matrix_t*)h->input, qargs->dsub, qargs->qnorm);

    // free the old input matrix
    h->input->free(h->input);
    h->input = (struct lightlm_matrix_s*)qmat;

    // rebuild the model with the new quantized matrix
    lightlm_model_free(h->model);
    h->model = lightlm_model_new((lightlm_matrix_t*)h->input, (lightlm_matrix_t*)h->output, (lightlm_loss_t*)h->loss, h->args->model == lightlm_model_name_sup);
}
