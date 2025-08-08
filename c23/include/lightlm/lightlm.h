#pragma once

#include "lightlm/args.h"
#include "lightlm/dictionary.h"
#include "lightlm/model.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lightlm_matrix_s;
struct lightlm_loss_s;

typedef struct {
    lightlm_args_t* args;
    lightlm_dictionary_t* dict;
    struct lightlm_matrix_s* input;
    struct lightlm_matrix_s* output;
    struct lightlm_loss_s* loss;
    lightlm_model_t* model;
} lightlm_t;

lightlm_t* lightlm_new();
void lightlm_free(lightlm_t* h);
void lightlm_train(lightlm_t* h, lightlm_args_t* args);
void lightlm_quantize(lightlm_t* h, lightlm_args_t* qargs);
void lightlm_test(lightlm_t* h, const char* filename, int32_t k, lightlm_real threshold);

#ifdef __cplusplus
}
#endif
