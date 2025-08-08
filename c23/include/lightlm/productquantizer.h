#pragma once

#include <stdint.h>
#include "lightlm/real.h"
#include "lightlm/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t dim_;
    int32_t nsubq_;
    int32_t dsub_;
    int32_t lastdsub_;
    lightlm_real* centroids_;
    int32_t centroids_size_;
} lightlm_product_quantizer_t;

lightlm_product_quantizer_t* lightlm_product_quantizer_new(int32_t dim, int32_t dsub);
void lightlm_product_quantizer_free(lightlm_product_quantizer_t* pq);
void lightlm_product_quantizer_train(lightlm_product_quantizer_t* pq, int n, const lightlm_real* x);
void lightlm_product_quantizer_compute_codes(const lightlm_product_quantizer_t* pq, const lightlm_real* x, uint8_t* codes, int32_t n);
void lightlm_product_quantizer_addcode(const lightlm_product_quantizer_t* pq, lightlm_vector_t* x, const uint8_t* codes, int32_t t, lightlm_real alpha);
lightlm_real lightlm_product_quantizer_mulcode(const lightlm_product_quantizer_t* pq, const lightlm_vector_t* x, const uint8_t* codes, int32_t t, lightlm_real alpha);

#ifdef __cplusplus
}
#endif
