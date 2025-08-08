#include "lightlm/quantmatrix.h"
#include "lightlm/productquantizer.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    lightlm_product_quantizer_t* pq_;
    lightlm_product_quantizer_t* npq_;
    uint8_t* codes_;
    uint8_t* norm_codes_;
    int qnorm_;
    int32_t codesize_;
} lightlm_quant_matrix_t;

static lightlm_real quant_matrix_dot_row(const lightlm_matrix_t* mat, const lightlm_vector_t* vec, int64_t i) {
    const lightlm_quant_matrix_t* qm = (const lightlm_quant_matrix_t*)mat->data;
    lightlm_real norm = 1.0;
    // TODO: norm quantization
    return lightlm_product_quantizer_mulcode(qm->pq_, vec, qm->codes_, i, norm);
}

static void quant_matrix_add_row_to_vector(const lightlm_matrix_t* mat, lightlm_vector_t* x, int32_t i, lightlm_real a) {
    const lightlm_quant_matrix_t* qm = (const lightlm_quant_matrix_t*)mat->data;
    lightlm_real norm = 1.0;
    // TODO: norm quantization
    lightlm_product_quantizer_addcode(qm->pq_, x, qm->codes_, i, a * norm);
}

static void quant_matrix_free(lightlm_matrix_t* mat) {
    if (mat == NULL) return;
    lightlm_quant_matrix_t* qm = (lightlm_quant_matrix_t*)mat->data;
    lightlm_product_quantizer_free(qm->pq_);
    lightlm_product_quantizer_free(qm->npq_);
    free(qm->codes_);
    free(qm->norm_codes_);
    free(qm);
    free(mat);
}


lightlm_matrix_t* lightlm_quant_matrix_new(lightlm_matrix_t* mat, int32_t dsub, int qnorm) {
    lightlm_dense_matrix_t* dm = (lightlm_dense_matrix_t*)mat->data;

    lightlm_quant_matrix_t* qm = (lightlm_quant_matrix_t*)malloc(sizeof(lightlm_quant_matrix_t));
    if (qm == NULL) return NULL;

    qm->qnorm_ = qnorm;
    qm->codesize_ = mat->m_ * ((mat->n_ + dsub - 1) / dsub);
    qm->codes_ = (uint8_t*)malloc(qm->codesize_ * sizeof(uint8_t));
    qm->pq_ = lightlm_product_quantizer_new(mat->n_, dsub);
    if (qnorm) {
        qm->norm_codes_ = (uint8_t*)malloc(mat->m_ * sizeof(uint8_t));
        qm->npq_ = lightlm_product_quantizer_new(1, 1);
    } else {
        qm->norm_codes_ = NULL;
        qm->npq_ = NULL;
    }

    // TODO: norm quantization

    lightlm_product_quantizer_train(qm->pq_, mat->m_, dm->data);
    lightlm_product_quantizer_compute_codes(qm->pq_, dm->data, qm->codes_, mat->m_);

    lightlm_matrix_t* qmat = (lightlm_matrix_t*)malloc(sizeof(lightlm_matrix_t));
    if (qmat == NULL) {
        // TODO: free qm
        return NULL;
    }

    qmat->m_ = mat->m_;
    qmat->n_ = mat->n_;
    qmat->data = qm;
    qmat->dot_row = quant_matrix_dot_row;
    qmat->add_row_to_vector_scaled = quant_matrix_add_row_to_vector;
    qmat->free = quant_matrix_free;
    // Other function pointers are not supported for quantized matrix

    return qmat;
}
