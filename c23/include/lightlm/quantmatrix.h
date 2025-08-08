#pragma once

#include "lightlm/matrix.h"
#include "lightlm/densematrix.h"

#ifdef __cplusplus
extern "C" {
#endif

lightlm_matrix_t* lightlm_quant_matrix_new(lightlm_matrix_t* mat, int32_t dsub, int qnorm);

#ifdef __cplusplus
}
#endif
