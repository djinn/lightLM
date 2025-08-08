#include "lightlm/loss.h"
#include "lightlm/args.h"
#include "lightlm/dictionary.h"
#include "lightlm/matrix.h"
#include "lightlm/vector.h"
#include "lightlm/densematrix.h"
#include <stdio.h>

int main(int argc, char** argv) {
    printf("Running loss test...\n");

    // 1. Create Args
    lightlm_args_t* args = lightlm_args_new();

    // 2. Create Dictionary (not really needed for this test)
    lightlm_dictionary_t* dict = lightlm_dictionary_new(args);

    // 3. Create Matrix
    int32_t nclasses = 10;
    int32_t dim = 100;
    lightlm_matrix_t* wo = lightlm_dense_matrix_new(nclasses, dim);
    lightlm_dense_matrix_zero(wo);

    // 4. Create Loss
    lightlm_loss_t* loss = lightlm_softmax_loss_new(wo);

    // 5. Create Model::State
    lightlm_model_state_t state;
    state.hidden = lightlm_vector_new(dim);
    state.output = lightlm_vector_new(nclasses);
    state.grad = lightlm_vector_new(dim);
    lightlm_vector_zero(state.hidden);
    lightlm_vector_zero(state.output);
    lightlm_vector_zero(state.grad);

    // 6. Call forward and compute_output
    int32_t targets[] = {3};
    loss->compute_output(loss, &state);
    printf("compute_output successful.\n");

    lightlm_real loss_val = loss->forward(loss, targets, 1, 0, &state, 0.1, 1);
    printf("forward successful. loss = %f\n", loss_val);


    // Cleanup
    loss->free(loss);
    wo->free(wo);
    lightlm_dictionary_free(dict);
    lightlm_args_free(args);
    lightlm_vector_free(state.hidden);
    lightlm_vector_free(state.output);
    lightlm_vector_free(state.grad);

    printf("Loss test finished successfully.\n");
    return 0;
}
