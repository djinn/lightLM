#include "lightlm/model.h"
#include "lightlm/loss.h"
#include "lightlm/args.h"
#include "lightlm/dictionary.h"
#include "lightlm/matrix.h"
#include "lightlm/vector.h"
#include "lightlm/densematrix.h"
#include <stdio.h>
#include "lightlm/dynamic_array.h"


int main(int argc, char** argv) {
    printf("Running training test...\n");

    // 1. Create Args
    lightlm_args_t* args = lightlm_args_new();
    args->minCount = 1;
    args->minCountLabel = 1;

    // 2. Create Dictionary
    lightlm_dictionary_t* dict = lightlm_dictionary_new(args);
    lightlm_dictionary_read_from_file(dict, "test_data.txt");
    printf("Dictionary created. nwords=%d, nlabels=%d\n", dict->nwords_, dict->nlabels_);

    // 3. Create Matrices
    int32_t dim = 10;
    lightlm_matrix_t* wi = lightlm_dense_matrix_new(dict->nwords_ + args->bucket, dim);
    lightlm_matrix_t* wo = lightlm_dense_matrix_new(dict->nlabels_, dim);
    lightlm_dense_matrix_zero(wi);
    lightlm_dense_matrix_zero(wo);

    // 4. Create Loss
    lightlm_loss_t* loss = lightlm_softmax_loss_new(wo);

    // 5. Create Model
    lightlm_model_t* model = lightlm_model_new(wi, wo, loss, 0);

    // 6. Create Model::State
    lightlm_model_state_t* state = lightlm_model_state_new(dim, dict->nlabels_, 0);

    // 7. Get a line of training data
    FILE* in = fopen("test_data.txt", "r");
    dynamic_array_t words;
    dynamic_array_t labels;
    dynamic_array_init(&words, 1000);
    dynamic_array_init(&labels, 1000);

    lightlm_dictionary_get_line(dict, in, (void*)&words, (void*)&labels);
    fclose(in);

    printf("Got a line of data. nwords=%d, nlabels=%d\n", words.size, labels.size);

    // 8. Call model update
    lightlm_model_update(model, words.data, words.size, labels.data, labels.size, 0, 0.1, state);
    printf("Model update successful.\n");

    // Cleanup
    free(words.data);
    free(labels.data);
    lightlm_model_free(model);
    loss->free(loss);
    wi->free(wi);
    wo->free(wo);
    lightlm_dictionary_free(dict);
    lightlm_args_free(args);
    lightlm_model_state_free(state);

    printf("Training test finished successfully.\n");
    return 0;
}
