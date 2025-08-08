#include "lightlm/dynamic_array.h"
#include <stdlib.h>

void dynamic_array_init(dynamic_array_t* arr, int capacity) {
    arr->data = (int32_t*)malloc(capacity * sizeof(int32_t));
    arr->size = 0;
    arr->capacity = capacity;
}

void dynamic_array_push(dynamic_array_t* arr, int32_t value) {
    if (arr->size == arr->capacity) {
        arr->capacity *= 2;
        arr->data = (int32_t*)realloc(arr->data, arr->capacity * sizeof(int32_t));
    }
    arr->data[arr->size++] = value;
}

void dynamic_array_free(dynamic_array_t* arr) {
    free(arr->data);
}
