#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t* data;
    int size;
    int capacity;
} dynamic_array_t;

void dynamic_array_init(dynamic_array_t* arr, int capacity);
void dynamic_array_push(dynamic_array_t* arr, int32_t value);
void dynamic_array_free(dynamic_array_t* arr);

#ifdef __cplusplus
}
#endif
