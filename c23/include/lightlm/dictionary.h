/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include "lightlm/args.h"
#include "lightlm/real.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t lightlm_id_type;
typedef enum {
    lightlm_entry_type_word = 0,
    lightlm_entry_type_label = 1
} lightlm_entry_type;

typedef struct {
    char* word;
    int64_t count;
    lightlm_entry_type type;
    int32_t* subwords;
    int32_t subwords_size;
} lightlm_entry;

// A basic hash map implementation for pruneidx_
typedef struct {
    int32_t key;
    int32_t value;
} lightlm_hash_node;

typedef struct {
    lightlm_hash_node* nodes;
    int32_t size;
    int32_t capacity;
} lightlm_hash_map;


typedef struct lightlm_dictionary_s {
    lightlm_args_t* args_;
    int32_t* word2int_;
    int32_t word2int_size_;

    lightlm_entry* words_;
    int32_t size_;
    int32_t capacity_;

    lightlm_real* pdiscard_;
    int32_t nwords_;
    int32_t nlabels_;
    int64_t ntokens_;

    int64_t pruneidx_size_;
    lightlm_hash_map* pruneidx_;
} lightlm_dictionary_t;

lightlm_dictionary_t* lightlm_dictionary_new(lightlm_args_t* args);
void lightlm_dictionary_free(lightlm_dictionary_t* dict);
void lightlm_dictionary_read_from_file(lightlm_dictionary_t* dict, const char* filename);
int32_t lightlm_dictionary_get_id(const lightlm_dictionary_t* dict, const char* word);
const char* lightlm_dictionary_get_word(const lightlm_dictionary_t* dict, int32_t id);
int32_t lightlm_dictionary_get_line(lightlm_dictionary_t* dict, void* in, void* words, void* labels);
int32_t lightlm_dictionary_nwords(const lightlm_dictionary_t* dict);
int32_t lightlm_dictionary_nlabels(const lightlm_dictionary_t* dict);
int64_t lightlm_dictionary_ntokens(const lightlm_dictionary_t* dict);

#ifdef __cplusplus
}
#endif
