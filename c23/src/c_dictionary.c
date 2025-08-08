/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightlm/dictionary.h"
#include "lightlm/dynamic_array.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

#define MAX_VOCAB_SIZE 30000000
#define MAX_LINE_SIZE 1024


// ==========================================================================
// Hash Map implementation for pruneidx_
// ==========================================================================

lightlm_hash_map* lightlm_hash_map_new(int32_t capacity) {
    lightlm_hash_map* map = (lightlm_hash_map*)malloc(sizeof(lightlm_hash_map));
    if (map == NULL) return NULL;
    map->nodes = (lightlm_hash_node*)calloc(capacity, sizeof(lightlm_hash_node));
    if (map->nodes == NULL) {
        free(map);
        return NULL;
    }
    for(int i = 0; i < capacity; i++) {
        map->nodes[i].key = -1; // -1 indicates an empty slot
    }
    map->size = 0;
    map->capacity = capacity;
    return map;
}

void lightlm_hash_map_free(lightlm_hash_map* map) {
    if (map == NULL) return;
    free(map->nodes);
    free(map);
}


// ==========================================================================
// Dictionary implementation
// ==========================================================================

lightlm_dictionary_t* lightlm_dictionary_new(lightlm_args_t* args) {
    lightlm_dictionary_t* dict = (lightlm_dictionary_t*)malloc(sizeof(lightlm_dictionary_t));
    if (dict == NULL) {
        return NULL;
    }

    dict->args_ = args;
    dict->word2int_size_ = MAX_VOCAB_SIZE;
    dict->word2int_ = (int32_t*)malloc(dict->word2int_size_ * sizeof(int32_t));
    if (dict->word2int_ == NULL) {
        free(dict);
        return NULL;
    }
    for (int i = 0; i < dict->word2int_size_; i++) {
        dict->word2int_[i] = -1;
    }

    dict->capacity_ = 1000; // Initial capacity
    dict->words_ = (lightlm_entry*)malloc(dict->capacity_ * sizeof(lightlm_entry));
    if (dict->words_ == NULL) {
        free(dict->word2int_);
        free(dict);
        return NULL;
    }

    dict->size_ = 0;
    dict->nwords_ = 0;
    dict->nlabels_ = 0;
    dict->ntokens_ = 0;
    dict->pdiscard_ = NULL;
    dict->pruneidx_size_ = -1;
    dict->pruneidx_ = NULL;

    return dict;
}

void lightlm_dictionary_free(lightlm_dictionary_t* dict) {
    if (dict == NULL) {
        return;
    }
    for (int32_t i = 0; i < dict->size_; i++) {
        free(dict->words_[i].word);
        free(dict->words_[i].subwords);
    }
    free(dict->words_);
    free(dict->word2int_);
    free(dict->pdiscard_);
    if (dict->pruneidx_) {
        lightlm_hash_map_free(dict->pruneidx_);
    }
    free(dict);
}

int32_t lightlm_dictionary_nwords(const lightlm_dictionary_t* dict) {
    return dict->nwords_;
}

int32_t lightlm_dictionary_nlabels(const lightlm_dictionary_t* dict) {
    return dict->nlabels_;
}

int64_t lightlm_dictionary_ntokens(const lightlm_dictionary_t* dict) {
    return dict->ntokens_;
}

const char* lightlm_dictionary_get_word(const lightlm_dictionary_t* dict, int32_t id) {
    if (id < 0 || id >= dict->size_) {
        return NULL;
    }
    return dict->words_[id].word;
}

static uint32_t lightlm_dictionary_hash(const char* str) {
    uint32_t h = 2166136261;
    for (size_t i = 0; str[i] != '\0'; i++) {
        h = h ^ (uint32_t)(int8_t)(str[i]);
        h = h * 16777619;
    }
    return h;
}

static int32_t lightlm_dictionary_find(const lightlm_dictionary_t* dict, const char* w, uint32_t h) {
    int32_t id = h % dict->word2int_size_;
    while (dict->word2int_[id] != -1 && strcmp(dict->words_[dict->word2int_[id]].word, w) != 0) {
        id = (id + 1) % dict->word2int_size_;
    }
    return id;
}

int32_t lightlm_dictionary_get_id(const lightlm_dictionary_t* dict, const char* w) {
    uint32_t h = lightlm_dictionary_hash(w);
    int32_t id = lightlm_dictionary_find(dict, w, h);
    return dict->word2int_[id];
}

static lightlm_entry_type lightlm_dictionary_get_type(const lightlm_dictionary_t* dict, const char* w) {
    return (strstr(w, dict->args_->label) == w) ? lightlm_entry_type_label : lightlm_entry_type_word;
}

void lightlm_dictionary_add(lightlm_dictionary_t* dict, const char* w) {
    uint32_t h = lightlm_dictionary_hash(w);
    int32_t id = lightlm_dictionary_find(dict, w, h);
    dict->ntokens_++;
    if (dict->word2int_[id] == -1) {
        if (dict->size_ == dict->capacity_) {
            dict->capacity_ *= 2;
            dict->words_ = (lightlm_entry*)realloc(dict->words_, dict->capacity_ * sizeof(lightlm_entry));
        }
        lightlm_entry* e = &dict->words_[dict->size_];
        e->word = strdup(w);
        e->count = 1;
        e->type = lightlm_dictionary_get_type(dict, w);
        e->subwords = NULL;
        e->subwords_size = 0;
        dict->word2int_[id] = dict->size_++;
    } else {
        dict->words_[dict->word2int_[id]].count++;
    }
}

static void push_subword(lightlm_entry* e, int32_t subword_id, int* capacity) {
    if (e->subwords_size == *capacity) {
        *capacity *= 2;
        e->subwords = (int32_t*)realloc(e->subwords, *capacity * sizeof(int32_t));
    }
    e->subwords[e->subwords_size++] = subword_id;
}

static void lightlm_dictionary_compute_subwords_for_entry(
    const lightlm_dictionary_t* dict,
    const char* word,
    lightlm_entry* entry,
    int* capacity
) {
    for (size_t i = 0; i < strlen(word); i++) {
        char ngram_str[MAX_LINE_SIZE];
        if ((word[i] & 0xC0) == 0x80) {
            continue;
        }
        int ngram_len = 0;
        for (size_t j = i, n = 1; j < strlen(word) && n <= dict->args_->maxn; n++) {
            ngram_str[ngram_len++] = word[j++];
            while (j < strlen(word) && (word[j] & 0xC0) == 0x80) {
                ngram_str[ngram_len++] = word[j++];
            }
            ngram_str[ngram_len] = '\0';
            if (n >= dict->args_->minn && !(n == 1 && (i == 0 || j == strlen(word)))) {
                uint32_t h = lightlm_dictionary_hash(ngram_str) % dict->args_->bucket;
                // Not handling pruning yet
                push_subword(entry, dict->nwords_ + h, capacity);
            }
        }
    }
}


void lightlm_dictionary_init_ngrams(lightlm_dictionary_t* dict) {
    for (int32_t i = 0; i < dict->size_; i++) {
        char word_with_bow_eow[MAX_LINE_SIZE];
        snprintf(word_with_bow_eow, MAX_LINE_SIZE, "<%s>", dict->words_[i].word);

        free(dict->words_[i].subwords);
        int capacity = 20;
        dict->words_[i].subwords = (int32_t*)malloc(capacity * sizeof(int32_t));
        dict->words_[i].subwords_size = 0;

        push_subword(&dict->words_[i], i, &capacity);

        if (strcmp(dict->words_[i].word, "</s>") != 0) {
            lightlm_dictionary_compute_subwords_for_entry(dict, word_with_bow_eow, &dict->words_[i], &capacity);
        }
    }
}

// Reads a word from a file. Returns 1 if a word was read, 0 otherwise.
// A word is a sequence of non-whitespace characters.
// A newline character is treated as the special EOS word.
static int lightlm_dictionary_read_word(FILE* in, char* word) {
    int c;
    int i = 0;
    while ((c = fgetc(in)) != EOF) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
            if (i == 0) {
                if (c == '\n') {
                    strcpy(word, "</s>");
                    return 1;
                }
                continue;
            } else {
                if (c == '\n') {
                    ungetc(c, in);
                }
                break;
            }
        }
        word[i++] = (char)c;
        if (i >= MAX_LINE_SIZE - 1) {
            // Avoid buffer overflow
            break;
        }
    }
    word[i] = '\0';
    return i > 0 || c != EOF;
}

static int compare_entries(const void* a, const void* b) {
    const lightlm_entry* e1 = (const lightlm_entry*)a;
    const lightlm_entry* e2 = (const lightlm_entry*)b;
    if (e1->type != e2->type) {
        return e1->type - e2->type;
    }
    return e2->count - e1->count;
}

void lightlm_dictionary_threshold(lightlm_dictionary_t* dict, int64_t t, int64_t tl) {
    qsort(dict->words_, dict->size_, sizeof(lightlm_entry), compare_entries);

    int32_t new_size = 0;
    for (int32_t i = 0; i < dict->size_; i++) {
        if ((dict->words_[i].type == lightlm_entry_type_word && dict->words_[i].count >= t) ||
            (dict->words_[i].type == lightlm_entry_type_label && dict->words_[i].count >= tl)) {
            dict->words_[new_size++] = dict->words_[i];
        } else {
            free(dict->words_[i].word);
            free(dict->words_[i].subwords);
        }
    }
    dict->size_ = new_size;
    dict->words_ = (lightlm_entry*)realloc(dict->words_, dict->size_ * sizeof(lightlm_entry));

    dict->nwords_ = 0;
    dict->nlabels_ = 0;
    for (int i = 0; i < dict->word2int_size_; i++) {
        dict->word2int_[i] = -1;
    }

    for (int32_t i = 0; i < dict->size_; i++) {
        uint32_t h = lightlm_dictionary_hash(dict->words_[i].word);
        int32_t id = lightlm_dictionary_find(dict, dict->words_[i].word, h);
        dict->word2int_[id] = i;
        if (dict->words_[i].type == lightlm_entry_type_word) {
            dict->nwords_++;
        } else {
            dict->nlabels_++;
        }
    }
}

void lightlm_dictionary_init_table_discard(lightlm_dictionary_t* dict) {
    dict->pdiscard_ = (lightlm_real*)malloc(dict->size_ * sizeof(lightlm_real));
    for (int32_t i = 0; i < dict->size_; i++) {
        lightlm_real f = (lightlm_real)dict->words_[i].count / (lightlm_real)dict->ntokens_;
        dict->pdiscard_[i] = sqrt(dict->args_->t / f) + dict->args_->t / f;
    }
}

void lightlm_dictionary_read_from_file(lightlm_dictionary_t* dict, const char* filename) {
    FILE* in = fopen(filename, "r");
    if (in == NULL) {
        // TODO: error handling
        return;
    }

    char word[MAX_LINE_SIZE];
    int64_t minThreshold = 1;
    while (lightlm_dictionary_read_word(in, word)) {
        lightlm_dictionary_add(dict, word);
        if (dict->ntokens_ % 1000000 == 0 && dict->args_->verbose > 1) {
            fprintf(stderr, "\rRead %" PRId64 "M words", dict->ntokens_ / 1000000);
            fflush(stderr);
        }
        if (dict->size_ > 0.75 * MAX_VOCAB_SIZE) {
            minThreshold++;
            lightlm_dictionary_threshold(dict, minThreshold, minThreshold);
        }
    }
    fclose(in);
    lightlm_dictionary_threshold(dict, dict->args_->minCount, dict->args_->minCountLabel);
    lightlm_dictionary_init_table_discard(dict);
    lightlm_dictionary_init_ngrams(dict);
    if (dict->args_->verbose > 0) {
        fprintf(stderr, "\rRead %" PRId64 "M words\n", dict->ntokens_ / 1000000);
        fprintf(stderr, "Number of words:  %d\n", dict->nwords_);
        fprintf(stderr, "Number of labels: %d\n", dict->nlabels_);
    }
    if (dict->size_ == 0) {
        // TODO: error handling
    }
}

static void lightlm_dictionary_add_word_ngrams(lightlm_dictionary_t* dict, dynamic_array_t* line, const dynamic_array_t* hashes, int32_t n) {
    for (int32_t i = 0; i < hashes->size; i++) {
        uint64_t h = hashes->data[i];
        for (int32_t j = i + 1; j < hashes->size && j < i + n; j++) {
            h = h * 116049371 + hashes->data[j];
            // Not handling pruning yet
            dynamic_array_push(line, dict->nwords_ + (h % dict->args_->bucket));
        }
    }
}

static void lightlm_dictionary_add_subwords(lightlm_dictionary_t* dict, dynamic_array_t* line, const char* token, int32_t wid) {
    if (wid < 0) { // out of vocab
        if (strcmp(token, "</s>") != 0) {
            // computeSubwords for OOV words
            // This is complex, will implement later
        }
    } else {
        if (dict->args_->maxn <= 0) { // in vocab w/o subwords
            dynamic_array_push(line, wid);
        } else { // in vocab w/ subwords
            const lightlm_entry* e = &dict->words_[wid];
            for (int i = 0; i < e->subwords_size; i++) {
                dynamic_array_push(line, e->subwords[i]);
            }
        }
    }
}


static void reset_file(FILE* in) {
    if (feof(in)) {
        clearerr(in);
        fseek(in, 0, SEEK_SET);
    }
}

int32_t lightlm_dictionary_get_line(
    lightlm_dictionary_t* dict,
    void* in_void,
    void* words_void,
    void* labels_void
) {
    FILE* in = (FILE*)in_void;
    dynamic_array_t* words = (dynamic_array_t*)words_void;
    dynamic_array_t* labels = (dynamic_array_t*)labels_void;
    dynamic_array_t word_hashes;
    dynamic_array_init(&word_hashes, 100);
    char token[MAX_LINE_SIZE];
    int32_t ntokens = 0;

    reset_file(in);
    words->size = 0;
    labels->size = 0;

    while (lightlm_dictionary_read_word(in, token)) {
        uint32_t h = lightlm_dictionary_hash(token);
        int32_t wid = lightlm_dictionary_get_id(dict, token);
        lightlm_entry_type type = (wid < 0) ? lightlm_dictionary_get_type(dict, token) : dict->words_[wid].type;

        ntokens++;
        if (type == lightlm_entry_type_word) {
            lightlm_dictionary_add_subwords(dict, words, token, wid);
            dynamic_array_push(&word_hashes, h);
        } else if (type == lightlm_entry_type_label && wid >= 0) {
            dynamic_array_push(labels, wid - dict->nwords_);
        }
        if (strcmp(token, "</s>") == 0) {
            break;
        }
    }
    lightlm_dictionary_add_word_ngrams(dict, words, &word_hashes, dict->args_->wordNgrams);
    dynamic_array_free(&word_hashes);
    return ntokens;
}
