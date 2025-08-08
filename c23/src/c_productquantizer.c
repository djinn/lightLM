#include "lightlm/productquantizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define KSUB (1 << 8)

static lightlm_real distL2(const lightlm_real* x, const lightlm_real* y, int32_t d) {
    lightlm_real dist = 0;
    for (int i = 0; i < d; i++) {
        lightlm_real tmp = x[i] - y[i];
        dist += tmp * tmp;
    }
    return dist;
}

lightlm_product_quantizer_t* lightlm_product_quantizer_new(int32_t dim, int32_t dsub) {
    lightlm_product_quantizer_t* pq = (lightlm_product_quantizer_t*)malloc(sizeof(lightlm_product_quantizer_t));
    if (pq == NULL) return NULL;

    pq->dim_ = dim;
    pq->dsub_ = dsub;
    pq->nsubq_ = dim / dsub;
    pq->lastdsub_ = dim % dsub;
    if (pq->lastdsub_ == 0) {
        pq->lastdsub_ = dsub;
    } else {
        pq->nsubq_++;
    }
    pq->centroids_size_ = dim * KSUB;
    pq->centroids_ = (lightlm_real*)malloc(pq->centroids_size_ * sizeof(lightlm_real));
    if (pq->centroids_ == NULL) {
        free(pq);
        return NULL;
    }

    return pq;
}

void lightlm_product_quantizer_free(lightlm_product_quantizer_t* pq) {
    if (pq == NULL) return;
    free(pq->centroids_);
    free(pq);
}

static const int32_t nbits_ = 8;
static const int32_t ksub_ = 1 << nbits_;
static const int32_t max_points_per_cluster_ = 256;
static const int32_t max_points_ = max_points_per_cluster_ * ksub_;
static const int32_t seed_ = 1234;
static const int32_t niter_ = 25;
static const float eps_ = 1e-7;


static lightlm_real* get_centroids(lightlm_product_quantizer_t* pq, int32_t m, uint8_t i) {
    if (m == pq->nsubq_ - 1) {
        return &pq->centroids_[m * ksub_ * pq->dsub_ + i * pq->lastdsub_];
    }
    return &pq->centroids_[(m * ksub_ + i) * pq->dsub_];
}

static const lightlm_real* get_centroids_const(const lightlm_product_quantizer_t* pq, int32_t m, uint8_t i) {
    if (m == pq->nsubq_ - 1) {
        return &pq->centroids_[m * ksub_ * pq->dsub_ + i * pq->lastdsub_];
    }
    return &pq->centroids_[(m * ksub_ + i) * pq->dsub_];
}

static lightlm_real assign_centroid(const lightlm_real* x, const lightlm_real* c0, uint8_t* code, int32_t d) {
    const lightlm_real* c = c0;
    lightlm_real dis = distL2(x, c, d);
    code[0] = 0;
    for (int j = 1; j < ksub_; j++) {
        c += d;
        lightlm_real disij = distL2(x, c, d);
        if (disij < dis) {
            code[0] = (uint8_t)j;
            dis = disij;
        }
    }
    return dis;
}

static void Estep(const lightlm_real* x, const lightlm_real* centroids, uint8_t* codes, int32_t d, int32_t n) {
    for (int i = 0; i < n; i++) {
        assign_centroid(x + i * d, centroids, codes + i, d);
    }
}

static void MStep(const lightlm_real* x0, lightlm_real* centroids, const uint8_t* codes, int32_t d, int32_t n) {
    int32_t* nelts = (int32_t*)calloc(ksub_, sizeof(int32_t));
    memset(centroids, 0, sizeof(lightlm_real) * d * ksub_);
    const lightlm_real* x = x0;
    for (int i = 0; i < n; i++) {
        auto k = codes[i];
        lightlm_real* c = centroids + k * d;
        for (int j = 0; j < d; j++) {
            c[j] += x[j];
        }
        nelts[k]++;
        x += d;
    }

    lightlm_real* c = centroids;
    for (int k = 0; k < ksub_; k++) {
        lightlm_real z = (lightlm_real)nelts[k];
        if (z != 0) {
            for (int j = 0; j < d; j++) {
                c[j] /= z;
            }
        }
        c += d;
    }

    for (int k = 0; k < ksub_; k++) {
        if (nelts[k] == 0) {
            int32_t m = 0;
            while ((double)rand() / RAND_MAX * (n - ksub_) >= nelts[m] - 1) {
                m = (m + 1) % ksub_;
            }
            memcpy(centroids + k * d, centroids + m * d, sizeof(lightlm_real) * d);
            for (int j = 0; j < d; j++) {
                int32_t sign = (j % 2) * 2 - 1;
                centroids[k * d + j] += sign * eps_;
                centroids[m * d + j] -= sign * eps_;
            }
            nelts[k] = nelts[m] / 2;
            nelts[m] -= nelts[k];
        }
    }
    free(nelts);
}

static void iota(int* first, int* last, int value) {
    while (first != last) {
        *first++ = value++;
    }
}

static void shuffle(int* first, int* last) {
    int n = last - first;
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = first[i];
        first[i] = first[j];
        first[j] = tmp;
    }
}

static void kmeans(const lightlm_real* x, lightlm_real* c, int32_t n, int32_t d) {
    int* perm = (int*)malloc(n * sizeof(int));
    iota(perm, perm + n, 0);
    shuffle(perm, perm + n);
    for (int i = 0; i < ksub_; i++) {
        memcpy(&c[i * d], x + perm[i] * d, d * sizeof(lightlm_real));
    }
    uint8_t* codes = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i = 0; i < niter_; i++) {
        Estep(x, c, codes, d, n);
        MStep(x, c, codes, d, n);
    }
    free(codes);
    free(perm);
}

void lightlm_product_quantizer_train(lightlm_product_quantizer_t* pq, int n, const lightlm_real* x) {
    if (n < ksub_) {
        return; // error
    }
    int* perm = (int*)malloc(n * sizeof(int));
    iota(perm, perm + n, 0);
    int d = pq->dsub_;
    int np = (n < max_points_) ? n : max_points_;
    lightlm_real* xslice = (lightlm_real*)malloc(np * pq->dsub_ * sizeof(lightlm_real));

    for (int m = 0; m < pq->nsubq_; m++) {
        if (m == pq->nsubq_ - 1) {
            d = pq->lastdsub_;
        }
        if (np != n) {
            shuffle(perm, perm + n);
        }
        for (int j = 0; j < np; j++) {
            memcpy(xslice + j * d, x + perm[j] * pq->dim_ + m * pq->dsub_, d * sizeof(lightlm_real));
        }
        kmeans(xslice, get_centroids(pq, m, 0), np, d);
    }
    free(xslice);
    free(perm);
}

static void compute_code(const lightlm_product_quantizer_t* pq, const lightlm_real* x, uint8_t* code) {
    int d = pq->dsub_;
    for (int m = 0; m < pq->nsubq_; m++) {
        if (m == pq->nsubq_ - 1) {
            d = pq->lastdsub_;
        }
        assign_centroid(x + m * pq->dsub_, get_centroids_const(pq, m, 0), code + m, d);
    }
}

void lightlm_product_quantizer_compute_codes(const lightlm_product_quantizer_t* pq, const lightlm_real* x, uint8_t* codes, int32_t n) {
    for (int i = 0; i < n; i++) {
        compute_code(pq, x + i * pq->dim_, codes + i * pq->nsubq_);
    }
}

void lightlm_product_quantizer_addcode(const lightlm_product_quantizer_t* pq, lightlm_vector_t* x, const uint8_t* codes, int32_t t, lightlm_real alpha) {
    int d = pq->dsub_;
    const uint8_t* code = codes + pq->nsubq_ * t;
    for (int m = 0; m < pq->nsubq_; m++) {
        const lightlm_real* c = get_centroids_const(pq, m, code[m]);
        if (m == pq->nsubq_ - 1) {
            d = pq->lastdsub_;
        }
        for (int n = 0; n < d; n++) {
            x->data[m * pq->dsub_ + n] += alpha * c[n];
        }
    }
}

lightlm_real lightlm_product_quantizer_mulcode(const lightlm_product_quantizer_t* pq, const lightlm_vector_t* x, const uint8_t* codes, int32_t t, lightlm_real alpha) {
    lightlm_real res = 0.0;
    int d = pq->dsub_;
    const uint8_t* code = codes + pq->nsubq_ * t;
    for (int m = 0; m < pq->nsubq_; m++) {
        const lightlm_real* c = get_centroids_const(pq, m, code[m]);
        if (m == pq->nsubq_ - 1) {
            d = pq->lastdsub_;
        }
        for (int n = 0; n < d; n++) {
            res += x->data[m * pq->dsub_ + n] * c[n];
        }
    }
    return res * alpha;
}
