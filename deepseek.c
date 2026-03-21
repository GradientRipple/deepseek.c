#define _CRT_SECURE_NO_DEPRECATE 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <fcntl.h>

#if defined(_WIN32) || defined(_WIN64)
#include <errno.h>
#include <io.h>
#include <fcntl.h>
#define WIN32_LEAN_AND_MEAN            
#include <windows.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>

#define ssize_t int64_t
#ifndef _WIN32_WINNT                      
#define _WIN32_WINNT    0x0501              
#endif

#ifndef _MSC_VER
#include <_mingw.h>
#endif

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_NONE       0
#define PROT_READ       1
#define PROT_WRITE      2
#define PROT_EXEC       4

#define MAP_FILE        0
#define MAP_SHARED      1
#define MAP_PRIVATE     2
#define MAP_TYPE        0xf
#define MAP_FIXED       0x10
#define MAP_ANONYMOUS   0x20
#define MAP_ANON        MAP_ANONYMOUS

#define MAP_FAILED      ((void *)-1)

#define MS_ASYNC        1
#define MS_SYNC         2
#define MS_INVALIDATE   4

#define CLOCK_REALTIME  0

    void* mmap(void* addr, size_t len, int prot, int flags, int fildes, ssize_t off);
    int     munmap(void* addr, size_t len);
#ifdef __cplusplus
};
#endif
#else
#include <unistd.h>
#include <sys/mman.h>

#endif

#if defined(_WIN32) || defined(_WIN64)
#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE    0x0020
#endif   

static int __map_mman_error(const uint32_t err, const int deferr)
{
    if (err == 0)
        return 0;
    return err;
}

static uint32_t __map_mmap_prot_page(const int prot)
{
    uint32_t protect = 0;

    if (prot == PROT_NONE)
        return protect;

    if ((prot & PROT_EXEC) != 0)
    {
        protect = ((prot & PROT_WRITE) != 0) ?
            PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    }
    else
    {
        protect = ((prot & PROT_WRITE) != 0) ?
            PAGE_READWRITE : PAGE_READONLY;
    }

    return protect;
}

static uint32_t __map_mmap_prot_file(const int prot)
{
    uint32_t desiredAccess = 0;

    if (prot == PROT_NONE)
        return desiredAccess;

    if ((prot & PROT_READ) != 0)
        desiredAccess |= FILE_MAP_READ;
    if ((prot & PROT_WRITE) != 0)
        desiredAccess |= FILE_MAP_WRITE;
    if ((prot & PROT_EXEC) != 0)
        desiredAccess |= FILE_MAP_EXECUTE;

    return desiredAccess;
}

void* mmap(void* addr, size_t len, int prot, int flags, int fildes, ssize_t off)
{
    HANDLE fm, h;
    void* map = MAP_FAILED;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4293)
#endif

    const uint32_t dwFileOffsetLow = (uint32_t)(off & 0xFFFFFFFFL);
    const uint32_t dwFileOffsetHigh = (uint32_t)((off >> 32) & 0xFFFFFFFFL);
    const uint32_t protect = __map_mmap_prot_page(prot);
    const uint32_t desiredAccess = __map_mmap_prot_file(prot);

    const ssize_t maxSize = off + (ssize_t)len;

    const uint32_t dwMaxSizeLow = (uint32_t)(maxSize & 0xFFFFFFFFL);
    const uint32_t dwMaxSizeHigh = (uint32_t)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    errno = 0;

    if (len == 0
        || (flags & MAP_FIXED) != 0
        || prot == PROT_EXEC)
    {
        errno = EINVAL;
        return MAP_FAILED;
    }

    h = ((flags & MAP_ANONYMOUS) == 0) ?
        (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

    if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
    {
        errno = EBADF;
        return MAP_FAILED;
    }

    fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

    if (fm == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

    CloseHandle(fm);

    if (map == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    return map;
}

int munmap(void* addr, size_t len)
{
    if (UnmapViewOfFile(addr))
        return 0;

    errno = __map_mman_error(GetLastError(), EPERM);

    return -1;
}

#endif

int GS = 0;         

#define PROMPT_BUFFER_SIZE 32768

#define ROPE_THETA   (10000.0f)
#define RMS_NORM_EPS (1e-06)

typedef struct {
    int magic_number;
    int version;
    int dim;   
    int hidden_dim;    
    int n_layers;    
    int n_heads;     
    int n_kv_heads;             
    int vocab_size;      
    int seq_len;    
    int head_dim;
    int group_size;       
    int shared_classifier;
} Config;

typedef struct {
    int8_t* q;      
    float* s;   
} QuantizedTensor;

typedef struct {
    QuantizedTensor* q_tokens;   
    float* token_embedding_table;    
    float* rms_att_weight;     
    float* rms_ffn_weight;   
    QuantizedTensor* wq;      
    QuantizedTensor* wk;      
    QuantizedTensor* wv;      
    QuantizedTensor* wo;      
    float* wq_bias;     
    float* wk_bias;     
    float* wv_bias;     

    QuantizedTensor* w1;    
    QuantizedTensor* w2;    
    QuantizedTensor* w3;    
    float* rms_final_weight;  
    QuantizedTensor* wcls;

} TransformerWeights;

typedef struct {
    float* x;       
    float* xb;        
    float* hb;         
    float* hb2;         
    QuantizedTensor xq;    
    QuantizedTensor hq;    
    float* q;   
    float* k;   
    float* v;   
    float* att;       
    float* logits;   
    float* key_cache;      
    float* value_cache;    
} RunState;

typedef struct {
    Config config;        
    TransformerWeights weights;      
    RunState state;           
    float* data;     
    ssize_t file_size;        
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor){ .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor){ .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void dequantize(QuantizedTensor* qx, float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = qx->q[i] * qx->s[i / GS];
}

void quantize(QuantizedTensor* qx, float* x, int n) {
    for (int group = 0; group < n / GS; group++) {
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
                wmax = val;
        }

        float scale = wmax / 127.0f;
        qx->s[group] = scale;

        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;  
            int8_t quantized = (int8_t)round(quant_value);    
            qx->q[group * GS + i] = quantized;
        }
    }
}

QuantizedTensor* init_quantized_tensors(void** ptr, int n, int size_each) {
    QuantizedTensor* res = malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)*ptr;
        *ptr = (int8_t*)*ptr + size_each;
        res[i].s = (float*)*ptr;
        *ptr = (float*)*ptr + size_each / GS;
    }
    return res;
}

void memory_map_weights(TransformerWeights* w, Config* p, void* ptr) {

    int head_size = p->head_dim;
    float* fptr = (float*)ptr;      

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;

    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;

    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (void*)fptr;        

    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);


    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));

    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));

    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));

    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);


    float* fptr_bias = (float*)ptr;

    w->wq_bias = fptr_bias;
    fptr_bias += p->n_layers * (p->n_heads * head_size);

    w->wk_bias = fptr_bias;
    fptr_bias += p->n_layers * (p->n_kv_heads * head_size);

    w->wv_bias = fptr_bias;
    fptr_bias += p->n_layers * (p->n_kv_heads * head_size);

    ptr = (void*)fptr_bias;     


    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);

    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);

    char* cptr = (char*)ptr;
    size_t n_freq = p->seq_len * head_size / 2;
    size_t bytes = n_freq * sizeof(float);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file)
    {
        exit(EXIT_FAILURE);
    }

#if defined(_WIN32) || defined(_WIN64)
    _fseeki64(file, 0, SEEK_END);        
    *file_size = _ftelli64(file);       
    _fseeki64(file, 0, SEEK_SET);   
#else
    fseek(file, 0, SEEK_END);        
    *file_size = ftell(file);       
    fseek(file, 0, SEEK_SET);   
#endif

    * data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED)
    {
        exit(EXIT_FAILURE);
    }
    fclose(file);

    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x73796630)
    {
        exit(EXIT_FAILURE);
    }
    if (config->version != 1)
    {
        exit(EXIT_FAILURE);
    }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    GS = config->group_size;            

    void* weights_ptr = ((char*)*data) + 256;      
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer* t, char* checkpoint_path, int ctx_length) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    free_run_state(&t->state);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0;
    int j;

    for (j = 0; j <= size - 8; j += 8) {
        ss += x[j] * x[j];
        ss += x[j + 1] * x[j + 1];
        ss += x[j + 2] * x[j + 2];
        ss += x[j + 3] * x[j + 3];
        ss += x[j + 4] * x[j + 4];
        ss += x[j + 5] * x[j + 5];
        ss += x[j + 6] * x[j + 6];
        ss += x[j + 7] * x[j + 7];
    }
    for (; j < size; j++) {
        ss += x[j] * x[j];
    }

    float inv_norm = 1.0f / sqrtf((ss / size) + 1e-6f);

    for (j = 0; j <= size - 8; j += 8) {
        o[j] = weight[j] * (inv_norm * x[j]);
        o[j + 1] = weight[j + 1] * (inv_norm * x[j + 1]);
        o[j + 2] = weight[j + 2] * (inv_norm * x[j + 2]);
        o[j + 3] = weight[j + 3] * (inv_norm * x[j + 3]);
        o[j + 4] = weight[j + 4] * (inv_norm * x[j + 4]);
        o[j + 5] = weight[j + 5] * (inv_norm * x[j + 5]);
        o[j + 6] = weight[j + 6] * (inv_norm * x[j + 6]);
        o[j + 7] = weight[j + 7] * (inv_norm * x[j + 7]);
    }
    for (; j < size; j++) {
        o[j] = weight[j] * (inv_norm * x[j]);
    }
}

float fast_exp(float x) {
    x = 1.0f + x / 256.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

void softmax(float* x, int size) {
    float max_val = -1e38f;
    int i;

    for (i = 0; i <= size - 8; i += 8) {
        if (x[i] > max_val) max_val = x[i];
        if (x[i + 1] > max_val) max_val = x[i + 1];
        if (x[i + 2] > max_val) max_val = x[i + 2];
        if (x[i + 3] > max_val) max_val = x[i + 3];
        if (x[i + 4] > max_val) max_val = x[i + 4];
        if (x[i + 5] > max_val) max_val = x[i + 5];
        if (x[i + 6] > max_val) max_val = x[i + 6];
        if (x[i + 7] > max_val) max_val = x[i + 7];
    }
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0;
    for (i = 0; i <= size - 8; i += 8) {
        x[i] = fast_exp(x[i] - max_val);
        x[i + 1] = fast_exp(x[i + 1] - max_val);
        x[i + 2] = fast_exp(x[i + 2] - max_val);
        x[i + 3] = fast_exp(x[i + 3] - max_val);
        x[i + 4] = fast_exp(x[i + 4] - max_val);
        x[i + 5] = fast_exp(x[i + 5] - max_val);
        x[i + 6] = fast_exp(x[i + 6] - max_val);
        x[i + 7] = fast_exp(x[i + 7] - max_val);
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] +
            x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
    }
    for (; i < size; i++) {
        x[i] = fast_exp(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (i = 0; i <= size - 8; i += 8) {
        x[i] *= inv_sum;
        x[i + 1] *= inv_sum;
        x[i + 2] *= inv_sum;
        x[i + 3] *= inv_sum;
        x[i + 4] *= inv_sum;
        x[i + 5] *= inv_sum;
        x[i + 6] *= inv_sum;
        x[i + 7] *= inv_sum;
    }
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
}

static inline void matmul(float* xout, QuantizedTensor* x, QuantizedTensor* w, int n, int d) {
    const int8_t* x_q = x->q;
    const int8_t* w_q = w->q;
    const float* w_s = w->s;
    const float* x_s = x->s;
    const int n_gs = n / GS;

    int i;
#pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        const int in = i * n;
        const int in_gs = i * n_gs;

        int j;
        int j_gs = 0;   
        for (j = 0; j <= n - GS; j += GS) {   
            int32_t ival = 0;

            const int8_t* x_ptr = &x_q[j];
            const int8_t* w_ptr = &w_q[in + j];

            for (int k = 0; k < GS; k++) {
                ival += x_ptr[k] * w_ptr[k];
            }

            val += ((float)ival) * w_s[in_gs + j_gs] * x_s[j_gs];

            j_gs++;
        }

        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;         
    int all_heads_dim = p->n_heads * p->head_dim;

    memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim;       

        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim);

        quantize(&s->xq, s->xb, p->dim);
        matmul(s->q, &s->xq, w->wq + l, p->dim, all_heads_dim);
        for (int i = 0; i < all_heads_dim; i++) {
            s->q[i] += w->wq_bias[l * all_heads_dim + i];
        }
        matmul(s->k, &s->xq, w->wk + l, p->dim, kv_dim);
        for (int i = 0; i < kv_dim; i++) {      
            s->k[i] += w->wk_bias[l * kv_dim + i];
        }
        matmul(s->v, &s->xq, w->wv + l, p->dim, kv_dim);
        for (int i = 0; i < kv_dim; i++) {      
            s->v[i] += w->wv_bias[l * kv_dim + i];
        }

        int head_size = p->head_dim;
        for (int i = 0; i < all_heads_dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(ROPE_THETA, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;             
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;        
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int h;
        const float inv_sqrt_head_dim = 1.0f / sqrtf(p->head_dim);
        const int head_dim = p->head_dim;

#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_dim;
            float* att = s->att + h * p->seq_len;

            int kv_head_idx = h / kv_mul;
            float* key_base = s->key_cache + loff + kv_head_idx * head_dim;
            float* value_base = s->value_cache + loff + kv_head_idx * head_dim;

            for (int t = 0; t <= pos; t++) {
                float* k = key_base + t * kv_dim;
                float score = 0;

                int i = 0;
                for (; i <= head_dim - 8; i += 8) {
                    score += q[i] * k[i];
                    score += q[i + 1] * k[i + 1];
                    score += q[i + 2] * k[i + 2];
                    score += q[i + 3] * k[i + 3];
                    score += q[i + 4] * k[i + 4];
                    score += q[i + 5] * k[i + 5];
                    score += q[i + 6] * k[i + 6];
                    score += q[i + 7] * k[i + 7];
                }
                for (; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score * inv_sqrt_head_dim;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_dim;

            for (int i = 0; i < head_dim; i++) {
                xb[i] = 0;
            }

            for (int t = 0; t <= pos; t++) {
                float* v = value_base + t * kv_dim;
                float att_weight = att[t];

                int i = 0;
                for (; i <= head_dim - 8; i += 8) {
                    xb[i] += att_weight * v[i];
                    xb[i + 1] += att_weight * v[i + 1];
                    xb[i + 2] += att_weight * v[i + 2];
                    xb[i + 3] += att_weight * v[i + 3];
                    xb[i + 4] += att_weight * v[i + 4];
                    xb[i + 5] += att_weight * v[i + 5];
                    xb[i + 6] += att_weight * v[i + 6];
                    xb[i + 7] += att_weight * v[i + 7];
                }
                for (; i < head_dim; i++) {
                    xb[i] += att_weight * v[i];
                }
            }
        }

        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb, &s->xq, w->wo + l, all_heads_dim, p->dim);

        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim);

        quantize(&s->xq, s->xb, p->dim);
        matmul(s->hb, &s->xq, w->w1 + l, p->dim, p->hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, p->dim, p->hidden_dim);

        for (int i = 0; i < p->hidden_dim; i++)
            s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(-s->hb[i])));

        quantize(&s->hq, s->hb, p->hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, p->hidden_dim, p->dim);

        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    quantize(&s->xq, s->x, p->dim);
    matmul(s->logits, &s->xq, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

long time_in_ms() {
    clock_t clock_time = clock();
    return (long)((double)clock_time / CLOCKS_PER_SEC * 1000.0);
}


typedef struct {
    char** vocab;
    float* merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->merge_scores = (float*)malloc(vocab_size * sizeof(float));

    FILE* file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        exit(EXIT_FAILURE);
    }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char*)malloc(1);
            t->vocab[i][0] = 0;      
        }
        else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char*)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = 0;      
        }
    }
    fclose(file);

    strcpy(t->prompt_template, "<｜User｜>%s<｜Assistant｜>\n");
    strcpy(t->system_prompt_template, "%s<｜User｜>%s<｜Assistant｜>\n");
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}

char* decode(Tokenizer* t, int token) {
    return t->vocab[token];
}

int str_lookup(char* str, char** vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;

    return -1;
}

void encode(Tokenizer* t, char* text, int* tokens, int* n_tokens) {
    char* str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    char special_token[64 + 1];

    *n_tokens = 0;

    for (char* c = text; *c != 0; c++) {
        int id, found_special_token = 0;

        str_buffer[0] = *c;
        str_buffer[1] = 0;

        if (*c == '<') {
            int end_of_token_pos = -1;
            found_special_token = 0;
            for (int k = 0; *c != 0 && k < 64; k++) {
                if (c[k] == '>') {
                    end_of_token_pos = k;
                    break;
                }
            }

            if (end_of_token_pos != -1) {
                strncpy(special_token, c, end_of_token_pos + 1);
                special_token[end_of_token_pos + 1] = 0;

                id = str_lookup(special_token, t->vocab, t->vocab_size);
                if (id != -1) {
                    c += end_of_token_pos;
                    found_special_token = 1;
                }
            }
        }

        if (!found_special_token)
            id = str_lookup(str_buffer, t->vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        }
        else {
            (*n_tokens)++;
        }
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break;            

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--;    
    }

    free(str_buffer);
}

typedef struct {
    float prob;
    int index;
} ProbIndex;         

typedef struct {
    int vocab_size;
    ProbIndex* probindex;      
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1;      
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0;
    int last_idx = n0 - 1;         
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;       
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index;      
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long* state) {     
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    if (sampler->temperature == 0) {
        return sample_argmax(logits, sampler->vocab_size);
    }
    else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            return sample_mult(logits, sampler->vocab_size, coin);
        }
        else {
            return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = 0;   
    }
}

void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* cli_user_prompt, char* system_prompt)
{
    char user_prompt[PROMPT_BUFFER_SIZE];
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(PROMPT_BUFFER_SIZE * sizeof(int));

    int user_turn = 1;   
    int next;                
    int token;                
    int pos = 0;         

    clock_t start_time = 0;
    int gen_tokens = 0;
    int in_generation = 0;

    while (1) {
        if (pos >= transformer->config.seq_len) {
            user_turn = 1;
            pos = 0;

            in_generation = 0;
            gen_tokens = 0;
        }

        if (user_turn)
        {
            if (cli_user_prompt != NULL) {
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            }
            else {
                read_stdin("\n>> ", user_prompt, sizeof(user_prompt));
                if (!user_prompt[0])
                    break;
            }

            if (pos == 0 && system_prompt)
                sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt);
            else
                sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt);

            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            pos = 0;     
            user_turn = 0;

            in_generation = 0;
            gen_tokens = 0;
        }

        if (pos < num_prompt_tokens) {
            token = prompt_tokens[pos];
        }
        else {
            token = next;
        }

        float* logits = forward(transformer, token, pos++);
        next = sample(sampler, logits);

        if (pos >= num_prompt_tokens) {
            if (token == tokenizer->bos_token_id || token == tokenizer->eos_token_id)
            {
                if (in_generation && gen_tokens > 0) {
                    clock_t end_time = clock();
                    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    if (elapsed > 0) {
                        printf("\n\n%.3f\n", gen_tokens / elapsed);
                    }
                }
                printf("\n");
                user_turn = 1;
                in_generation = 0;
                gen_tokens = 0;
            }
            else if (next != tokenizer->bos_token_id && next != tokenizer->eos_token_id)
            {
                if (!in_generation)
                {
                    start_time = clock();
                    in_generation = 1;
                    gen_tokens = 0;

                    printf("<think>\n");
                    fflush(stdout);
                }
                gen_tokens++;
                printf("%s", decode(tokenizer, next));
                fflush(stdout);

            }
        }
    }
    free(prompt_tokens);
}

int main(int argc, char* argv[])
{
    omp_set_num_threads(omp_get_num_procs());

#if defined(_WIN32) || defined(_WIN64)
    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);
#endif

    char* checkpoint_path = NULL;    
    char* tokenizer_path = NULL;
    float temperature = 0.95f;             
    float topp = 0.7f;                      
    int steps = 4096;                 
    char* prompt = NULL;          
    unsigned long long rng_seed = 0;       
    char* system_prompt = NULL;          
    int ctx_length = 4096;

    if (argc >= 3) {
        checkpoint_path = argv[1];
        tokenizer_path = argv[2];
    }
    else
    {
        exit(EXIT_FAILURE);
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.7;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;     

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);        

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
