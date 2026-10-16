/*
 * pth2onnx.c — Convert RVC v2 .pth voice model to ONNX.
 *
 * Usage: pth2onnx <input.pth> <output.onnx>
 *
 * Reads the .pth file (ZIP archive containing pickle tensors),
 * derives the architecture parameters, emits the complete ONNX
 * graph, and writes the result to an .onnx file.
 */

#include "rvc_graph.h"
#include "pth_loader.h"
#include "debug.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static int write_file(const char *path, const uint8_t *data, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "pth2onnx: cannot open %s for writing\n", path);
        return -1;
    }
    if (fwrite(data, 1, len, f) != len) {
        fprintf(stderr, "pth2onnx: short write to %s\n", path);
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    /* Check for --debug flag or SPCV_DEBUG env var */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) spcv_debug_set(1);
    }
    if (getenv("SPCV_DEBUG")) spcv_debug_set(1);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.pth> <output.onnx>\n", argv[0]);
        return 1;
    }

    const char *pth_path = argv[1];
    const char *onnx_path = argv[2];

    SPCV_TRACE("main", "starting pth2onnx: %s -> %s", pth_path, onnx_path);
    printf("pth2onnx: loading %s\n", pth_path);

    /* Load the .pth file (heap-allocated: pickle_ctx_t is ~600KB) */
    pth_model_t *model = (pth_model_t *)calloc(1, sizeof(pth_model_t));
    if (!model) {
        fprintf(stderr, "pth2onnx: OOM allocating pth_model_t\n");
        return 1;
    }
    SPCV_TRACE("main", "calling pth_load...");
    if (pth_load(model, pth_path) != 0) {
        SPCV_TRACE("main", "pth_load FAILED");
        fprintf(stderr, "pth2onnx: failed to load %s\n", pth_path);
        free(model);
        return 1;
    }

    int n_tensors = model->pkl.n_tensors;
    SPCV_TRACE("main", "pth_load OK, %d tensors", n_tensors);
    printf("pth2onnx: %d tensors loaded\n", n_tensors);

    if (model->pkl.has_config) {
        pth_config_t *cfg = &model->pkl.config;
        printf("pth2onnx: config: ques=%d, filter=%d, ff=%d, p_dim=%d, heads=%d, layers=%d\n",
               cfg->ques, cfg->filter_channels, cfg->ff_dim, cfg->p_dim,
               cfg->p_heads, cfg->p_layers);
        printf("pth2onnx: upsample: n=%d, initial=%d, rates=%d/%d/%d/%d\n",
               cfg->n_upsample, cfg->upsample_initial,
               cfg->upsample_rates[0], cfg->upsample_rates[1],
               cfg->upsample_rates[2], cfg->upsample_rates[3]);
        printf("pth2onnx: sr=%d, resblock_k=%d\n", cfg->sr, cfg->resblock_kernel);
    }

    /* Derive architecture parameters */
    RvcParams params;
    int hub_dim = 768;
    if (model->pkl.has_config)
        hub_dim = model->pkl.config.ques;
    rvc_derive_params(&params, NULL, 0, hub_dim);

    /* Build weight array from pth tensors */
    RvcWeight *weights = (RvcWeight *)calloc(n_tensors, sizeof(RvcWeight));
    for (int i = 0; i < n_tensors; i++) {
        pth_tensor_t *t = &model->pkl.tensors[i];
        weights[i].name = t->name;
        weights[i].ndim = t->ndim;
        weights[i].dims = t->shape;
        weights[i].nbytes = t->size_bytes;
        /* data will be read from the .pth file when needed */
        weights[i].data = NULL;
    }

    /* Emit the graph */
    SPCV_TRACE("main", "calling rvc_emit_graph...");
    printf("pth2onnx: emitting graph...\n");
    Buf **nodes = NULL;
    size_t n_nodes = 0;
    if (rvc_emit_graph(&params, weights, n_tensors, &nodes, &n_nodes) != 0) {
        SPCV_TRACE("main", "rvc_emit_graph FAILED");
        fprintf(stderr, "pth2onnx: graph emission failed\n");
        return 1;
    }
    SPCV_TRACE("main", "rvc_emit_graph OK, %zu nodes", n_nodes);
    printf("pth2onnx: %zu nodes emitted\n", n_nodes);

    /* Build the ONNX graph */
    Buf graph;
    buf_init(&graph);

    /* Build input ValueInfos */
    Buf inputs[6];
    for (int i = 0; i < 6; i++) buf_init(&inputs[i]);

    onnx_emit_value_info(&inputs[0], "phone", ONNX_TENSOR_FLOAT,
                         (int64_t[]){1, 0, 768}, (const char*[]){NULL, "T", NULL}, 3);
    onnx_emit_value_info(&inputs[1], "phone_lengths", ONNX_TENSOR_INT64,
                         (int64_t[]){1}, NULL, 1);
    onnx_emit_value_info(&inputs[2], "pitch", ONNX_TENSOR_INT64,
                         (int64_t[]){1, 0}, (const char*[]){NULL, "T"}, 2);
    onnx_emit_value_info(&inputs[3], "pitchf", ONNX_TENSOR_FLOAT,
                         (int64_t[]){1, 0}, (const char*[]){NULL, "T"}, 2);
    onnx_emit_value_info(&inputs[4], "ds", ONNX_TENSOR_INT64,
                         (int64_t[]){1}, NULL, 1);
    onnx_emit_value_info(&inputs[5], "rnd", ONNX_TENSOR_FLOAT,
                         (int64_t[]){1, 192, 0}, (const char*[]){NULL, NULL, "T"}, 3);

    /* Build output ValueInfo */
    Buf output;
    buf_init(&output);
    onnx_emit_value_info(&output, "audio", ONNX_TENSOR_FLOAT,
                         (int64_t[]){0, 1, 0}, (const char*[]){"N", NULL, "T"}, 3);

    /* Build initializers from weight tensors.
       Each weight is a TensorProto in graph.initializer (field 5).
       We need to read the actual data from the .pth file. */
    int n_init = 0;
    Buf *inits = (Buf *)calloc(n_tensors, sizeof(Buf));
    SPCV_TRACE("main", "building %d initializers", n_tensors);
    for (int i = 0; i < n_tensors; i++) {
        pth_tensor_t *t = &model->pkl.tensors[i];
        buf_init(&inits[i]);

        /* Read tensor data */
        size_t data_size = (size_t)t->size_bytes;
        uint8_t *data = (uint8_t *)malloc(data_size);
        if (!data) {
            SPCV_TRACE("main", "OOM allocating %zu bytes for %s", data_size, t->name);
            fprintf(stderr, "pth2onnx: OOM for tensor %s (%zu bytes)\n", t->name, data_size);
            free(data);
            continue;
        }
        if (pth_read_tensor_raw(model, t, data) != 0) {
            SPCV_TRACE("main", "pth_read_tensor_raw FAILED for %s", t->name);
            fprintf(stderr, "pth2onnx: failed to read tensor %s\n", t->name);
            free(data);
            continue;
        }

        /* Determine ONNX type */
        int onnx_type = ONNX_TENSOR_FLOAT;
        if (t->dtype == 0) onnx_type = ONNX_TENSOR_FLOAT; /* fp16 → float */
        else if (t->dtype == 1) onnx_type = ONNX_TENSOR_FLOAT;
        else if (t->dtype == 2) onnx_type = ONNX_TENSOR_INT32;
        else if (t->dtype == 3) onnx_type = ONNX_TENSOR_INT64;

        /* Convert fp16 → fp32 if needed (bit manipulation, no libm) */
        if (t->dtype == 0 && onnx_type == ONNX_TENSOR_FLOAT) {
            size_t n_elems = data_size / 2;
            float *f32_data = (float *)malloc(n_elems * sizeof(float));
            if (f32_data) {
                const uint16_t *fp16 = (const uint16_t *)data;
                for (size_t j = 0; j < n_elems; j++) {
                    uint32_t h = fp16[j];
                    uint32_t sign = (h & 0x8000u) << 16;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f32_bits;
                    if (exp == 0) {
                        if (mant == 0) {
                            f32_bits = sign; /* ±0 */
                        } else {
                            /* denormal: normalize */
                            int shift = 0;
                            while (!(mant & 0x400)) { mant <<= 1; shift++; }
                            mant &= 0x3FF;
                            uint32_t new_exp = 127 - 15 - shift + 1;
                            f32_bits = sign | (new_exp << 23) | (mant << 13);
                        }
                    } else if (exp == 31) {
                        f32_bits = sign | 0x7F800000u | (mant << 13); /* Inf/NaN */
                    } else {
                        f32_bits = sign | ((exp + 112) << 23) | (mant << 13);
                    }
                    memcpy(&f32_data[j], &f32_bits, 4);
                }
                free(data);
                data = (uint8_t *)f32_data;
                data_size = n_elems * sizeof(float);
            }
        }

        onnx_emit_tensor(&inits[i], t->name, t->shape, t->ndim,
                         onnx_type, data, data_size);
        free(data);
        n_init++;
    }

    /* Build the graph */
    const Buf **init_ptrs = (const Buf **)malloc(n_init * sizeof(Buf *));
    for (int i = 0; i < n_init; i++) init_ptrs[i] = &inits[i];
    const Buf *input_ptrs[6];
    for (int i = 0; i < 6; i++) input_ptrs[i] = &inputs[i];
    const Buf *output_ptrs[1] = { &output };
    onnx_build_graph(&graph, "rvc_v2",
                     (const Buf **)nodes, n_nodes,
                     init_ptrs, n_init,
                     input_ptrs, 6,
                     output_ptrs, 1);
    free(init_ptrs);

    /* Build the model */
    Buf model_buf;
    buf_init(&model_buf);
    onnx_emit_model(&model_buf, &graph, 8, 16, "spartacvc");

    SPCV_TRACE("main", "writing %s (%zu bytes)", onnx_path, model_buf.len);
    printf("pth2onnx: writing %s (%zu bytes)\n", onnx_path, model_buf.len);
    if (write_file(onnx_path, model_buf.data, model_buf.len) != 0) {
        SPCV_TRACE("main", "write_file FAILED");
        fprintf(stderr, "pth2onnx: failed to write output\n");
        return 1;
    }

    SPCV_TRACE("main", "SUCCESS");
    printf("pth2onnx: done!\n");

    /* Cleanup */
    rvc_free_nodes(nodes, n_nodes);
    free(weights);
    free(inits);
    pth_free(model);
    free(model);
    for (int i = 0; i < 6; i++) buf_free(&inputs[i]);
    buf_free(&output);
    buf_free(&graph);
    buf_free(&model_buf);

    return 0;
}
