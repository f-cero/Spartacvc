/**
 * @file    main.c
 * @brief   spartacvc — RVC voice conversion in pure C.
 *
 * Pipeline:
 *   1. Read WAV (16-bit PCM mono)
 *   2. Resample to 16 kHz
 *   3. HuBERT ONNX -> embeddings [1, T, 256|768]
 *   4. Resample embeddings to model fps (per-model, NOT hardcoded)
 *   5. Mel spectrogram (16 kHz, 1024 FFT, 160 hop, 128 mels)
 *   6. RMVPE ONNX -> f0 [Hz] per frame
 *   7. Pitch shift (semitones)
 *   8. Resample f0 to model fps
 *   9. Quantize f0 to RVC 1..255 scale
 *  10. Voice decoder ONNX -> PCM @ model_sr
 *  11. Write WAV @ model_sr
 *
 * Per-model parameters (sr, fps) are derived from the voice model,
 * never hardcoded.  256-dim models → 32kHz/80fps, 768-dim → 48kHz/100fps.
 *
 * Usage:
 *   spartacvc.exe input.wav --voice model.onnx --hubert hubert.onnx
 *                  [--rmvpe rmvpe.onnx] [--pitch +12] [--out output.wav]
 *                  [--config config.json] [--debug]
 *
 * Config file (JSON) keys:
 *   voice_model, hubert_model, rmvpe_model,
 *   pitch_shift_semitones, output_dir
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onnx.h"
#include "onnx_init.h"
#include "mel.h"
#include "f0.h"
#include "debug.h"

/* ------------------------------------------------------------------ */
/*  WAV I/O                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    float *data;
    int64_t n;
    int sr;
} spcv_wav;

/**
 * @brief  Read a WAV file (16-bit or 32-bit float PCM, mono or stereo).
 *
 * @param  path  file path
 * @param  w     [out] WAV data (caller frees w->data)
 * @return  0 on success, -1 on failure
 */
int spcv_wav_read(const char *path, spcv_wav *w) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) { fclose(f); return -1; }
    uint16_t ch = 0, bits = 0;
    uint32_t sr = 0, dsz = 0;
    int has_fmt = 0, has_data = 0;
    uint8_t *raw = NULL;

    while (!has_data && !feof(f)) {
        char cn[4]; uint32_t cs;
        if (fread(cn, 1, 4, f) != 4) break;
        if (fread(&cs, 4, 1, f) != 1) break;
        if (memcmp(cn, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (fread(fmt, 1, 16, f) != 16) break;
            ch = fmt[2] | (fmt[3] << 8);
            sr = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits = fmt[14] | (fmt[15] << 8);
            if (cs > 16) fseek(f, cs - 16, SEEK_CUR);
            has_fmt = 1;
        } else if (memcmp(cn, "data", 4) == 0) {
            dsz = cs;
            raw = malloc(dsz);
            if (fread(raw, 1, dsz, f) != dsz) { free(raw); fclose(f); return -1; }
            has_data = 1;
        } else {
            fseek(f, cs, SEEK_CUR);
        }
    }
    fclose(f);
    if (!has_fmt || !has_data) { free(raw); return -1; }

    int64_t ns = dsz / (ch * (bits / 8));
    w->n = ns;
    w->sr = (int)sr;
    w->data = malloc(ns * sizeof(float));
    if (bits == 16) {
        int16_t *p = (int16_t *)raw;
        for (int64_t i = 0; i < ns; i++) w->data[i] = (float)p[i * ch] / 32768.0f;
    } else if (bits == 32) {
        float *p = (float *)raw;
        for (int64_t i = 0; i < ns; i++) w->data[i] = p[i * ch];
    } else if (bits == 8) {
        uint8_t *p = raw;
        for (int64_t i = 0; i < ns; i++) w->data[i] = ((float)p[i * ch] - 128.0f) / 128.0f;
    } else { free(raw); free(w->data); return -1; }
    free(raw);
    return 0;
}

/**
 * @brief  Write 16-bit PCM mono WAV.
 *
 * @param  path  output file path
 * @param  data  float samples in [-1, 1]
 * @param  n     number of samples
 * @param  sr    sample rate
 * @return  0 on success, -1 on failure
 */
int spcv_wav_write(const char *path, const float *data, int64_t n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t dsz = (uint32_t)(n * 2);
    uint32_t rsz = 36 + dsz;
    fwrite("RIFF", 1, 4, f); fwrite(&rsz, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    uint32_t fsz = 16; uint16_t af = 1, ch = 1, bp = 16;
    uint32_t br = sr * 2, ba = 2;
    fwrite("fmt ", 1, 4, f); fwrite(&fsz, 4, 1, f);
    fwrite(&af, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f); fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f); fwrite(&bp, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dsz, 4, 1, f);
    for (int64_t i = 0; i < n; i++) {
        int32_t s = (int32_t)(data[i] * 32767.0f);
        if (s > 32767) s = 32767; if (s < -32768) s = -32768;
        int16_t v = (int16_t)s;
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Resampling helpers                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief  Linear-interpolation resample of a 1-D signal.
 */
static float *resample_1d(const float *in, int64_t n_in, int64_t n_out) {
    if (n_in == n_out) return (float *)in;
    float *out = malloc(n_out * sizeof(float));
    if (!out) return NULL;
    double ratio = (double)n_in / n_out;
    for (int64_t i = 0; i < n_out; i++) {
        double pos = i * ratio;
        int64_t i0 = (int64_t)pos;
        int64_t i1 = i0 + 1;
        double frac = pos - i0;
        if (i1 >= n_in) i1 = n_in - 1;
        out[i] = (float)(in[i0] * (1.0 - frac) + in[i1] * frac);
    }
    return out;
}

/**
 * @brief  Linear-interpolation resample of embeddings along time axis.
 */
static float *resample_embed(const float *emb, int64_t t_in, int64_t t_out, int dim) {
    if (t_in == t_out) return (float *)emb;
    float *out = malloc(t_out * dim * sizeof(float));
    if (!out) return NULL;
    double ratio = (double)t_in / t_out;
    for (int64_t t = 0; t < t_out; t++) {
        double pos = t * ratio;
        int64_t i0 = (int64_t)pos;
        int64_t i1 = i0 + 1;
        double frac = pos - i0;
        if (i0 >= t_in) i0 = t_in - 1;
        if (i1 >= t_in) i1 = t_in - 1;
        for (int d = 0; d < dim; d++)
            out[t * dim + d] = (float)(emb[i0 * dim + d] * (1.0 - frac) +
                                       emb[i1 * dim + d] * frac);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const char *input = NULL, *voice_path = NULL, *hubert_path = NULL;
    const char *rmvpe_path = NULL;
    const char *output = NULL;
    const char *config_path = NULL;
    int pitch_st = 0;
    int pitch_from_config = 0;
    int debug = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--voice")) { if (++i < argc) voice_path = argv[i]; }
        else if (!strcmp(argv[i], "--hubert")) { if (++i < argc) hubert_path = argv[i]; }
        else if (!strcmp(argv[i], "--rmvpe")) { if (++i < argc) rmvpe_path = argv[i]; }
        else if (!strcmp(argv[i], "--pitch")) { if (++i < argc) pitch_st = atoi(argv[i]); }
        else if (!strcmp(argv[i], "--out")) { if (++i < argc) output = argv[i]; }
        else if (!strcmp(argv[i], "--config")) { if (++i < argc) config_path = argv[i]; }
        else if (!strcmp(argv[i], "--debug")) debug = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("spartacvc — RVC voice conversion in pure C\n");
            printf("usage: spartacvc <input.wav> --voice <v.onnx> --hubert <h.onnx>\n");
            printf("       [--rmvpe <r.onnx>] [--pitch +N] [--out <o.wav>]\n");
            printf("       [--config <config.json>] [--debug]\n");
            return 0;
        }
        else if (argv[i][0] != '-') { if (!input) input = argv[i]; }
    }

    /* Load config file (fills in any missing values) */
    if (config_path) {
        FILE *cf = fopen(config_path, "r");
        if (!cf) {
            fprintf(stderr, "error: cannot open config: %s\n", config_path);
            return 1;
        }
        char line[1024];
        while (fgets(line, sizeof(line), cf)) {
            char key[256], val[512];
            if (sscanf(line, " \"%255[^\"]\" : \"%511[^\"]\"", key, val) == 2) {
                if (!strcmp(key, "voice_model") && !voice_path) voice_path = strdup(val);
                else if (!strcmp(key, "hubert_model") && !hubert_path) hubert_path = strdup(val);
                else if (!strcmp(key, "rmvpe_model") && !rmvpe_path) rmvpe_path = strdup(val);
                else if (!strcmp(key, "pitch_shift_semitones")) { pitch_st = atoi(val); pitch_from_config = 1; }
                else if (!strcmp(key, "output_dir") && !output) { /* used later */ }
            }
        }
        fclose(cf);
        SPCV_TRACE("main", "config loaded: %s", config_path);
    }

    if (!rmvpe_path) rmvpe_path = "models/ozada/rmvpe.onnx";

    if (!input || !voice_path || !hubert_path) {
        fprintf(stderr, "error: need input.wav --voice <v.onnx> --hubert <h.onnx>\n");
        return 1;
    }

    if (debug) spcv_debug_set(1);
    SPCV_TRACE("main", "spartacvc starting");

    /* 1. ORT init */
    if (spcv_onnx_init() != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: ONNX Runtime init failed\n");
        return 1;
    }

    /* 2. Load voice model first — its config determines fps and sr */
    spcv_onnx_session *voice = NULL;
    SPCV_TRACE("main", "loading voice: %s", voice_path);
    if (spcv_onnx_load(voice_path, &voice) != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: voice load\n"); return 1;
    }

    /* 3. Derive per-model parameters from the voice model */
    int model_sr = 0, model_fps = 0, model_dim = 0;
    if (spcv_voice_model_config(voice, &model_sr, &model_fps, &model_dim) != 0) {
        fprintf(stderr, "FATAL: cannot determine voice model config\n");
        spcv_onnx_free(voice);
        return 1;
    }
    SPCV_TRACE("main", "model: sr=%d fps=%d dim=%d", model_sr, model_fps, model_dim);

    /* 4. Load HuBERT + RMVPE */
    spcv_onnx_session *hub = NULL, *rmvpe = NULL;
    SPCV_TRACE("main", "loading HuBERT: %s", hubert_path);
    if (spcv_onnx_load(hubert_path, &hub) != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: HuBERT load\n"); return 1;
    }
    SPCV_TRACE("main", "loading RMVPE: %s", rmvpe_path);
    if (spcv_onnx_load(rmvpe_path, &rmvpe) != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: RMVPE load\n"); return 1;
    }

    /* 5. Read WAV */
    spcv_wav w;
    if (spcv_wav_read(input, &w) != 0) {
        fprintf(stderr, "FATAL: WAV read: %s\n", input); return 1;
    }
    SPCV_TRACE("main", "input: %lld samples @ %d Hz", (long long)w.n, w.sr);

    /* 6. Resample to 16 kHz (HuBERT input rate) */
    float *audio16 = resample_1d(w.data, w.n,
                                  (int64_t)((double)w.n * 16000.0 / w.sr));
    int64_t n16 = (int64_t)((double)w.n * 16000.0 / w.sr);
    SPCV_TRACE("main", "resampled: %lld samples @ 16 kHz (%.2fs)",
               (long long)n16, (double)n16 / 16000.0);

    /* 7. HuBERT inference */
    SPCV_TRACE("main", "HuBERT inference...");
    spcv_onnx_input hin = {
        .name = hub->inputs[0].name, .data = audio16,
        .ndim = 3, .dims = {1, 1, n16}, .elem_type = 1
    };
    spcv_onnx_output hout = { .name = hub->outputs[0].name,
                              .buf = NULL, .buf_size = 0, .ort_value = NULL };
    if (spcv_onnx_run(hub, &hin, 1, &hout, 1) != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: HuBERT run\n"); return 1;
    }
    int h_ndim; int64_t h_dims[8];
    spcv_onnx_output_shape(&hout, &h_ndim, h_dims);
    const float *emb = (const float *)spcv_onnx_output_data(&hout);
    if (!emb) { fprintf(stderr, "FATAL: HuBERT no output\n"); return 1; }
    int64_t t_hub = h_dims[1];
    int emb_dim = (int)h_dims[2];
    SPCV_TRACE("main", "HuBERT: T=%lld dim=%d", (long long)t_hub, emb_dim);

    /* 8. Target frame count from model fps */
    int64_t t_model = (int64_t)((double)n16 * model_fps / 16000.0);
    SPCV_TRACE("main", "target: %lld frames @ %d fps (dim=%d)",
               (long long)t_model, model_fps, emb_dim);

    /* 9. Mel spectrogram (16 kHz audio) */
    SPCV_TRACE("main", "mel spectrogram...");
    int mel_frames = (int)spcv_mel_n_frames(n16);
    float *mel = malloc((size_t)mel_frames * 128 * sizeof(float));
    if (!mel) { fprintf(stderr, "FATAL: mel alloc\n"); return 1; }
    if (spcv_mel_compute(audio16, n16, mel, 128) < 0) {
        fprintf(stderr, "FATAL: mel\n"); return 1;
    }
    SPCV_TRACE("main", "mel: %d frames x 128", mel_frames);

    /* 10. Resample HuBERT embeddings to model fps */
    float *emb_model = resample_embed(emb, t_hub, t_model, emb_dim);
    spcv_onnx_free_output(&hout);
    free(audio16);
    free(w.data);

    /* 11. f0 extraction (RMVPE) */
    SPCV_TRACE("main", "f0 extraction...");
    float *f0 = malloc((size_t)mel_frames * sizeof(float));
    if (!f0) { fprintf(stderr, "FATAL: f0 alloc\n"); return 1; }
    if (spcv_f0_extract(mel, mel_frames, rmvpe, f0) != 0) {
        fprintf(stderr, "FATAL: f0\n"); return 1;
    }
    free(mel);

    /* 12. Pitch shift */
    spcv_f0_shift(f0, mel_frames, pitch_st);

    /* 13. Resample f0 to model fps */
    float *f0_model = resample_1d(f0, mel_frames, t_model);
    free(f0);

    /* 14. Quantize f0 to RVC 1..255 scale */
    int64_t *pitch = malloc((size_t)t_model * sizeof(int64_t));
    spcv_f0_quantize(f0_model, (int)t_model, pitch);

    /* 15. Voice decoder (6 inputs) */
    SPCV_TRACE("main", "voice decoder...");

    int64_t phone_lengths = t_model;
    int64_t ds = 0;
    int rnd_dim = 192;
    float *rnd = malloc((size_t)rnd_dim * t_model * sizeof(float));
    srand(42);
    for (size_t i = 0; i < (size_t)rnd_dim * t_model; i++)
        rnd[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);

    spcv_onnx_input vin[6] = {
        { .name = voice->inputs[0].name, .data = emb_model,
          .ndim = 3, .dims = {1, t_model, emb_dim}, .elem_type = 1 },
        { .name = voice->inputs[1].name, .data = &phone_lengths,
          .ndim = 1, .dims = {1}, .elem_type = 7 },
        { .name = voice->inputs[2].name, .data = pitch,
          .ndim = 2, .dims = {1, t_model}, .elem_type = 7 },
        { .name = voice->inputs[3].name, .data = f0_model,
          .ndim = 2, .dims = {1, t_model}, .elem_type = 1 },
        { .name = voice->inputs[4].name, .data = &ds,
          .ndim = 1, .dims = {1}, .elem_type = 7 },
        { .name = voice->inputs[5].name, .data = rnd,
          .ndim = 3, .dims = {1, rnd_dim, t_model}, .elem_type = 1 }
    };
    spcv_onnx_output vout = { .name = voice->outputs[0].name,
                              .buf = NULL, .buf_size = 0, .ort_value = NULL };

    if (spcv_onnx_run(voice, vin, 6, &vout, 1) != SPCV_ONNX_OK) {
        fprintf(stderr, "FATAL: voice run\n"); return 1;
    }

    int v_ndim; int64_t v_dims[8];
    spcv_onnx_output_shape(&vout, &v_ndim, v_dims);
    const float *pcm = (const float *)spcv_onnx_output_data(&vout);
    if (!pcm) { fprintf(stderr, "FATAL: voice no output\n"); return 1; }

    int64_t out_n = 1;
    for (int i = 0; i < v_ndim; i++) out_n *= v_dims[i];
    SPCV_TRACE("main", "voice output: %lld samples @ %d Hz",
               (long long)out_n, model_sr);

    /* 16. Write WAV at model's native sample rate */
    if (!output) {
        char buf[512];
        const char *base = strrchr(input, '/');
        if (!base) base = strrchr(input, '\\');
        if (!base) base = input;
        else base++;
        snprintf(buf, sizeof(buf), "%s_converted.wav", base);
        output = buf;
    }
    if (spcv_wav_write(output, pcm, out_n, model_sr) != 0) {
        fprintf(stderr, "FATAL: WAV write\n"); return 1;
    }
    SPCV_TRACE("main", "wrote: %s (%lld samples @ %d Hz)",
               output, (long long)out_n, model_sr);

    /* Cleanup */
    spcv_onnx_free_output(&vout);
    free(emb_model);
    free(pitch);
    free(f0_model);
    free(rnd);
    spcv_onnx_free(hub);
    spcv_onnx_free(voice);
    spcv_onnx_free(rmvpe);
    spcv_onnx_shutdown();

    printf("PASS: %s -> %s (%.2fs @ %d Hz)\n",
           input, output, (double)out_n / model_sr, model_sr);
    return 0;
}
