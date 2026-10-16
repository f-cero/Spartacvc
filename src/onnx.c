/**
 * @file    onnx.c
 * @brief   ONNX Runtime C API wrapper — implementation.
 *
 * All ONNX Runtime calls in spartacvc go through this file.
 * No other .c file includes onnxruntime_c_api.h.
 *
 * Build:
 *   -DSPCV_DEBUG          enables trace macros
 *   -I<ort>/include       ORT headers
 *   <ort>/lib/onnxruntime.lib   ORT import library (link)
 *   onnxruntime.dll must be on PATH at runtime
 *
 * Thread-safety:
 *   ORT sessions are thread-safe for concurrent Run() calls.
 *   spcv_onnx_init() / spcv_onnx_shutdown() are NOT thread-safe;
 *   call them once at startup / teardown.
 *
 * ORT C API gotchas (learned the hard way):
 *   - Run() param order: (..., out_names, OUT_LEN, out_values)
 *   - CastTypeInfoToTensorInfo returns a VIEW — free via ReleaseTypeInfo only
 *   - SessionGetInputName allocates via ORT allocator — free via alloc->Free
 *   - ONNX type 1 = FLOAT (4 bytes), type 10 = FLOAT16 (2 bytes)
 */

#include "onnx.h"
#include "debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  ORT C API include (this is the ONLY file that includes it)         */
/* ------------------------------------------------------------------ */
#include "onnxruntime_c_api.h"

#ifdef _WIN32
#include <windows.h>
#endif

/* ------------------------------------------------------------------ */
/*  Dynamic loading of onnxruntime.dll                                 */
/* ------------------------------------------------------------------ */

#ifdef _WIN32
static HMODULE g_ort_hmod = NULL;
#endif

/**
 * @brief  Get the OrtApi pointer (cached).
 *         Uses AddDllDirectory + LoadLibraryEx to ensure we load the
 *         correct onnxruntime.dll (not the System32 stub).
 * @return OrtApi* or NULL if ORT is not available
 */
static const OrtApi *ort_api(void) {
    static const OrtApi *api = NULL;
    if (!api) {
#ifdef _WIN32
        if (!g_ort_hmod) {
            /* Get our own directory and load DLL from there */
            char exe_path[MAX_PATH];
            GetModuleFileNameA(NULL, exe_path, MAX_PATH);
            char *slash = strrchr(exe_path, '\\');
            if (slash) *slash = 0;
            
            char full_path[MAX_PATH + 32];
            snprintf(full_path, sizeof(full_path), "%s\\onnxruntime.dll", exe_path);
            g_ort_hmod = LoadLibraryExA(full_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!g_ort_hmod)
                g_ort_hmod = LoadLibraryA("onnxruntime.dll");
            if (!g_ort_hmod) {
                SPCV_TRACE("onnx", "FATAL: LoadLibrary(onnxruntime.dll) failed: %lu", GetLastError());
                return NULL;
            }
            
            char loaded_from[MAX_PATH];
            GetModuleFileNameA(g_ort_hmod, loaded_from, MAX_PATH);
            SPCV_TRACE("onnx", "loaded onnxruntime.dll from: %s", loaded_from);
        }
        typedef const OrtApiBase *(*ort_get_api_base_fn)(void);
        ort_get_api_base_fn fn = (ort_get_api_base_fn)GetProcAddress(g_ort_hmod, "OrtGetApiBase");
        if (!fn) {
            SPCV_TRACE("onnx", "FATAL: GetProcAddress(OrtGetApiBase) failed: %lu", GetLastError());
            return NULL;
        }
        const OrtApiBase *base = fn();
#else
        const OrtApiBase *base = OrtGetApiBase();
#endif
        if (base) api = base->GetApi(ORT_API_VERSION);
    }
    return api;
}

/**
 * @brief  Convert a UTF-8 path to ORTCHAR_T (wchar_t on Windows).
 * @param  path       UTF-8 input
 * @param  wbuf       caller buffer
 * @param  wbuf_size  buffer size in ORTCHAR_T units
 * @return pointer to wbuf, or NULL on overflow
 */
static const ORTCHAR_T *to_ortchar(const char *path, ORTCHAR_T *wbuf, size_t wbuf_size) {
#ifdef _WIN32
    size_t len = strlen(path);
    if (len + 1 > wbuf_size) return NULL;
    for (size_t i = 0; i < len; i++) wbuf[i] = (wchar_t)(unsigned char)path[i];
    wbuf[len] = 0;
    return wbuf;
#else
    (void)wbuf; (void)wbuf_size;
    return path;
#endif
}

/**
 * @brief  Bytes per element for an ONNX tensor type.
 * @param  onnx_type  ONNX TensorElementDataType enum value
 * @return size in bytes (4 for FLOAT, 8 for INT64, etc.)
 */
static size_t elem_size(int onnx_type) {
    switch (onnx_type) {
        case 1:  return 4;   /* FLOAT */
        case 2:  return 1;   /* UINT8 */
        case 3:  return 1;   /* INT8 */
        case 4:  return 2;   /* UINT16 */
        case 5:  return 2;   /* INT16 */
        case 6:  return 4;   /* INT32 */
        case 7:  return 8;   /* INT64 */
        case 9:  return 1;   /* BOOL */
        case 10: return 2;   /* FLOAT16 */
        case 11: return 8;   /* DOUBLE */
        default: return 4;
    }
}

/**
 * @brief  Read a tensor's type info into our descriptor.
 *
 * IMPORTANT: CastTypeInfoToTensorInfo returns a VIEW into the OrtTypeInfo.
 * Do NOT call ReleaseTensorTypeAndShapeInfo on the view.
 * Only call ReleaseTypeInfo on the original.
 *
 * @param  api    ORT API
 * @param  sess   ORT session
 * @param  idx    tensor index
 * @param  is_in  1 = input, 0 = output
 * @param  td     [out] descriptor to fill
 * @return  0 on success, -1 on failure
 */
static int read_tensor_desc(const OrtApi *api, OrtSession *sess,
                            size_t idx, int is_in, spcv_tensor_desc *td) {
    OrtStatus *s;
    memset(td, 0, sizeof(*td));

    /* Name */
    char *name = NULL;
    OrtAllocator *alloc = NULL;
    s = api->GetAllocatorWithDefaultOptions(&alloc);
    if (s) { api->ReleaseStatus(s); return -1; }

    if (is_in)
        s = api->SessionGetInputName(sess, idx, alloc, &name);
    else
        s = api->SessionGetOutputName(sess, idx, alloc, &name);
    if (s) { api->ReleaseStatus(s); return -1; }
    strncpy(td->name, name, sizeof(td->name) - 1);
    alloc->Free(alloc, name);

    /* Type info */
    OrtTypeInfo *ti = NULL;
    if (is_in)
        s = api->SessionGetInputTypeInfo(sess, idx, &ti);
    else
        s = api->SessionGetOutputTypeInfo(sess, idx, &ti);
    if (s) { api->ReleaseStatus(s); return -1; }

    OrtTensorTypeAndShapeInfo *tsi = NULL;
    s = api->CastTypeInfoToTensorInfo(ti, &tsi);
    if (s || !tsi) { api->ReleaseTypeInfo(ti); return -1; }

    ONNXTensorElementDataType et;
    api->GetTensorElementType(tsi, &et);
    td->elem_type = (int)et;

    size_t dc = 0;
    api->GetDimensionsCount(tsi, &dc);
    td->ndim = (int)dc;
    if (dc > 0 && dc <= 8)
        api->GetDimensions(tsi, td->dims, dc);
    td->byte_size = 0; /* dynamic */

    /* Free ONLY the type info — tsi is a view, not a separate object */
    api->ReleaseTypeInfo(ti);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Global state                                                       */
/* ------------------------------------------------------------------ */

static int     g_ort_initialized = 0;
static OrtEnv *g_ort_env        = NULL;
static spcv_ep g_ep             = SPCV_EP_AUTO;

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

const char *spcv_onnx_status_str(spcv_onnx_status st) {
    switch (st) {
        case SPCV_ONNX_OK:        return "ok";
        case SPCV_ONNX_ERR_LOAD:  return "model load failed";
        case SPCV_ONNX_ERR_INPUT: return "bad input";
        case SPCV_ONNX_ERR_RUN:   return "inference failed";
        case SPCV_ONNX_ERR_ALLOC: return "allocation failed";
        case SPCV_ONNX_ERR_API:   return "ORT API error";
        default:                  return "unknown error";
    }
}

int spcv_onnx_init(void) {
    if (g_ort_initialized) return SPCV_ONNX_OK;

    const OrtApi *api = ort_api();
    if (!api) {
        SPCV_TRACE("onnx", "FATAL: ORT API unavailable");
        return SPCV_ONNX_ERR_API;
    }

    OrtStatus *s = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "spartacvc", &g_ort_env);
    if (s) {
        SPCV_TRACE("onnx", "FATAL: CreateEnv: %s", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        return SPCV_ONNX_ERR_API;
    }

    g_ort_initialized = 1;
    SPCV_TRACE("onnx", "ORT env created (api=%d)", ORT_API_VERSION);
    return SPCV_ONNX_OK;
}

void spcv_onnx_set_ep(spcv_ep ep) {
    g_ep = ep;
}

void spcv_onnx_shutdown(void) {
    if (!g_ort_initialized) return;
    const OrtApi *api = ort_api();
    if (api && g_ort_env) api->ReleaseEnv(g_ort_env);
    g_ort_env = NULL;
    g_ort_initialized = 0;
#ifdef _WIN32
    if (g_ort_hmod) { FreeLibrary(g_ort_hmod); g_ort_hmod = NULL; }
#endif
    SPCV_TRACE("onnx", "ORT env released");
}

/* ------------------------------------------------------------------ */
/*  Session load / free                                                */
/* ------------------------------------------------------------------ */

int spcv_onnx_load(const char *path, spcv_onnx_session **sess_out) {
    if (!path || !sess_out) return SPCV_ONNX_ERR_INPUT;
    *sess_out = NULL;

    const OrtApi *api = ort_api();
    if (!api) return SPCV_ONNX_ERR_API;

    SPCV_TRACE("onnx", "loading: %s", path);

    OrtSessionOptions *opts = NULL;
    OrtStatus *s = api->CreateSessionOptions(&opts);
    if (s) { api->ReleaseStatus(s); return SPCV_ONNX_ERR_API; }

    /* Execution provider selection */
    if (g_ep == SPCV_EP_DML || g_ep == SPCV_EP_AUTO) {
        s = api->SessionOptionsAppendExecutionProvider(
            opts, "DmlExecutionProvider", NULL, NULL, 0);
        if (s) {
            SPCV_TRACE("onnx", "DML EP unavailable, falling back to CPU");
            api->ReleaseStatus(s);
            if (g_ep == SPCV_EP_DML) {
                api->ReleaseSessionOptions(opts);
                return SPCV_ONNX_ERR_LOAD;
            }
        } else {
            SPCV_TRACE("onnx", "DML EP enabled");
        }
    }

    ORTCHAR_T wbuf[2048];
    const ORTCHAR_T *wpath = to_ortchar(path, wbuf, sizeof(wbuf)/sizeof(wbuf[0]));
    if (!wpath) { api->ReleaseSessionOptions(opts); return SPCV_ONNX_ERR_INPUT; }

    OrtSession *session = NULL;
    s = api->CreateSession(g_ort_env, wpath, opts, &session);
    if (s) {
        SPCV_TRACE("onnx", "FAIL: CreateSession: %s", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        api->ReleaseSessionOptions(opts);
        return SPCV_ONNX_ERR_LOAD;
    }

    spcv_onnx_session *sess = (spcv_onnx_session *)calloc(1, sizeof(*sess));
    if (!sess) {
        api->ReleaseSession(session);
        api->ReleaseSessionOptions(opts);
        return SPCV_ONNX_ERR_ALLOC;
    }
    sess->ort_session = session;

    s = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault,
                                  (OrtMemoryInfo **)&sess->ort_meminfo);
    if (s) {
        api->ReleaseStatus(s);
        api->ReleaseSession(session);
        free(sess);
        api->ReleaseSessionOptions(opts);
        return SPCV_ONNX_ERR_API;
    }

    s = api->GetAllocatorWithDefaultOptions((OrtAllocator **)&sess->ort_alloc);
    if (s) {
        api->ReleaseStatus(s);
        api->ReleaseMemoryInfo((OrtMemoryInfo *)sess->ort_meminfo);
        api->ReleaseSession(session);
        free(sess);
        api->ReleaseSessionOptions(opts);
        return SPCV_ONNX_ERR_API;
    }

    /* Enumerate inputs */
    size_t n_in = 0;
    s = api->SessionGetInputCount(session, &n_in);
    if (s) { api->ReleaseStatus(s); goto fail; }
    sess->n_inputs = (int)n_in;
    for (size_t i = 0; i < n_in && i < 8; i++) {
        if (read_tensor_desc(api, session, i, 1, &sess->inputs[i]) != 0) goto fail;
        SPCV_TRACE("onnx", "  in[%d]: %s type=%d ndim=%d", (int)i, sess->inputs[i].name, sess->inputs[i].elem_type, sess->inputs[i].ndim);
    }

    /* Enumerate outputs */
    size_t n_out = 0;
    s = api->SessionGetOutputCount(session, &n_out);
    if (s) { api->ReleaseStatus(s); goto fail; }
    sess->n_outputs = (int)n_out;
    for (size_t i = 0; i < n_out && i < 8; i++) {
        if (read_tensor_desc(api, session, i, 0, &sess->outputs[i]) != 0) goto fail;
        SPCV_TRACE("onnx", "  out[%d]: %s type=%d ndim=%d", (int)i, sess->outputs[i].name, sess->outputs[i].elem_type, sess->outputs[i].ndim);
    }

    *sess_out = sess;
    api->ReleaseSessionOptions(opts);
    SPCV_TRACE("onnx", "loaded: %d in, %d out", sess->n_inputs, sess->n_outputs);
    return SPCV_ONNX_OK;

fail:
    if (sess->ort_meminfo) api->ReleaseMemoryInfo((OrtMemoryInfo *)sess->ort_meminfo);
    api->ReleaseSession(session);
    free(sess);
    api->ReleaseSessionOptions(opts);
    return SPCV_ONNX_ERR_LOAD;
}

void spcv_onnx_free(spcv_onnx_session *sess) {
    if (!sess) return;
    const OrtApi *api = ort_api();
    if (api) {
        if (sess->ort_session) api->ReleaseSession((OrtSession *)sess->ort_session);
        if (sess->ort_meminfo) api->ReleaseMemoryInfo((OrtMemoryInfo *)sess->ort_meminfo);
    }
    free(sess);
    SPCV_TRACE("onnx", "session freed");
}

/* ------------------------------------------------------------------ */
/*  Inference                                                          */
/* ------------------------------------------------------------------ */

int spcv_onnx_run(spcv_onnx_session *sess,
                  const spcv_onnx_input  *inputs,  int n_inputs,
                  spcv_onnx_output     *outputs, int n_outputs) {
    if (!sess || !inputs || !outputs) return SPCV_ONNX_ERR_INPUT;
    const OrtApi *api = ort_api();
    if (!api) return SPCV_ONNX_ERR_API;

    SPCV_TRACE("onnx", "run: %d in, %d out", n_inputs, n_outputs);

    /* Build OrtValue* for each input */
    OrtValue *ort_inputs[8] = {NULL};
    int n_created = 0;

    for (int i = 0; i < n_inputs && i < 8; i++) {
        const spcv_onnx_input *in = &inputs[i];
        size_t byte_size = elem_size(in->elem_type);
        for (int d = 0; d < in->ndim; d++) byte_size *= (size_t)in->dims[d];

        OrtStatus *s = api->CreateTensorWithDataAsOrtValue(
            (OrtMemoryInfo *)sess->ort_meminfo,
            (void *)in->data, byte_size,
            in->dims, (size_t)in->ndim,
            (ONNXTensorElementDataType)in->elem_type,
            &ort_inputs[i]);
        if (s) {
            SPCV_TRACE("onnx", "FAIL: CreateTensor[%d] (%s): %s", i, in->name, api->GetErrorMessage(s));
            api->ReleaseStatus(s);
            for (int j = 0; j < n_created; j++) api->ReleaseValue(ort_inputs[j]);
            return SPCV_ONNX_ERR_INPUT;
        }
        n_created++;
    }

    const char *in_names[8];
    for (int i = 0; i < n_inputs && i < 8; i++) in_names[i] = inputs[i].name;
    const char *out_names[8];
    for (int i = 0; i < n_outputs && i < 8; i++) out_names[i] = outputs[i].name;

    OrtValue *ort_outputs[8] = {NULL};

    /*
     * CRITICAL: Run() param order is:
     *   (session, run_opts, in_names, in_values, in_len,
     *    out_names, OUT_LEN, out_values)
     * NOT (out_names, out_values, OUT_LEN)
     */
    SPCV_TRACE("onnx", "  ORT Run...");
    OrtStatus *s = api->Run(
        (OrtSession *)sess->ort_session,
        NULL,
        in_names,  ort_inputs,  (size_t)n_inputs,
        out_names, (size_t)n_outputs, ort_outputs);

    if (s) {
        SPCV_TRACE("onnx", "FAIL: Run: %s", api->GetErrorMessage(s));
        api->ReleaseStatus(s);
        for (int i = 0; i < n_created; i++) api->ReleaseValue(ort_inputs[i]);
        return SPCV_ONNX_ERR_RUN;
    }

    for (int i = 0; i < n_outputs && i < 8; i++)
        outputs[i].ort_value = ort_outputs[i];

    for (int i = 0; i < n_created; i++) api->ReleaseValue(ort_inputs[i]);
    SPCV_TRACE("onnx", "  Run ok");
    return SPCV_ONNX_OK;
}

const void *spcv_onnx_output_data(const spcv_onnx_output *out) {
    if (!out || !out->ort_value) return NULL;
    const OrtApi *api = ort_api();
    if (!api) return NULL;
    void *data = NULL;
    if (api->GetTensorMutableData((OrtValue *)out->ort_value, &data) == NULL)
        return data;
    return NULL;
}

void spcv_onnx_output_shape(const spcv_onnx_output *out, int *ndim, int64_t *dims) {
    if (ndim) *ndim = 0;
    if (!out || !out->ort_value) return;
    const OrtApi *api = ort_api();
    if (!api) return;
    OrtTensorTypeAndShapeInfo *tsi = NULL;
    if (api->GetTensorTypeAndShape((OrtValue *)out->ort_value, &tsi) != NULL) return;
    size_t dc = 0;
    api->GetDimensionsCount(tsi, &dc);
    if (ndim) *ndim = (int)dc;
    if (dims && dc > 0 && dc <= 8)
        api->GetDimensions(tsi, dims, dc);
    api->ReleaseTensorTypeAndShapeInfo(tsi);
}

void spcv_onnx_free_output(spcv_onnx_output *out) {
    if (!out || !out->ort_value) return;
    const OrtApi *api = ort_api();
    if (api) api->ReleaseValue((OrtValue *)out->ort_value);
    out->ort_value = NULL;
}
