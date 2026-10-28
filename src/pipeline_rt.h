// pipeline_rt.h — Real-time RVC pipeline worker thread.
//
// Captures audio from WASAPI, runs the RVC pipeline in chunks,
// and renders the result to a WASAPI output device.
#ifndef SPCV_PIPELINE_RT_H
#define SPCV_PIPELINE_RT_H

#ifdef __cplusplus
extern "C" {
#endif

/** Start the real-time pipeline.
 *  @param model_path    Path to .gguf or .pth model.
 *  @param be_id         Backend: 0=CPU, 1=CUDA, 3=Vulkan.
 *  @param gpu_id        GPU device index.
 *  @param in_dev        Input device ID (WASAPI), or NULL for default.
 *  @param out_dev       Output device ID (WASAPI), or NULL for default.
 *  @param hubert_path   Path to HuBERT ONNX model.
 *  @param rmvpe_path    Path to RMVPE ONNX model.
 *  @return              Thread handle, or NULL on failure. */
void * spcv_rt_start(const wchar_t *model_path, int be_id, int gpu_id,
                     const wchar_t *in_dev, const wchar_t *out_dev,
                     const wchar_t *hubert_path, const wchar_t *rmvpe_path);

/** Stop the real-time pipeline and wait for the thread to exit. */
void spcv_rt_stop(void *thread);

/** Is the pipeline running? */
int spcv_rt_is_running(void);

#ifdef __cplusplus
}
#endif

#endif
