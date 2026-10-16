# pth2onnx — Implementation Plan

## Goal
Write a C tool that reads an RVC `.pth` voice model and writes a valid `.onnx` file
loadable by ONNX Runtime. No Python. No tensor-swap. Full graph emission from scratch.

## Architecture Parameters (derived from .pth, NOT hardcoded)

All params come from the `config` list in the .pth pickle:

| Index | Name | Example (nahida) | Example (tsv2) |
|-------|------|-------------------|----------------|
| 0 | spec_channels | 1025 | 1025 |
| 1 | segment_size | 32 | 32 |
| 2 | inter_channels | 192 | 192 |
| 3 | hidden_channels | 192 | 192 |
| 4 | filter_channels | 768 | 768 |
| 5 | n_heads | 2 | 2 |
| 6 | n_layers | 6 | 6 |
| 7 | kernel_size | 3 | 3 |
| 8 | p_dropout | 0 | 0 |
| 9 | resblock | '1' | '1' |
| 10 | resblock_kernel_sizes | [3,7,11] | [3,7,11] |
| 11 | resblock_dilation_sizes | [[1,3,5],[1,3,5],[1,3,5]] | same |
| 12 | upsample_rates | [12,10,2,2] | [10,10,2,2] |
| 13 | upsample_initial_channel | 512 | 512 |
| 14 | upsample_kernel_sizes | [24,20,4,4] | [16,16,4,4] |
| 15 | spk_embed_dim | 109 | 109 |
| 16 | gin_channels | 256 | 256 |
| 17 | sr | 48000 | 40000 |

**HuBERT dim**: derived from `enc_p.emb_phone.weight[1]` (768 or 256)
**k_channels**: hidden_channels / n_heads (e.g. 192/2 = 96)
**window_size**: 10 (fixed in Encoder constructor)
**n_flows**: 4 (fixed: flows are indices 0,2,4,6 in the ModuleList)

## Model Forward Pass (the graph we must emit)

```
INPUTS: phone[1,T,hub_dim], phone_lengths[1], pitch[1,T], pitchf[1,T], ds[1], rnd[1,192,T]
OUTPUT: audio[1,1,samples]

1. SPEAKER EMBEDDING
   g = emb_g.weight[ds]          → Gather(emb_g.weight, ds, axis=0)
   g = g.unsqueeze(0).transpose(1,2)  → [1, gin_channels, 1]

2. ENCODER (TextEncoder768 or TextEncoder256)
   x = emb_phone(phone)                          → Linear: MatMul + Add
   x = x + emb_pitch(pitch)                      → Gather + Transpose + Add
   x = x * sqrt(hidden_channels)                 → Mul
   x = LeakyReLU(x, 0.1)                        → LeakyRelu
   x = transpose(x, 1, -1)                      → Transpose [0,2,1]
   x_mask = sequence_mask(phone_lengths)         → Range, Less, Cast, Unsqueeze
   x = x * x_mask
   x = encoder(x, x_mask)                        → 6x (attn + norm + ffn + norm)
   stats = proj(x) * x_mask                      → Conv1d(k=1) + Mul
   m, logs = split(stats, inter_channels, dim=1) → Split

3. LATENT
   z_p = (m + exp(logs) * rnd) * x_mask          → Exp, Mul, Add, Mul

4. FLOW (reverse)
   for i in [6, 4, 2, 0]:                        → reversed order
     x0, x1 = split(z, half_channels, dim=1)     → Split
     h = pre(x0) * x_mask                        → Conv1d(k=1) + Mul
     h = WN(h, x_mask, g)                        → WaveNet
     m = post(h) * x_mask                        → Conv1d(k=1) + Mul
     x1 = (x1 - m) * x_mask                      → Sub, Mul
     z = concat(x0, x1, dim=1)                   → Concat
     z = flip(z, dim=1)                          → Slice with negative step

5. DECODER (GeneratorNSF)
   har_source = SineGen(pitchf, upp)             → Sin, CumSum, etc.
   har_source = tanh(linear(har_source))         → MatMul + Add + Tanh
   har_source = transpose(har_source)            → Transpose
   x = conv_pre(z * x_mask)                      → Conv1d(k=7)
   x = x + cond(g)                               → Conv1d(k=1) + Add
   for each upsample stage i:
     x = LeakyReLU(x, 0.1)
     x = ConvTranspose(x)                        → ConvTranspose
     x = x + noise_convs[i](har_source)          → Conv1d + Add
     for each resblock j:
       x = ResBlock1(x)                          → 3x (LeakyRelu, Conv, LeakyRelu, Conv, Add)
     x = x / num_kernels                         → Mul by scalar
   x = LeakyReLU(x, 0.1)
   x = conv_post(x)                              → Conv1d(k=7, no bias)
   x = tanh(x)                                   → Tanh
```

## Sub-module Graphs

### LayerNorm (channels C, eps=1e-5)
```
x = transpose(x, 1, -1)           → Transpose [0,2,1]
mean = ReduceMean(x, axis=-1)     → ReduceMean
x_centered = x - mean             → Sub
var = ReduceMean(x_centered²)     → Pow(2), ReduceMean
x_norm = x_centered / sqrt(var + eps)  → Add(eps), Sqrt, Div
x = x_norm * gamma + beta         → Mul, Add
x = transpose(x, 1, -1)           → Transpose [0,2,1]
```

### MultiHeadAttention (channels C, n_heads H, k_channels K, window_size W=10)
```
q = Conv1d(x, k=1)                → Conv
k = Conv1d(c, k=1)                → Conv
v = Conv1d(c, k=1)                → Conv

# Reshape to [b, H, T, K]
q = reshape(q, [b, H, K, T]) → transpose([b, H, T, K])
k = reshape(k, [b, H, K, T]) → transpose([b, H, T, K])
v = reshape(v, [b, H, K, T]) → transpose([b, H, T, K])

scores = (q / sqrt(K)) @ k^T     → Div, MatMul

# Relative position attention (window_size=10)
rel_k = get_relative_embeddings(emb_rel_k, T)  → Pad, Slice
rel_logits = (q/sqrt(K)) @ rel_k^T             → MatMul
scores_local = rel_pos_to_abs_pos(rel_logits)  → Pad, Reshape, Slice
scores = scores + scores_local                 → Add

# Mask
scores = masked_fill(scores, mask==0, -1e4)    → Equal, Cast, Where

p_attn = softmax(scores, dim=-1)              → Softmax
output = p_attn @ v                           → MatMul

# Relative values
rel_weights = abs_pos_to_rel_pos(p_attn)      → Pad, Reshape, Slice
rel_v = get_relative_embeddings(emb_rel_v, T) → Pad, Slice
output = output + rel_weights @ rel_v         → MatMul, Add

# Reshape back to [b, C, T]
output = transpose → reshape                   → Transpose, Reshape

x = Conv1d(output, k=1)                       → Conv (conv_o)
```

### FFN (in C, out C, filter F, kernel K)
```
x = Pad(x, [0,0, pad_l, pad_r])   → Pad
x = Conv1d(x, F, K)               → Conv
x = ReLU(x)                       → Relu
x = Pad(x, [0,0, pad_l, pad_r])   → Pad
x = Conv1d(x, C, K)               → Conv
```
Where pad_l = (K-1)//2, pad_r = K//2 for same padding.

### WaveNet WN (hidden C, kernel K, n_layers L, gin G)
```
if gin: g = Conv1d(g, 2*C*L, k=1)  → Conv (cond_layer)

output = zeros_like(x)
for i in range(L):
    x_in = Conv1d(x, 2*C, K, dilation=i)  → Conv (in_layers[i])
    if gin: g_l = slice(g, i*2C, (i+1)*2C) → Slice
    in_act = x_in + g_l                      → Add
    t_act = tanh(in_act[:, :C, :])           → Slice, Tanh
    s_act = sigmoid(in_act[:, C:, :])        → Slice, Sigmoid
    acts = t_act * s_act                     → Mul
    res_skip = Conv1d(acts, 2C or C, k=1)   → Conv (res_skip_layers[i])
    if i < L-1:
        res = res_skip[:, :C, :]             → Slice
        x = (x + res) * x_mask               → Add, Mul
        output = output + res_skip[:, C:, :] → Slice, Add
    else:
        output = output + res_skip           → Add
return output * x_mask                       → Mul
```

### ResBlock1 (channels C, kernel K, dilations [d1,d2,d3])
```
for each (c1, c2) pair:
    x = LeakyReLU(x, 0.1)       → LeakyRelu
    x = Conv1d(x, C, K, dil)    → Conv (convs1[i])
    x = LeakyReLU(x, 0.1)       → LeakyRelu
    x = Conv1d(x, C, K, dil=1)  → Conv (convs2[i])
    x = x + x_in                → Add
```

### SineGen (sr, harmonic_num=0)
```
f0 = f0[:, None].transpose(1, 2)    → Unsqueeze, Transpose
f0_buf = zeros(b, 1, dim)           → Constant
f0_buf[:, :, 0] = f0               → (harmonic_num=0, so just f0)
rad_values = (f0_buf / sr) % 1      → Div, Sub(Floor) or Mod
rand_ini = rand(b, dim)             → RandomUniform
rand_ini[:, 0] = 0                  → ScatterND
rad_values += rand_ini              → Add
tmp = cumsum(rad_values, dim=1)     → CumSum
tmp *= upp                          → Mul
tmp = interpolate(tmp, upp)         → Resize
tmp %= 1                            → Sub(Floor)
shift = (tmp[1:] - tmp[:-1]) < 0    → Slice, Sub, Less, Cast
cumsum_shift = cumsum(shift)        → CumSum
sine = sin(cumsum(rad + shift) * 2π) → CumSum, Mul, Sin
sine *= sine_amp                    → Mul
```

## Weight Norm Resolution

For every `*.weight_g` + `*.weight_v` pair:
```
weight = weight_g * weight_v / ||weight_v||
```
where `||weight_v||` is computed over all dims except dim 0.

For Conv1d weight [out, in, k]: norm per output channel over [in, k].
For ConvTranspose1d weight [in, out, k]: norm per input channel over [out, k].

**In the ONNX graph, weight norm is expressed as nodes:**
- ReduceL2(weight_v, axes=[1,2,keepdims=1]) → norm
- Div(weight_v, norm) → v_hat
- Mul(v_hat, weight_g) → weight

This is how PyTorch's ONNX exporter does it — the weight_norm is NOT pre-computed,
it's emitted as graph nodes (ReduceL2, Div, Mul). This is why the ONNX has
104 ReduceL2 nodes.

**CRITICAL**: We must emit weight norm as graph nodes, NOT pre-compute the
weights. The initializers in the ONNX are `weight_g` and `weight_v` (raw),
and the graph computes the actual weight.

## ONNX Protobuf Structure

### Wire Format
```
tag = (field_number << 3) | wire_type
wire_type: 0=varint, 1=64bit, 2=length-delimited, 5=32bit
```

### ModelProto (verified from actual ONNX files)
| Field | Number | Type | Value |
|-------|--------|------|-------|
| ir_version | 1 | varint | 7 |
| opset_import | 2 | repeated msg | domain="", version=13 |
| producer_name | 7 | string | "pytorch" |
| producer_version | 8 | string | "2.0.0" |
| graph | 11 | msg | GraphProto |
| model_version | 14 | varint | 0 |

### GraphProto
| Field | Number | Type |
|-------|--------|------|
| name | 1 | string |
| node | 2 | repeated NodeProto |
| initializer | 5 | repeated TensorProto |
| input | 11 | repeated ValueInfoProto |
| output | 12 | repeated ValueInfoProto |

### NodeProto
| Field | Number | Type |
|-------|--------|------|
| input | 1 | repeated string |
| output | 2 | repeated string |
| name | 3 | string |
| op_type | 4 | string |
| attribute | 5 | repeated AttributeProto |

### TensorProto
| Field | Number | Type |
|-------|--------|------|
| dims | 1 | repeated int64 (varint) |
| data_type | 2 | varint (1=FLOAT, 7=INT64, 9=BOOL) |
| name | 4 | string |
| raw_data | 13 | bytes |

### AttributeProto
| Field | Number | Type |
|-------|--------|------|
| name | 1 | string |
| f | 2 | float (32bit) |
| i | 3 | varint |
| s | 4 | bytes |
| t | 5 | TensorProto msg |
| ints | 7 | repeated varint |
| type | 20 | varint (1=FLOAT, 2=INT, 3=STRING, 4=TENSOR, 6=FLOATS, 7=INTS) |

### ValueInfoProto
| Field | Number | Type |
|-------|--------|------|
| name | 1 | string |
| type | 2 | TypeProto msg |

### TypeProto → TensorTypeProto → TensorShapeProto → Dimension
```
TypeProto: tensor_type(1) = TensorTypeProto
TensorTypeProto: elem_type(1) = varint, shape(2) = TensorShapeProto
TensorShapeProto: dim(1) = repeated Dimension
Dimension: dim_value(1) = varint, dim_param(2) = string
```

## Implementation Steps

### Step 1: `src/onnx_emit.h` + `src/onnx_emit.c`
Minimal protobuf writer:
```c
typedef struct { uint8_t *data; size_t len, cap; } Buf;
void buf_init(Buf *b);
void buf_free(Buf *b);
void buf_put(Buf *b, const void *p, size_t n);
void pb_varint(Buf *b, uint64_t v);
void pb_tag(Buf *b, uint32_t field, uint32_t wire);
void pb_uint64(Buf *b, uint32_t field, uint64_t v);
void pb_float(Buf *b, uint32_t field, float f);
void pb_bytes(Buf *b, uint32_t field, const uint8_t *p, size_t n);
void pb_string(Buf *b, uint32_t field, const char *s);
void pb_message(Buf *b, uint32_t field, const Buf *msg);
```

### Step 2: `src/onnx_emit.c` — ONNX message emitters
```c
void emit_tensor_f32(Buf *b, const char *name, int64_t *dims, int ndim, const float *data, size_t count);
void emit_tensor_i64(Buf *b, const char *name, int64_t *dims, int ndim, const int64_t *data, size_t count);
void emit_node(Buf *graph, const char *op, const char *name,
               const char **inputs, int n_in, const char **outputs, int n_out,
               const Attr *attrs, int n_attrs);
void emit_value_info(Buf *b, const char *name, int elem_type, int64_t *dims, int ndim);
void emit_model(Buf *out, const Buf *graph, int ir_version, int opset);
```

### Step 3: `src/rvc_graph.h` + `src/rvc_graph.c`
The RVC v2 graph topology. Functions:
```c
void emit_speaker_embedding(Buf *g, ...);
void emit_encoder_input(Buf *g, ...);
void emit_attention_layer(Buf *g, int layer_idx, ...);
void emit_ffn_layer(Buf *g, int layer_idx, ...);
void emit_encoder_proj(Buf *g, ...);
void emit_flow_layer(Buf *g, int flow_idx, ...);
void emit_sinegen(Buf *g, ...);
void emit_decoder(Buf *g, ...);
void emit_resblock1(Buf *g, int rb_idx, ...);
void emit_wavenet(Buf *g, ...);
void emit_layernorm(Buf *g, ...);
```

Each function emits the exact ONNX nodes for that sub-module, using the
weight names from the .pth as initializer references.

### Step 4: `src/pth2onnx.c`
Main entry point:
```c
int main(int argc, char **argv) {
    // 1. Read .pth → config + tensors (use existing pth_loader)
    // 2. Derive architecture params from config + tensor shapes
    // 3. Build ONNX graph (call rvc_graph functions)
    // 4. Serialize to .onnx file
}
```

### Step 5: Build + test
```
cmd //c build_pth2onnx.bat
./pth2onnx C:/temp/nahida_extracted/nahida-jp.pth C:/temp/nahida_converted.onnx
```
Then load with ONNX Runtime and compare output to `nahida_out.wav`.

## File Structure
```
src/onnx_emit.h      — protobuf writer API
src/onnx_emit.c      — protobuf writer impl
src/rvc_graph.h      — RVC graph emitter API
src/rvc_graph.c      — RVC graph emitter impl (the big one)
src/pth2onnx.c       — main entry point
build_pth2onnx.bat   — build script
```

## Key References
- .pth file: `C:/temp/nahida_extracted/nahida-jp.pth`
- Python model: `/c/temp/rvcwheel/rvcx/rvc_python/lib/infer_pack/models_onnx.py`
- Python attentions: `/c/temp/rvcwheel/rvcx/rvc_python/lib/infer_pack/attentions.py`
- Python modules: `/c/temp/rvcwheel/rvcx/rvc_python/lib/infer_pack/modules.py`
- Python commons: `/c/temp/rvcwheel/rvcx/rvc_python/lib/infer_pack/commons.py`
- Existing C: `src/pth_loader.c`, `src/pickle.c`, `src/zip_reader.c` (WORKING)
- Reference ONNX: `models/nahida.onnx` (4606 nodes, 457 inits)
- Reference output: `C:/temp/nahida_out.wav`

## CRITICAL NOTES
- **Weight norm is emitted as graph nodes** (ReduceL2, Div, Mul), NOT pre-computed
- **The .pth has `weight_g` and `weight_v`** — these are the initializers
- **The graph computes the actual weight** from g and v
- **All weights in .pth are float16** — must convert to float32 for ONNX
- **Dynamic axes**: phone[1,T,dim], pitch[1,T], pitchf[1,T], rnd[1,192,T]
- **The graph is deterministic** — same config → same topology, different weights
- **256-dim vs 768-dim**: only `emb_phone` changes (256 vs 768 input dim)
- **Kernel sizes vary**: [24,20,4,4] vs [16,16,4,4] — from config
