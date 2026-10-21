# RVC `.pth` → GGUF Conversion Spec

Version: `rvc.gguf.v1`

This is the convention I use for converting RVC / VITS-style `.pth` checkpoints into GGUF for a native C runtime.

Important high-level point:

> GGUF is being used as a **weight container + metadata container**, not as an ONNX-like graph format.  
> The RVC computation graph is implemented in C code, similar in spirit to Kokoro-style `src/core/*.h` static graphs.

---

## 1. GGUF tensor naming convention

### 1.1 Source state dict

RVC `.pth` files commonly contain something like:

```python
ckpt = torch.load(path, map_location="cpu")

state = None
if "weight" in ckpt:
    state = ckpt["weight"]
elif "model" in ckpt:
    state = ckpt["model"]
else:
    state = ckpt

config = ckpt.get("config", None)
```

Only the model state dict is converted. Optimizer state is ignored.

---

### 1.2 Name normalization

For every tensor name in the PyTorch state dict:

1. Strip common wrapper prefixes:

   ```text
   module.
   model.
   net_g.
   synthesizer.
   generator.
   ```

2. Keep the remaining dotted PyTorch path.

3. Do not rename module indices.

4. Keep `flows.0`, `flows.1`, etc. exactly as-is.

5. Keep `resblocks.0.convs1.0`, etc. exactly as-is.

Example:

```text
module.dec.ups.0.weight_g
```

becomes:

```text
dec.ups.0.weight_g
```

before weight-norm handling.

---

### 1.3 Weight-norm naming

PyTorch old-style `weight_norm` stores:

```text
layer.weight_g
layer.weight_v
```

For RVC GGUF, the default runtime convention is:

> Fuse all weight-normed weights during conversion and store a single normal weight tensor named `layer.weight`.

So:

```text
dec.ups.0.weight_g
dec.ups.0.weight_v
```

become one GGUF tensor:

```text
dec.ups.0.weight
```

Similarly:

```text
flow.flows.0.enc.in_layers.0.weight_g
flow.flows.0.enc.in_layers.0.weight_v
```

become:

```text
flow.flows.0.enc.in_layers.0.weight
```

Metadata should record:

```text
rvc.weight_norm = "fused"
```

If you intentionally want to preserve weight-norm parameters instead of fusing, use:

```text
layer.weight.g
layer.weight.v
```

and set:

```text
rvc.weight_norm = "split"
```

But for RVC inference I strongly recommend fused weights. It simplifies the C runtime and avoids runtime normalization error after quantization.

---

### 1.4 Bias, norm, embedding names

Biases and norms keep their PyTorch names:

```text
layer.bias
layer.weight
norm.weight
norm.bias
emb.weight
```

Examples:

```text
dec.conv_pre.bias
dec.resblocks.0.convs1.0.bias
flow.flows.0.proj.bias
enc_p.encoder.norm_layers_1.0.weight
enc_p.encoder.norm_layers_1.0.bias
emb_g.weight
```

---

### 1.5 Unknown tensors

Any tensor that is not recognized should still be written using its normalized dotted name.

The C runtime may ignore it, but preserving it makes debugging and future op support easier.

Metadata:

```text
rvc.unknown_tensors_preserved = 1
```

---

## 2. GGUF shape convention

GGUF / ggml tensor dimensions are stored as `ne[0], ne[1], ne[2], ...`, where:

```text
ne[0] = fastest-varying dimension in memory
```

PyTorch tensors are row-major / C-contiguous. The conversion rule I use is:

> Copy the raw contiguous PyTorch bytes unchanged, and reverse the logical dimension list when writing GGUF `ne`.

So for a contiguous PyTorch tensor with shape:

```text
[d0, d1, d2, ..., dn]
```

the GGUF logical dimensions are:

```text
[dn, ..., d2, d1, d0]
```

This means no byte-level transpose is performed.

---

### 2.1 Common shape mappings

| PyTorch parameter | PyTorch shape | GGUF `ne` | Notes |
|---|---:|---:|---|
| Bias / LayerNorm gamma/beta | `[C]` | `[C]` | 1D |
| Embedding | `[vocab, dim]` | `[dim, vocab]` | Row per token |
| Linear weight | `[out, in]` | `[in, out]` | Suitable for `ggml_mul_mat` |
| Conv1d weight | `[out, in_per_group, K]` | `[K, in_per_group, out]` | Copy flat |
| ConvTranspose1d weight | `[in, out_per_group, K]` | `[K, out_per_group, in]` | Copy flat |
| Conv2d weight | `[out, in_per_group, KH, KW]` | `[KW, KH, in_per_group, out]` | Rare in RVC |
| ConvTranspose2d weight | `[in, out_per_group, KH, KW]` | `[KW, KH, out_per_group, in]` | Rare in RVC |

---

### 2.2 Conv1d example

PyTorch Conv1d weight:

```text
[out_channels, in_channels_per_group, kernel_size]
```

Memory order is:

```text
out_channels slowest
kernel_size fastest
```

GGUF name:

```text
some.conv.weight
```

GGUF shape:

```text
[kernel_size, in_channels_per_group, out_channels]
```

Bias:

```text
some.conv.bias
```

GGUF shape:

```text
[out_channels]
```

---

### 2.3 ConvTranspose1d example

PyTorch ConvTranspose1d weight:

```text
[in_channels, out_channels_per_group, kernel_size]
```

GGUF shape:

```text
[kernel_size, out_channels_per_group, in_channels]
```

Bias:

```text
[out_channels]
```

This is important for RVC decoder upsampling layers:

```text
dec.ups.0.weight
dec.ups.1.weight
...
```

---

### 2.4 Linear vs 1x1 Conv

Some VITS/RVC modules use `nn.Conv1d(..., kernel_size=1)` where conceptually a Linear layer would also work.

If PyTorch shape is:

```text
[out, in]
```

store as:

```text
[in, out]
```

If PyTorch shape is:

```text
[out, in, 1]
```

store as:

```text
[1, in, out]
```

The C graph decides whether to implement it as matrix multiplication over time or as a 1x1 conv.

---

## 3. Graph definition: C code, not a GGUF graph

GGUF does not define RVC ops.

The graph is implemented in C.

The recommended layout is:

```text
model.gguf
```

contains:

```text
weights
config metadata
```

The C code contains:

```text
RVC graph construction
op kernels
tensor lookup by name
```

So yes, it is closer to Kokoro-style static C graph code than to a separate graph file.

---

### 3.1 What GGUF stores

GGUF stores:

```text
tensors
metadata
```

For RVC, metadata should include enough information for the C graph to reconstruct module shapes and attributes.

Recommended metadata keys:

```text
general.architecture = "rvc"
general.name = "rvc-model"

rvc.version = "v1"
rvc.weight_norm = "fused"
rvc.causal = 0 or 1

rvc.sample_rate = 48000
rvc.upp_total = 480

rvc.config.json = "{...original RVC config...}"
```

`rvc.config.json` should contain the original RVC config verbatim if available.

Example, conceptually:

```json
{
  "sampling_rate": 48000,
  "inter_channels": 192,
  "hidden_channels": 192,
  "filter_channels": 768,
  "n_heads": 2,
  "n_layers": 6,
  "kernel_size": 3,
  "p_dropout": 0.1,
  "resblock": "1",
  "resblock_kernel_sizes": [3, 7, 11],
  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
  "upsample_rates": [12, 10, 2, 2],
  "upsample_initial_channel": 512,
  "upsample_kernel_sizes": [24, 20, 4, 4],
  "gin_channels": 256,
  "n_speakers": 1
}
```

The exact fields depend on the RVC version. Store whatever the `.pth` contains.

---

### 3.2 What GGUF does not store

GGUF does not store:

```text
execution order
op types
padding modes
dilation rates
flow direction
causal behavior
```

Unless you intentionally add custom metadata.

The C runtime knows the RVC graph and looks up tensors by name.

Example:

```c
struct ggml_tensor *w = ggml_get_tensor(ctx, "dec.ups.0.weight");
struct ggml_tensor *b = ggml_get_tensor(ctx, "dec.ups.0.bias");
```

The C code knows that `dec.ups.0` is a `ConvTranspose1d`.

---

### 3.3 Optional embedded graph metadata

If you want a data-driven converter or debugging aid, you can embed a JSON graph description:

```text
rvc.graph.json = "{...}"
```

But the runtime does not require it.

For production RVC inference, I recommend:

```text
GGUF = weights + config metadata
C code = graph
```

---

## 4. RVC-specific tensor map

The exact module names can vary slightly between RVC forks, but the following covers the common VITS/RVC `SynthesizerTrn` layout.

---

### 4.1 Speaker embedding

PyTorch:

```text
emb_g.weight
```

Shape:

```text
[n_speakers, gin_channels]
```

GGUF:

```text
emb_g.weight
```

GGUF shape:

```text
[gin_channels, n_speakers]
```

If `gin_channels == 0` or the model has no speaker embedding, this tensor may be absent.

---

### 4.2 Text / condition encoder `enc_p`

Common tensors:

```text
enc_p.emb.weight
enc_p.f0_proj.weight
enc_p.f0_proj.bias
enc_p.encoder.attn_layers.{i}.q_proj.weight
enc_p.encoder.attn_layers.{i}.k_proj.weight
enc_p.encoder.attn_layers.{i}.v_proj.weight
enc_p.encoder.attn_layers.{i}.out_proj.weight
enc_p.encoder.attn_layers.{i}.q_proj.bias
enc_p.encoder.attn_layers.{i}.k_proj.bias
enc_p.encoder.attn_layers.{i}.v_proj.bias
enc_p.encoder.attn_layers.{i}.out_proj.bias
enc_p.encoder.norm_layers_1.{i}.weight
enc_p.encoder.norm_layers_1.{i}.bias
enc_p.encoder.ffn_layers.{i}.conv1.weight
enc_p.encoder.ffn_layers.{i}.conv1.bias
enc_p.encoder.ffn_layers.{i}.conv2.weight
enc_p.encoder.ffn_layers.{i}.conv2.bias
enc_p.encoder.norm_layers_2.{i}.weight
enc_p.encoder.norm_layers_2.{i}.bias
enc_p.proj.weight
enc_p.proj.bias
```

`enc_p.proj` is commonly a 1x1 Conv1d projecting to mean/logvar:

```text
[2 * inter_channels, hidden_channels, 1]
```

GGUF shape:

```text
[1, hidden_channels, 2 * inter_channels]
```

Bias:

```text
[2 * inter_channels]
```

If attention projections are Linear:

```text
[out, in]
```

GGUF:

```text
[in, out]
```

If they are 1x1 Conv1d:

```text
[out, in, 1]
```

GGUF:

```text
[1, in, out]
```

The C graph should handle both by inspecting tensor rank.

---

### 4.3 Posterior encoder `enc_q`

Usually not needed for inference/conversion, but preserve it if present.

Common tensors:

```text
enc_q.pre.weight
enc_q.pre.bias
enc_q.enc.in_layers.{j}.weight
enc_q.enc.in_layers.{j}.bias
enc_q.enc.res_skip_layers.{j}.weight
enc_q.enc.res_skip_layers.{j}.bias
enc_q.enc.cond_layer.weight
enc_q.enc.cond_layer.bias
enc_q.proj.weight
enc_q.proj.bias
```

After weight-norm fusion:

```text
enc_q.enc.in_layers.{j}.weight
enc_q.enc.res_skip_layers.{j}.weight
enc_q.enc.cond_layer.weight
```

`enc_q.proj` often has shape:

```text
[2 * inter_channels, hidden_channels, 1]
```

GGUF:

```text
[1, hidden_channels, 2 * inter_channels]
```

---

### 4.4 Flow stack

Flow tensors follow:

```text
flow.flows.{i}.enc.in_layers.{j}.weight
flow.flows.{i}.enc.in_layers.{j}.bias

flow.flows.{i}.enc.res_skip_layers.{j}.weight
flow.flows.{i}.enc.res_skip_layers.{j}.bias

flow.flows.{i}.enc.cond_layer.weight
flow.flows.{i}.enc.cond_layer.bias

flow.flows.{i}.proj.weight
flow.flows.{i}.proj.bias
```

Important: in VITS-style WaveNet blocks, the conditioning layer is usually a single conv named:

```text
cond_layer
```

not a list of `cond_layers`.

Its output channel count is usually:

```text
2 * hidden_channels * n_wavenet_layers
```

So for a flow WN with hidden size `H` and `L` layers:

```text
flow.flows.{i}.enc.cond_layer.weight
```

has PyTorch shape:

```text
[2 * H * L, gin_channels, 1]
```

GGUF shape:

```text
[1, gin_channels, 2 * H * L]
```

The C code slices the conditioned speaker tensor per WN layer:

```text
slice l = [l * 2 * H, (l + 1) * 2 * H)
```

---

### 4.5 Decoder `dec`

Common HiFi-GAN/VITS decoder tensors:

```text
dec.conv_pre.weight
dec.conv_pre.bias

dec.ups.{i}.weight
dec.ups.{i}.bias

dec.resblocks.{i}.convs1.{j}.weight
dec.resblocks.{i}.convs1.{j}.bias

dec.resblocks.{i}.convs2.{j}.weight
dec.resblocks.{i}.convs2.{j}.bias

dec.conv_post.weight
dec.conv_post.bias

dec.cond.weight
dec.cond.bias
```

Speaker conditioning conv:

```text
dec.cond.weight
```

is usually a 1x1 Conv1d:

```text
[upsample_initial_channel, gin_channels, 1]
```

GGUF:

```text
[1, gin_channels, upsample_initial_channel]
```

Decoder upsampling layers are ConvTranspose1d.

PyTorch:

```text
[in_channels, out_channels, kernel_size]
```

GGUF:

```text
[kernel_size, out_channels, in_channels]
```

Bias:

```text
[out_channels]
```

Residual block convs are ordinary Conv1d.

PyTorch:

```text
[out_channels, in_channels, kernel_size]
```

GGUF:

```text
[kernel_size, in_channels, out_channels]
```

Bias:

```text
[out_channels]
```

---

### 4.6 Source module

If the checkpoint includes the NSF source module:

```text
source_module.l_linear.weight
source_module.l_linear.bias
```

For `harmonic_num = 0`, this is a `Linear(1, 1)`.

PyTorch shape:

```text
[1, 1]
```

GGUF shape:

```text
[1, 1]
```

Bias:

```text
[1]
```

The sinusoidal generator itself has no learned weights.

---

## 5. Weight-norm mapping

### 5.1 Default behavior: fuse

For every pair:

```text
foo.weight_g
foo.weight_v
```

produce:

```text
foo.weight
```

Metadata:

```text
rvc.weight_norm = "fused"
```

Do not store:

```text
foo.weight_g
foo.weight_v
```

unless using split mode.

---

### 5.2 Fusion formula

Assume old-style PyTorch `weight_norm` with default `dim=0`.

Given:

```text
V = weight_v
G = weight_g
```

For a tensor `V` with shape:

```text
[D0, D1, D2, ...]
```

`G` is normally stored as:

```text
[D0, 1, 1, ...]
```

Squeeze `G` to:

```text
[D0]
```

Then for each index `i` along axis 0:

```text
norm_i = sqrt(sum over all non-axis-0 dimensions of V[i, ...]^2)
scale_i = G[i] / norm_i
W[i, ...] = V[i, ...] * scale_i
```

For Conv1d:

```text
V shape: [out_channels, in_channels_per_group, kernel_size]
G shape: [out_channels, 1, 1]
```

Fused:

```text
W[o, c, k] = V[o, c, k] * G[o] / sqrt(sum_c,sum_k V[o, c, k]^2)
```

For ConvTranspose1d:

```text
V shape: [in_channels, out_channels_per_group, kernel_size]
G shape: [in_channels, 1, 1]
```

Normalization is still along axis 0, i.e. per input channel:

```text
W[i, o, k] = V[i, o, k] * G[i] / sqrt(sum_o,sum_k V[i, o, k]^2)
```

---

### 5.3 Epsilon / zero norm

PyTorch’s old weight-norm does not generally behave like `F.normalize` with a user epsilon. For exactness, use:

```text
scale = g / norm
```

without adding epsilon unless `norm == 0`.

For safety in C:

```c
if (norm == 0.0f) {
    scale = 0.0f;
} else {
    scale = g / norm;
}
```

In practice, trained RVC weights should not have zero norm.

---

### 5.4 Split mode, optional

If preserving weight-norm parameters:

```text
foo.weight_g -> foo.weight.g
foo.weight_v -> foo.weight.v
```

Store `foo.weight.g` as 1D:

```text
[D0]
```

Store `foo.weight.v` using the normal fused-weight shape convention.

Metadata:

```text
rvc.weight_norm = "split"
```

C runtime then computes fused weight once at load time.

This is useful for debugging, but not recommended for quantized deployment.

---

## 6. Causal conv padding mapping

Padding is not stored in PyTorch weights.

It comes from module attributes, model config, or the forward code.

Therefore:

> GGUF tensor names do not encode padding.  
> The C graph must know whether a conv is causal, same-padded, or explicitly padded.

---

### 6.1 Ordinary Conv1d

For a Conv1d with:

```text
kernel_size = K
dilation = D
stride = S
```

effective kernel size:

```text
K_eff = (K - 1) * D + 1
```

For causal convolution:

```text
pad_left = K_eff - 1 = (K - 1) * D
pad_right = 0
```

For non-causal same convolution with odd effective kernel:

```text
pad_left = pad_right = (K_eff - 1) / 2
```

This matches the usual VITS WaveNet padding:

```python
padding = int((kernel_size * dilation - dilation) / 2)
```

because:

```text
kernel_size * dilation - dilation
= dilation * (kernel_size - 1)
= K_eff - 1
```

---

### 6.2 RVC/VITS WaveNet `in_layers`

For:

```text
flow.flows.{i}.enc.in_layers.{j}.weight
enc_q.enc.in_layers.{j}.weight
```

the C graph must know:

```text
kernel_size
dilation
causal
```

If non-causal VITS default:

```text
pad_left = pad_right = ((K - 1) * D) / 2
```

If causal RVC variant:

```text
pad_left = (K - 1) * D
pad_right = 0
```

The weight shape is identical in both cases.

---

### 6.3 1x1 convolutions

For:

```text
flow.flows.{i}.proj.weight
flow.flows.{i}.enc.res_skip_layers.{j}.weight
flow.flows.{i}.enc.cond_layer.weight
enc_p.proj.weight
dec.cond.weight
```

kernel size is 1, dilation is 1, padding is 0.

No causal concern.

---

### 6.4 ConvTranspose1d / decoder upsampling

For decoder upsampling layers:

```text
dec.ups.{i}.weight
```

do not infer padding purely from weight shape.

You need:

```text
stride
kernel_size
padding
output_padding
causal trim
```

In non-causal HiFi-GAN-style RVC decoders, padding is usually chosen so that:

```text
output_length = input_length * stride
```

But exact values should be extracted from the Python module:

```python
m = model.dec.ups[i]
print(m.stride)
print(m.kernel_size)
print(m.padding)
print(m.output_padding)
```

If using a causal decoder, you also need explicit trim:

```text
trim_left
trim_right
```

I recommend storing explicit attributes in metadata if they are not derivable from the standard config.

Example optional metadata:

```text
rvc.op.dec.ups.0.type = "conv_transpose1d"
rvc.op.dec.ups.0.attrs = "{\"stride\":12,\"kernel\":24,\"padding\":6,\"output_padding\":0,\"trim_left\":0,\"trim_right\":0}"
```

If you do not want per-op metadata, generate a C header with these values during conversion.

---

## 7. Flow coupling layer mapping

RVC/VITS flows are affine coupling flows.

For each flow index `i`:

```text
flow.flows.{i}
```

weights:

```text
flow.flows.{i}.enc.*
flow.flows.{i}.proj.weight
flow.flows.{i}.proj.bias
```

---

### 7.1 Channel split

Input tensor:

```text
x: [C, T]
```

Split along channel dimension:

```text
C0 = C / 2

x0 = x[0:C0, :]
x1 = x[C0:C, :]
```

`C` must be even.

---

### 7.2 Coupling network

The condition path uses `x0`:

```text
h = WN(flow.flows.{i}.enc, x0, mask, g)
```

Then:

```text
stats = conv1x1(flow.flows.{i}.proj, h)
stats = stats * mask
```

Split stats:

```text
m    = stats[0:C0, :]
logs = stats[C0:C, :]
```

---

### 7.3 Inference / reverse direction

For RVC synthesis/conversion, flows are normally run in reverse order:

```text
for i = n_flows - 1 down to 0:
    flow_step_reverse(i)
```

Reverse coupling formula:

```text
x1_out = (x1 * exp(logs) + m) * mask
x_out = concat(x0, x1_out)
```

Full step:

```text
x0, x1 = split(x)

h = WN(flow.flows.{i}.enc, x0, mask, g)

stats = conv1x1(flow.flows.{i}.proj, h) * mask

m, logs = split(stats)

x1 = (x1 * exp(logs) + m) * mask

x = concat(x0, x1)
```

---

### 7.4 Forward / training direction

For completeness, the non-reverse/training direction is:

```text
x1_out = (x1 - m) * exp(-logs) * mask
x_out = concat(x0, x1_out)
```

You usually do not need this for RVC inference.

---

### 7.5 Mask handling

If the C runtime processes fixed-length segments and does not track masks, use an all-ones mask.

If variable-length frames are supported:

```text
mask shape: [1, T]
```

Apply mask after stats and after output:

```text
stats *= mask
x1_out *= mask
```

---

## 8. WaveNet inside flows and posterior encoder

The `enc` module inside flows is a VITS-style WaveNet.

Tensor prefixes:

```text
flow.flows.{i}.enc.in_layers.{j}
flow.flows.{i}.enc.res_skip_layers.{j}
flow.flows.{i}.enc.cond_layer
```

Let:

```text
H = hidden_channels
L = number of WN layers
G = gin_channels
```

---

### 8.1 `in_layers`

Each:

```text
flow.flows.{i}.enc.in_layers.{j}.weight
```

is a dilated Conv1d:

```text
[H -> 2H]
```

PyTorch shape:

```text
[2H, H, K]
```

GGUF shape:

```text
[K, H, 2H]
```

Bias:

```text
[2H]
```

The output is split into two halves:

```text
a = out[0:H]
b = out[H:2H]

act = tanh(a) * sigmoid(b)
```

---

### 8.2 `cond_layer`

Usually one conv:

```text
flow.flows.{i}.enc.cond_layer.weight
```

Shape:

```text
[2 * H * L, G, 1]
```

GGUF shape:

```text
[1, G, 2 * H * L]
```

Bias:

```text
[2 * H * L]
```

At runtime:

```text
g_full = conv1x1(cond_layer, g)
```

Then for WN layer `j`:

```text
g_j = g_full[j * 2H : (j + 1) * 2H, :]
```

Add to `in_layers[j]` output before gating.

---

### 8.3 `res_skip_layers`

For all layers except the last:

```text
flow.flows.{i}.enc.res_skip_layers.{j}.weight
```

shape:

```text
[2H, H, 1]
```

GGUF:

```text
[1, H, 2H]
```

For the last layer:

```text
[H, H, 1]
```

GGUF:

```text
[1, H, H]
```

The tensor shape itself tells the C runtime which case it is.

Conceptual WN loop:

```text
output = zeros[H, T]
x = input

g_full = cond_layer(g) if gin_channels > 0

for j in 0..L-1:
    x_in = dilated_conv(in_layers[j], x)

    if gin_channels > 0:
        x_in += slice(g_full, j*2H, 2H)

    a = x_in[0:H]
    b = x_in[H:2H]

    act = tanh(a) * sigmoid(b)

    rs = conv1x1(res_skip_layers[j], act)

    if j < L-1:
        res  = rs[0:H]
        skip = rs[H:2H]
        output += skip
    else:
        res = rs
        output += res

    x = (x + res) * mask

return output * mask
```

---

## 9. Recommended metadata keys

Minimum:

```text
general.architecture = "rvc"
rvc.version = "v1"
rvc.weight_norm = "fused"
rvc.config.json = "{...}"
```

Recommended:

```text
rvc.sample_rate = 48000
rvc.upp_total = 480
rvc.causal = 0
rvc.use_speaker_embedding = 1
rvc.gin_channels = 256
rvc.hidden_channels = 192
rvc.inter_channels = 192
rvc.n_flow_layers = 4
rvc.n_speakers = 1
```

If `upp_total` is not stored, compute it in C from:

```text
product(upsample_rates)
```

But storing it explicitly is useful.

---

## 10. Converter pseudocode

```python
STRIP_PREFIXES = [
    "module.",
    "model.",
    "net_g.",
    "synthesizer.",
    "generator.",
]

def normalize_name(name):
    for prefix in STRIP_PREFIXES:
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name

def fuse_weight_norm(v, g, dim=0):
    # v: PyTorch tensor
    # g: PyTorch tensor, usually shape [D0, 1, 1, ...]

    norm = torch.linalg.vector_norm(
        v,
        ord=2,
        dim=dim,
        keepdim=True,
    )

    # Avoid division by zero in pathological cases.
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)

    return v * (g / norm)

def gguf_shape_from_torch(torch_shape):
    # Mechanical rule: reverse dimensions.
    return tuple(reversed(torch_shape))

def convert_state_dict(state):
    tensors = {}
    weight_g = {}
    weight_v = {}

    for raw_name, tensor in state.items():
        name = normalize_name(raw_name)

        if name.endswith(".weight_g"):
            base = name[: -len(".weight_g")]
            weight_g[base] = tensor.contiguous().float()
            continue

        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            weight_v[base] = tensor.contiguous().float()
            continue

        tensors[name] = tensor.contiguous().float()

    # Fuse weight norm.
    for base in list(weight_v.keys()):
        if base in weight_g:
            fused = fuse_weight_norm(
                weight_v[base],
                weight_g[base],
                dim=0,
            )
            tensors[base + ".weight"] = fused.contiguous().float()
        else:
            # If only weight_v exists, preserve it.
            tensors[base + ".weight.v"] = weight_v[base]

    for base in list(weight_g.keys()):
        if base not in weight_v:
            tensors[base + ".weight.g"] = weight_g[base]

    return tensors
```

When writing GGUF:

```python
for name, tensor in tensors.items():
    gguf_name = name
    gguf_ne = gguf_shape_from_torch(tensor.shape)
    write_tensor(gguf_name, tensor.numpy(), gguf_ne)
```

---

## 11. Validation checklist

After conversion, verify:

### 11.1 Name checks

You should see names like:

```text
dec.ups.0.weight
dec.resblocks.0.convs1.0.weight
flow.flows.0.enc.in_layers.0.weight
flow.flows.0.enc.cond_layer.weight
flow.flows.0.proj.weight
enc_p.proj.weight
```

You should not see:

```text
.weight_g
.weight_v
```

unless using split mode.

---

### 11.2 Shape checks

For a Conv1d:

```text
gguf_ne[0] == kernel_size
gguf_ne[1] == in_channels_per_group
gguf_ne[2] == out_channels
bias_ne[0] == out_channels
```

For a ConvTranspose1d:

```text
gguf_ne[0] == kernel_size
gguf_ne[1] == out_channels_per_group
gguf_ne[2] == in_channels
bias_ne[0] == out_channels
```

For Linear:

```text
gguf_ne[0] == in_features
gguf_ne[1] == out_features
bias_ne[0] == out_features
```

---

### 11.3 Weight-norm numerical check

For a fused weight `W`, compare against Python:

```python
import torch

v = state["flow.flows.0.enc.in_layers.0.weight_v"]
g = state["flow.flows.0.enc.in_layers.0.weight_g"]

norm = torch.linalg.vector_norm(v, ord=2, dim=0, keepdim=True)
w_ref = v * (g / norm)
```

Then compare against the GGUF tensor:

```text
flow.flows.0.enc.in_layers.0.weight
```

after reshaping according to the reverse-dimension rule.

---

### 11.4 Runtime check

For deterministic comparison:

1. Disable noise in the source module.
2. Use a fixed f0 array.
3. Use a fixed speaker ID.
4. Compare:
   - `enc_p` output,
   - flow output,
   - decoder input,
   - final waveform.

Do not compare final noisy output unless the RNG is identical.

---

## 12. Summary

### Tensor naming

Use normalized PyTorch dotted names.

Fuse:

```text
foo.weight_g + foo.weight_v -> foo.weight
```

Optional split mode:

```text
foo.weight.g
foo.weight.v
```

---

### Shape convention

Copy contiguous PyTorch bytes unchanged.

GGUF dimensions are reversed relative to PyTorch dimensions.

Examples:

```text
Linear [out, in]              -> [in, out]
Conv1d [out, in, K]           -> [K, in, out]
ConvTranspose1d [in, out, K]  -> [K, out, in]
Embedding [vocab, dim]        -> [dim, vocab]
```

---

### Graph definition

No separate graph file is required.

Use:

```text
GGUF = weights + metadata
C code = static RVC graph
```

This is the Kokoro-like approach.

---

### RVC op mappings

Important mappings already handled:

```text
weight_norm:
    fuse to normal weight by default

causal conv:
    pad_left = (kernel_size - 1) * dilation
    pad_right = 0

non-causal same conv:
    pad_left = pad_right = ((kernel_size - 1) * dilation) / 2

flow coupling:
    split channels in half
    WN on first half
    proj to m/logs
    reverse inference:
        x1 = x1 * exp(logs) + m
        x = concat(x0, x1)

WN cond_layer:
    one conv projecting g to 2*H*L channels
    slice 2*H channels per WN layer
```

This convention gives you a direct mechanical mapping from RVC `.pth` tensor names to GGUF tensor names while keeping the C runtime simple and static.