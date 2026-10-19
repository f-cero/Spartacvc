# Precise C Porting Guide: `SineGen` + `SourceModuleHnNSF` for RVC  
For: pure C, `harmonic_num = 0`, mono output, no PyTorch/numpy.

This guide assumes:

```c
harmonic_num = 0
dim = harmonic_num + 1 = 1
sine_amp = 0.1
noise_std = 0.003
voiced_threshold = 0
sampling_rate = from config, not hardcoded
upp = total upsample factor, integer, from config
f0 length = T
output length = N = T * upp
```

The most important fact for your case:

> With `harmonic_num = 0`, the random initial phase does **not** matter.  
> The actual clean sine phase is a zero-order-hold accumulation of `f0[frame] / sampling_rate`.  
> The complicated `tmp_over_one / cumsum_shift` logic is an integer-cycle wrap correction. Mathematically it does not change the sine; it exists mainly for float32 numerical behavior and to keep the phase accumulator bounded.

---

## 1. What the algorithm actually computes

For each model frame `i`, `f0[i]` is a fundamental frequency in Hz.

For `harmonic_num = 0`, only the fundamental is generated.

Define:

```text
sr = sampling_rate
U  = upp
T  = number of f0 frames
N  = T * U
```

For each frame `i`:

```text
r_i = fractional_part(f0[i] / sr)
```

This is the phase increment per audio sample, in cycles/sample.

Usually `f0[i] / sr < 1`, so the fractional part does nothing, but it matches:

```python
(f0_buf / self.sampling_rate) % 1
```

The actual per-audio-sample increment stream used by PyTorch is nearest-neighbor upsampled:

```text
R_j = r_{floor(j / U)}        for j = 0 .. N-1
```

That means each f0 frame is held constant for exactly `U` audio samples.

The clean sine before noise is:

```text
A_j = sum_{m=0}^{j} R_m
clean_j = sine_amp * sin(2*pi*A_j)
```

In exact math, this is the core.

The PyTorch code then adds an integer cycle correction:

```text
cumsum_shift_j ∈ {0, -1}
A_j = sum_{m=0}^{j} (R_m + cumsum_shift_m)
```

Since the cumulative correction is always an integer number of cycles:

```text
sin(2*pi*(A_j + integer)) = sin(2*pi*A_j)
```

Therefore, mathematically, the wrap correction does not change the ideal waveform. It matters for reproducing PyTorch float32 rounding and for preventing a large phase argument to `sin`.

---

## 2. Python-to-C meaning for `harmonic_num = 0`

Python concept | C equivalent
---|---
`rad_values` | `r[i] = frac(f0[i] / sr)`
`F.interpolate(..., mode="nearest")` on `rad_values` | `R[j] = r[j / U]`
`torch.cumsum(rad_values, dim=1)` | cumulative sum over frames
`tmp_over_one` | linearly interpolated, upsampled cumulative phase used only for wrap detection
`tmp_over_one %= 1` | fractional part
`cumsum_shift` | `-1` at detected wrap positions, `0` otherwise
`uv` | nearest-upsampled voiced/unvoiced flag
`noise` | Gaussian noise scaled by voiced/unvoiced amplitude
`l_linear` | scalar affine transform: `w*x + b`
`l_tanh` | `tanhf(...)`

---

## 3. Exact formulas

### 3.1 Frame-domain quantities

For `i = 0 .. T-1`:

```text
x_i = f0[i] / sr
r_i = frac(x_i) = x_i - floor(x_i)
```

Voiced/unvoiced frame flag:

```text
uv_frame_i = 1.0 if f0[i] > voiced_threshold else 0.0
```

Because `voiced_threshold = 0`, this means:

```text
f0[i] > 0   => voiced
f0[i] == 0  => unvoiced
f0[i] < 0   => unvoiced
```

Use strict `>`, not `>=`.

Cumulative frame phase:

```text
C_i = sum_{k=0}^{i} r_k
```

Scaled cumulative phase used for wrap detection:

```text
V_i = U * C_i
```

This corresponds to:

```python
tmp_over_one = torch.cumsum(rad_values, dim=1) * upp
```

---

### 3.2 Nearest interpolation exactly

For integer upsampling factor `U`, PyTorch nearest interpolation of a 1D sequence of length `T` to length `N = T*U` is block replication:

```text
input_index(j) = floor(j / U)
```

So:

```text
R_j = r_{floor(j / U)}
UV_j = uv_frame_{floor(j / U)}
```

In C:

```c
size_t frame = j / U;
float inc = r[frame];
float uv = uv_frame[frame];
```

There is no linear blending. There is no half-pixel offset for this integer nearest upsample case.

---

### 3.3 `F.interpolate(mode="linear", align_corners=True)` exactly

This is used only on `V_i`, not on `f0` directly.

Input length: `T`  
Output length: `N = T * U`

If `T == 1`, output is constant:

```text
Y_j = V_0
```

If `T > 1`:

For output index `j`, compute:

```text
pos_j = j * (T - 1) / (N - 1)
```

Then:

```text
k = floor(pos_j)
a = pos_j - k
```

If `k >= T - 1`:

```text
Y_j = V_{T-1}
```

Otherwise:

```text
Y_j = (1 - a) * V_k + a * V_{k+1}
```

This is the exact `align_corners=True` formula.

Important:

```text
output 0       maps exactly to input 0
output N - 1   maps exactly to input T - 1
```

Do not use:

```text
j / U
```

for linear `align_corners=True`. That is a different mapping.

---

### 3.4 Fractional part / modulo

PyTorch:

```python
tmp_over_one %= 1
```

For positive values this is just the fractional part.

Use:

```text
frac(x) = x - floor(x)
```

So:

```text
P_j = frac(Y_j)
```

In C:

```c
double p = y - floor(y);
```

This is safer than `fmod` for negative values too, because it matches Python-style modulo behavior by returning a value in `[0, 1)`.

---

### 3.5 Exact wrap-around correction formula

PyTorch does:

```python
tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
cumsum_shift = torch.zeros_like(rad_values)
cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
```

For `dim = 1`, flattened to 1D:

For `j = 0`:

```text
cumsum_shift_0 = 0
```

For `j >= 1`:

```text
D_j = P_j - P_{j-1}

if D_j < 0:
    cumsum_shift_j = -1
else:
    cumsum_shift_j = 0
```

Equivalently:

```text
W_0 = 0
W_j = 1 if j > 0 and P_j < P_{j-1}, else 0

cumsum_shift_j = -W_j
```

Then final phase:

```text
A_j = sum_{m=0}^{j} (R_m + cumsum_shift_m)
```

Clean sine:

```text
clean_j = sine_amp * sin(2*pi*A_j)
```

Again: `cumsum_shift` is always an integer, so in real arithmetic it does not change the sine. It changes floating-point rounding and keeps the phase from growing large.

---

### 3.6 Noise amplitude

After nearest upsampling:

```text
UV_j = uv_frame_{floor(j / U)}
```

Noise amplitude:

```text
noise_amp_j = UV_j * noise_std + (1 - UV_j) * (sine_amp / 3)
```

With defaults:

```text
voiced:   noise_amp = 0.003
unvoiced: noise_amp = 0.1 / 3 = 0.033333333333...
```

Generate Gaussian noise:

```text
noise_j = noise_amp_j * gaussian_normal()
```

Final noisy sine:

```text
sine_j = clean_j * UV_j + noise_j
```

For unvoiced frames, the clean sine is muted and only noise remains.

---

### 3.7 `SourceModuleHnNSF`

For `harmonic_num = 0`:

```python
self.l_linear = torch.nn.Linear(1, 1)
self.l_tanh = torch.nn.Tanh()
```

So for each sample:

```text
source_j = tanh(linear_weight * sine_j + linear_bias)
```

where:

```text
linear_weight = l_linear.weight[0][0]
linear_bias   = l_linear.bias[0]
```

This scalar affine transform must be loaded from the model checkpoint. Do not assume `weight = 1` and `bias = 0`.

---

## 4. Does random initial phase matter?

Python:

```python
rand_ini = torch.rand(1, self.dim)
rand_ini[:, 0] = 0
rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
```

For `harmonic_num = 0`:

```text
dim = 1
rand_ini has shape (1, 1)
rand_ini[:, 0] is set to 0
```

Therefore:

```text
rand_ini = 0
```

So for your configuration:

> Random initial phase has zero effect.

It only matters if `harmonic_num > 0`. Even then, the fundamental at index 0 gets zero random phase; only harmonics get random initial phase.

---

## 5. Step-by-step pseudocode

This is the exact port for `harmonic_num = 0`.

```text
Inputs:
    f0[0 .. T-1]
    sr = sampling_rate
    U = upp
    sine_amp = 0.1
    noise_std = 0.003
    voiced_threshold = 0
    linear_weight
    linear_bias

Output:
    out[0 .. N-1], where N = T * U

Allocate:
    r[0 .. T-1]
    V[0 .. T-1]
    uv_frame[0 .. T-1]

cum = 0

for i = 0 .. T-1:
    x = f0[i] / sr
    r[i] = frac(x)
    cum = cum + r[i]
    V[i] = cum * U
    if f0[i] > voiced_threshold:
        uv_frame[i] = 1.0
    else:
        uv_frame[i] = 0.0

phase = 0
prev_p = 0

for j = 0 .. N-1:
    frame = floor(j / U)

    inc = r[frame]
    uv = uv_frame[frame]

    if T == 1 or N == 1:
        y = V[0]
    else:
        pos = j * (T - 1) / (N - 1)
        k = floor(pos)
        a = pos - k

        if k >= T - 1:
            y = V[T - 1]
        else:
            y = (1 - a) * V[k] + a * V[k + 1]

    p = frac(y)

    if j == 0:
        shift = 0
    else:
        if p < prev_p:
            shift = -1
        else:
            shift = 0

    prev_p = p

    phase = phase + inc + shift

    clean = sine_amp * sin(2*pi*phase)

    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    noise = noise_amp * gaussian_normal()

    sine_noisy = clean * uv + noise

    out[j] = tanh(linear_weight * sine_noisy + linear_bias)
```

For deterministic debugging, set:

```text
noise = 0
```

or add an `enable_noise` flag.

---

## 6. Reference C implementation

This implementation prioritizes correctness and clarity. It uses `double` internally for phase and interpolation. Final output is `float`.

Compile with:

```sh
gcc source.c -lm
```

```c
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int sample_rate;       /* from model config, e.g. 48000 */
    int upp;               /* total upsample factor, e.g. 480 */
    float sine_amp;        /* default 0.1f */
    float noise_std;       /* default 0.003f */
    float voiced_threshold;/* default 0.0f */
    bool enable_noise;     /* false for deterministic debug */
} RvcSineGenParams;

typedef struct {
    RvcSineGenParams sinegen;
    float linear_weight;   /* l_linear.weight[0][0] */
    float linear_bias;     /* l_linear.bias[0] */
} RvcSourceModuleParams;

/* Fractional part matching Python/PyTorch modulo 1 for normal values. */
static double rvc_frac1(double x) {
    return x - floor(x);
}

/* ------------------------------------------------------------------ */
/* Simple RNG for noise.                                              */
/* This is not intended to match PyTorch's RNG exactly.               */
/* For exact tensor comparison, disable noise or match Philox RNG.    */
/* ------------------------------------------------------------------ */

static uint64_t rvc_lcg_next(uint64_t *state) {
    *state = (*state) * UINT64_C(6364136223846793005)
                    + UINT64_C(1442695040888963407);
    return *state;
}

static float rvc_uniform01(uint64_t *state) {
    uint64_t x = rvc_lcg_next(state);
    /* 53-bit mantissa uniform in [0, 1) */
    return (float)((double)(x >> 11) * (1.0 / 9007199254740992.0));
}

/* One Gaussian sample using Box-Muller. */
static float rvc_randn_bm(uint64_t *state) {
    float u1;
    do {
        u1 = rvc_uniform01(state);
    } while (u1 <= 1e-30f);

    float u2 = rvc_uniform01(state);

    float radius = sqrtf(-2.0f * logf(u1));
    float angle  = cosf(6.283185307179586f * u2);

    return radius * angle;
}

/* ------------------------------------------------------------------ */
/* SineGen forward for harmonic_num = 0.                              */
/* ------------------------------------------------------------------ */

int rvc_sinegen_forward(
    const float *f0,
    int T,
    const RvcSineGenParams *params,
    uint64_t *rng_state,
    float *out_noisy_sine,     /* length T * upp */
    float *out_uv,             /* optional, length T * upp */
    float *out_noise           /* optional, length T * upp */
) {
    if (!params || !f0 || !out_noisy_sine) return -1;
    if (T <= 0) return -1;
    if (params->sample_rate <= 0) return -1;
    if (params->upp <= 0) return -1;

    const size_t U = (size_t)params->upp;
    const size_t N = (size_t)T * U;

    double *r = NULL;
    double *V = NULL;
    float *uv_frame = NULL;

    r = (double *)malloc((size_t)T * sizeof(double));
    V = (double *)malloc((size_t)T * sizeof(double));
    uv_frame = (float *)malloc((size_t)T * sizeof(float));

    if (!r || !V || !uv_frame) {
        free(r);
        free(V);
        free(uv_frame);
        return -1;
    }

    /* ------------------------------------------------------------ */
    /* Frame-domain quantities.                                     */
    /* ------------------------------------------------------------ */

    double cum = 0.0;

    for (int i = 0; i < T; ++i) {
        double x = (double)f0[i] / (double)params->sample_rate;

        /* rad_values = (f0 / sampling_rate) % 1 */
        double rv = rvc_frac1(x);

        /*
         * For harmonic_num = 0:
         * rand_ini[:, 0] = 0, so no random initial phase.
         */
        r[i] = rv;

        cum += rv;

        /*
         * tmp_over_one before interpolation:
         * cumsum(rad_values, dim=1) * upp
         */
        V[i] = cum * (double)U;

        /* uv = (f0 > voiced_threshold).float() */
        uv_frame[i] = (f0[i] > params->voiced_threshold) ? 1.0f : 0.0f;
    }

    /* ------------------------------------------------------------ */
    /* Audio-rate generation.                                       */
    /* ------------------------------------------------------------ */

    const double two_pi = 2.0 * M_PI;

    double phase = 0.0;
    double prev_p = 0.0;

    for (size_t j = 0; j < N; ++j) {
        /* nearest interpolation index */
        const size_t frame = j / U;

        /* nearest-upsampled rad_values */
        const double inc = r[frame];

        /* -------------------------------------------------------- */
        /* Linear interpolation of V with align_corners=True.       */
        /* -------------------------------------------------------- */

        double y;

        if (T == 1 || N == 1) {
            y = V[0];
        } else {
            double pos = (double)j * (double)(T - 1) / (double)(N - 1);
            size_t k = (size_t)pos;
            double a = pos - (double)k;

            if (k >= (size_t)(T - 1)) {
                y = V[T - 1];
            } else {
                y = (1.0 - a) * V[k] + a * V[k + 1];
            }
        }

        /* tmp_over_one %= 1 */
        double p = rvc_frac1(y);

        /* -------------------------------------------------------- */
        /* cumsum_shift.                                            */
        /* -------------------------------------------------------- */

        double shift = 0.0;

        if (j > 0 && p < prev_p) {
            shift = -1.0;
        }

        prev_p = p;

        /* -------------------------------------------------------- */
        /* Final cumulative phase.                                  */
        /* -------------------------------------------------------- */

        phase += inc + shift;

        float clean = (float)(sin(two_pi * phase) * (double)params->sine_amp);

        /* nearest-upsampled uv */
        float uv = uv_frame[frame];

        /* noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3 */
        float noise_amp =
            uv * params->noise_std +
            (1.0f - uv) * (params->sine_amp / 3.0f);

        float noise = 0.0f;
        if (params->enable_noise && rng_state != NULL) {
            noise = noise_amp * rvc_randn_bm(rng_state);
        }

        /* sine_waves = sine_waves * uv + noise */
        out_noisy_sine[j] = clean * uv + noise;

        if (out_uv != NULL) {
            out_uv[j] = uv;
        }

        if (out_noise != NULL) {
            out_noise[j] = noise;
        }
    }

    free(r);
    free(V);
    free(uv_frame);

    return 0;
}

/* ------------------------------------------------------------------ */
/* SourceModuleHnNSF forward for harmonic_num = 0.                    */
/* ------------------------------------------------------------------ */

int rvc_source_module_forward(
    const float *f0,
    int T,
    const RvcSourceModuleParams *params,
    uint64_t *rng_state,
    float *out_source,                /* length T * upp */
    float *optional_sine_buffer       /* optional length T * upp */
) {
    if (!params || !f0 || !out_source) return -1;
    if (T <= 0) return -1;
    if (params->sinegen.upp <= 0) return -1;

    const size_t N = (size_t)T * (size_t)params->sinegen.upp;

    float *sine = optional_sine_buffer;
    bool sine_allocated = false;

    if (sine == NULL) {
        sine = (float *)malloc(N * sizeof(float));
        if (!sine) return -1;
        sine_allocated = true;
    }

    int rc = rvc_sinegen_forward(
        f0,
        T,
        &params->sinegen,
        rng_state,
        sine,
        NULL,
        NULL
    );

    if (rc == 0) {
        for (size_t j = 0; j < N; ++j) {
            float x = params->linear_weight * sine[j] + params->linear_bias;
            out_source[j] = tanhf(x);
        }
    }

    if (sine_allocated) {
        free(sine);
    }

    return rc;
}
```

Example setup:

```c
RvcSourceModuleParams params;

params.sinegen.sample_rate = 48000;       /* from config */
params.sinegen.upp = 480;                 /* from config */
params.sinegen.sine_amp = 0.1f;
params.sinegen.noise_std = 0.003f;
params.sinegen.voiced_threshold = 0.0f;
params.sinegen.enable_noise = true;       /* false for phase debugging */

params.linear_weight = loaded_weight;     /* l_linear.weight[0][0] */
params.linear_bias = loaded_bias;         /* l_linear.bias[0] */

uint64_t rng_state = 0x123456789ABCDEFULL;

int rc = rvc_source_module_forward(
    f0,
    T,
    &params,
    &rng_state,
    source_out,
    NULL
);
```

---

## 7. Simplified mathematically equivalent clean-sine version

If you do not need bitwise PyTorch float32 compatibility, and only need the correct clean sine, you can skip the entire `tmp_over_one` / `cumsum_shift` machinery.

For `harmonic_num = 0`, the ideal clean sine is:

```c
double phase = 0.0;

for (size_t j = 0; j < N; ++j) {
    size_t frame = j / U;

    phase += (double)f0[frame] / (double)sample_rate;

    /* Keep phase in [0, 1) for numerical stability. */
    phase -= floor(phase);

    clean[j] = (float)(sin(2.0 * M_PI * phase) * sine_amp);
}
```

This is mathematically equivalent to the PyTorch clean sine for `harmonic_num = 0`, because the PyTorch `cumsum_shift` only subtracts integer cycles.

However, if your goal is to reproduce PyTorch's float32 tensor values as closely as possible, use the full reference implementation above.

---

## 8. Edge cases

### 8.1 `f0[i] = 0`, unvoiced

For an unvoiced frame:

```text
r_i = 0
uv_frame_i = 0
```

During that frame:

```text
clean sine is muted
noise_amp = sine_amp / 3 = 0.033333...
output is only noise before linear/tanh
```

Phase accumulation does not advance during `f0 = 0` frames because the increment is zero.

### 8.2 First sample

PyTorch uses cumulative sum inclusive of the current sample.

So the first output sample uses:

```text
A_0 = R_0 = r_0
```

not zero.

Do this:

```c
phase += inc;
sample = sin(2*pi*phase);
```

Not:

```c
sample = sin(2*pi*phase);
phase += inc;
```

unless you intentionally want a one-sample phase shift.

### 8.3 Last sample

Nearest interpolation maps:

```text
j = N - 1
frame = floor((N - 1) / U) = T - 1
```

So the last audio sample uses the last f0 frame.

Linear `align_corners=True` also maps:

```text
j = N - 1
pos = T - 1
Y = V_{T-1}
```

### 8.4 `T == 1`

When there is only one f0 frame, linear interpolation cannot use `(N - 1)` in the usual way. PyTorch effectively repeats the single input value.

Use:

```text
Y_j = V_0
```

for all `j`.

Nearest still replicates the only frame `U` times.

### 8.5 `U == 1`

If `upp == 1`, then `N = T`.

Nearest becomes identity:

```text
frame = j
```

Linear `align_corners=True` also becomes identity for integer positions.

The algorithm still works.

---

## 9. Numerical precision: should phase be `float` or `double`?

Recommended:

> Use `double` for phase accumulation and interpolation math.  
> Output `float` after `sin`, noise, linear, and tanh.

Reason:

`float` has only about 7 decimal digits of precision. Audio-length phase accumulators can become large, and `sinf` with a large argument can lose precision.

The PyTorch code uses float32 tensors and inserts integer-cycle corrections to reduce the argument size. If you use `double`, you can simply keep phase reduced to `[0, 1)`:

```c
phase -= floor(phase);
```

If you need near-bitwise compatibility with PyTorch float32:

- Use `float` for tensor-like values.
- Use `sinf`, `tanhf`, `floorf`.
- Replicate the exact `tmp_over_one` / `cumsum_shift` logic.
- Expect small differences anyway due to library `sinf` and hardware differences.

For a native C RVC implementation, numerical stability is usually more important than bitwise PyTorch compatibility. Use `double` internally.

---

## 10. Why your previous `phase += 2*pi*f0_interp/sr` was wrong

You said you previously did something like:

```c
phase += 2*pi * f0_interp[t] / sr;
```

with linear interpolation of f0.

There are several issues.

### 10.1 PyTorch does not linearly interpolate f0 for the phase increment

The actual increment stream is:

```text
R_j = r_{floor(j / U)}
```

That is nearest/zero-order hold.

If you do:

```text
phase += f0_linear[j] / sr
```

you are integrating a different frequency trajectory.

This is the most important algorithmic difference.

### 10.2 Linear interpolation with `align_corners=True` is not the same as frame hold

If you used `align_corners=True` to interpolate f0, then:

```text
pos = j * (T - 1) / (N - 1)
```

This does not place frame boundaries at multiples of `U`.

For example, with `T = 410`, `U = 480`, `N = 196800`:

```text
pos_j = j * 409 / 196799
```

That is not the same as:

```text
frame_j = floor(j / 480)
```

The PyTorch code uses nearest for the actual increments.

### 10.3 Random initial phase is not the cause for `harmonic_num = 0`

For `dim = 1`:

```text
rand_ini = 0
```

So random initial phase cannot explain the mismatch.

### 10.4 Wrap correction is mostly numerical, not mathematical

Because `cumsum_shift` subtracts integer cycles:

```text
sin(2*pi*(phase + integer)) = sin(2*pi*phase)
```

So the wrap correction cannot fix a fundamentally wrong frequency/phase interpolation in exact math.

It matters if:

- you use float32,
- the phase grows large,
- you want to reproduce PyTorch rounding.

But if your clean phase is otherwise correct and you use double plus modulo reduction, the wrap correction is not required for waveform correctness.

### 10.5 Noise can produce relL2 around sqrt(2)

This is important.

If you compare final noisy outputs with different RNGs, and the signal is mostly unvoiced/noise-like, two independent noise signals have:

```text
||x - y|| / ||x|| ≈ sqrt(2) ≈ 1.414
```

Your reported:

```text
relL2 = 1.47
```

is very close to what you can get from comparing independent noise.

Therefore, before debugging phase:

1. Disable noise in both implementations, or compare the clean sine before noise.
2. Verify `linear_weight` and `linear_bias`.
3. Verify nearest f0 hold, not linear f0 interpolation.

---

## 11. Validation checklist

Use this order.

### Step 1: Compare clean sine, not final source

Temporarily disable noise in C:

```c
params.sinegen.enable_noise = false;
```

In Python, either:

- compare before noise is added, or
- set noise to zeros, or
- monkeypatch `torch.randn_like` to zeros.

Compare:

```text
clean sine before l_linear/l_tanh
```

If this matches, phase is correct.

### Step 2: Use a constant f0

Example:

```text
f0[i] = 100.0 for all i
sr = 48000
U = 480
```

Then:

```text
r = 100 / 48000 = 1/480
```

Each frame advances exactly one cycle.

The clean output should be a pure 100 Hz sine.

First clean sample should be approximately:

```text
0.1 * sin(2*pi/480)
```

not zero.

### Step 3: Test f0 = 0

If all f0 are zero:

```text
clean sine should be zero
uv should be zero
output before linear/tanh should be noise only with std 0.1/3
```

If noise is disabled, output before linear/tanh should be zero, then after linear/tanh:

```text
tanh(bias)
```

if bias is nonzero.

### Step 4: Verify linear layer

For `harmonic_num = 0`:

```text
source = tanh(w * sine_noisy + b)
```

Load:

```text
w = source_module.l_linear.weight[0][0]
b = source_module.l_linear.bias[0]
```

Do not assume identity.

### Step 5: Only then enable noise

If noise is enabled, exact sample-by-sample comparison requires matching PyTorch's RNG and Gaussian implementation.

Using an LCG + Box-Muller is fine for inference, but it will not bitwise match PyTorch.

---

## 12. Minimal mental model

For your exact RVC configuration:

```text
f0 frame rate = sample_rate / upp
```

Each f0 frame describes the frequency for `upp` audio samples.

The source generator does:

```text
for each audio sample:
    use current frame's f0 without interpolation
    phase += f0_frame / sample_rate
    sine = 0.1 * sin(2*pi*phase)
    if voiced:
        add small noise, std 0.003
    else:
        mute sine and add larger noise, std 0.1/3
    output = tanh(w * noisy_sine + b)
```

The complicated PyTorch wrap logic is a float-friendly integer-cycle remover. Implement it if you want to mirror PyTorch closely; otherwise a double accumulator with modulo reduction is simpler and more stable.