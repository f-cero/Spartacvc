# Tray App Polish Spec (Release-Ready)

The tray app (`spartacvc_tray.exe`) is the **main app for release**.
This spec covers the features needed to make it production-quality.

---

## 1. Icon

- **Design confirmed:** green angular "S" on dark maroon rounded square
  (source: user-provided PNG, 256×256).
- Tray icon and file icon must be the **same image**.
- Multi-resolution .ico: 16, 24, 32, 48, 64, 128, 256 px
- Embedded via `spartacvc.rc` → `spartacvc.res` → linker

### 1a. Tray icon states (colorblind-friendly, shape-differentiated)

| State | Visual | Shape cue |
|-------|--------|----------|
| **Loaded** (default) | Full-color S on maroon | Solid filled S |
| **Loading** | S with diagonal stripe overlay / half-filled | Partial fill |
| **Unloaded** | Outline-only S on gray | Hollow S, no color |

- Colorblind safety: states differ by **fill** (solid / striped / outline),
  not just hue. Deuteranopia/protanopia safe.
- 3 separate icons in the .ico file (or 3 .ico files), swapped via
  `Shell_NotifyIconW(NIM_MODIFY)` with different `hIcon`.
- File explorer icon: always the loaded (full-color) variant.

---

## 2. Model Loading

### 2a. Load on launch
- On startup, load the model saved in `spartacvc.cfg` (if any).
- Show loading status in tray tip: `Spartacvς [Vulkan] loading nahida…`
- When loaded, tip changes to: `Spartacvς [Vulkan] nahida ●`

### 2b. Load on model selection
- Selecting a model from the menu triggers an async load.
- The load runs on a worker thread (not the UI thread).
- Menu stays responsive during load.

### 2c. Cancel in-progress load
- If the user selects a different model while a load is in progress:
  1. Signal the current load thread to stop (atomic flag).
  2. Wait for it to finish (bounded timeout ~2s).
  3. Start the new load.
- The old model's GPU tensors are freed before the new load begins.

### 2d. VRAM residency toggle
- **Resident** (default): model weights stay in VRAM after load.
  Fast inference, uses VRAM.
- **On-demand**: model weights are loaded to VRAM for each inference
  call, then evicted. Slower inference, saves VRAM.
- Toggle in menu: `Model: ● Resident / ○ On-demand`
- Saved in `spartacvc.cfg` as `resident=1` or `resident=0`.

### 2e. Force evict / unload
- Menu item: `Unload model` — frees all GPU/CPU tensors.
- After unload, tip shows: `Spartacvς [Vulkan] (no model)`
- Next inference triggers a reload.

### 2f. Implementation notes
- The ggml pipeline currently loads the model from disk on every call
  (`spcv_ggml_infer(gguf_path, ...)`). This must be refactored:
  - **`spcv_model_load(path)`** → returns a `spcv_model_t*` handle
  - **`spcv_model_infer(handle, ...)`** → uses the loaded handle
  - **`spcv_model_free(handle)`** → frees all tensors
  - Handle stores: ggml context, tensor pointers, config (dim, sr, fps)
- The handle is stored in a global (or passed to the worker thread).
- Thread safety: only one load/infer at a time (mutex or generation counter).

---

## 3. Model Conversion Dialog

A proper Win32 dialog window (not a menu item) for converting `.pth` → `.gguf`
or other formats.

### 3a. Layout

```
┌─────────────────────────────────────────────────────────┐
│  Convert Models                                          │
├─────────────────────────────────────────────────────────┤
│  Source models:                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ ☑ nahida-jp.pth                                   │  │
│  │ ☐ travisscottv2.pth                               │  │
│  │ ☐ beyonce.pth                                     │  │
│  │ ☐ lana.pth                                        │  │
│  │  (multi-select listbox, scans models/voices/)     │  │
│  └───────────────────────────────────────────────────┘  │
│  [+ Browse for more...]                                  │
│                                                          │
│  Output type:  (●) GGUF    ( ) ONNX    ( ) Both         │
│                                                          │
│  Destination:  [models\voices\          ] [Browse...]   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Progress:  2/4 done   62% total                   │  │
│  │ Current:   nahida-jp  ████████████░░░░  78%       │  │
│  │ Remaining: 2 models (~45s est.)                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  [  Begin  ]  [  Cancel  ]                               │
└─────────────────────────────────────────────────────────┘
```

### 3b. Behavior

- **Source list**: scans `models/voices/` for `.pth` files (and any other
  convertible format). Multi-select checkbox list.
- **Browse for more**: `GetOpenFileNameW` with multi-select
  (`OFN_ALLOWMULTISELECT`), filter `*.pth;*.onnx`.
- **Output type**: radio buttons — GGUF, ONNX, Both.
  - GGUF: `.pth` → `.gguf` (via `pth2gguf` code, in-process)
  - ONNX: `.pth` → `.onnx` (via Python export, `tools/export_pth.py`)
  - Both: run both conversions
- **Destination**: default = `models/voices/` (next to exe).
  `Browse...` → folder picker (`SHBrowseForFolderW` or IFileDialog).
- **Begin**: starts conversion on a worker thread. UI stays responsive.
- **Progress tracker** (updated via `PostMessageW` from worker):
  - `X / Y models converted`
  - `Z remaining`
  - `Current: <model name>`
  - `Total progress: NN%`
  - `Current model progress: NN%` (if the converter can report it)
  - Estimated time remaining
- **Cancel**: sets a stop flag; worker checks between models and
  (if possible) between tensor conversions.
- On completion: message box "Done: 4/4 models converted to models\voices\".

### 3c. Implementation notes

- The `pth2gguf` conversion code already exists in C (`src/pth2gguf.c`).
  It can be called in-process from the tray app (no CLI round-trip).
- For ONNX export, we still need Python (`tools/export_pth.py`).
  The dialog should detect if Python is available and warn if not.
- Progress reporting: the C `pth2gguf` code needs a progress callback
  (`typedef void (*spcv_progress_fn)(int tensors_done, int tensors_total,
  const char *tensor_name)`).
- The dialog is a separate Win32 window (not a property sheet).
  Created on demand, destroyed on close.

---

## 4. Menu Layout (Final)

```
Spartacvς
─────────────────────────────────
Model: nahida ▸
  ✓ nahida
    travisscottv2
    beyonce
  ─────────────────
  Convert models...
  [✓] Resident in VRAM
  Unload model
─────────────────────────────────
Pitch: +2 ▸
  (submenu: -12 … 0 … +12)
─────────────────────────────────
Devices ▸
  In:  Microphone (Realtek)
  Out: Spartacvς (RVC output) ▸
    Spartacvς (RVC output)
    Speakers (Realtek)
    …
  ─────────────────
  Install virtual device
  Uninstall virtual device
  Rename devices...
─────────────────────────────────
Runtime ▸
  ✓ Vulkan  (Radeon Vega FE)
    CUDA    (Tesla PG500)
    CPU
─────────────────────────────────
● Running / Start
─────────────────────────────────
Convert file...
─────────────────────────────────
Quit
```

### 4a. Pitch as submenu
- **Bug:** 25 pitch items in the main menu made it taller than the screen;
  Windows stretched item heights (~2.5×). Fix: pitch is a submenu.
- Parent item shows current value: `Pitch: +2 ▸`
- Submenu: -12 … -1, 0, +1 … +12 with checkmark on current.

### 4b. Device config submenu
- In/Out device selection moves here.
- **Install virtual device** — installs the Spartacvς WASAPI virtual
  audio endpoint pair (see §8). Requires admin. Shows progress.
- **Uninstall virtual device** — removes it. Requires admin.
- **Rename devices...** — dialog to rename the virtual devices
  (e.g., "Spartacvς (RVC output)" → "My RVC Mic").
  Implemented via `IPropertyStore::SetValue(DEVPKEY_Device_FriendlyName)`
  on the device's property store (requires admin + audio service restart).
  Fallback: instruct user to rename in Windows Sound settings.

### 4c. Runtime submenu
- Backend selection (CPU/CUDA/Vulkan) moves here.
- Shows GPU name next to each GPU backend.
- Grayed out if the runtime DLL is missing (see §9).

---

## 5. Virtual Audio Device (MANDATORY, not optional)

> User directive: "I know #5 option B is a big deal, but it's not really
> optional, we can't rely on other software here, I want the app to
> 'just work'."

### 5a. What the user sees

Two virtual audio endpoints appear in Windows Sound settings:

| Device name | Role | User action |
|-------------|------|-------------|
| **Spartacvς (RVC output)** | Virtual **mic** — other apps (Discord, games) select this as input | Nothing — it just appears |
| **Spartacvς input (send audio here to be RVC'd)** | Virtual **speaker** — user selects this as the output device in Spartacvς | Select in Spartacvς menu |

Audio flow:
```
Mic (real) → WASAPI capture → RVC pipeline → WASAPI render
  → "Spartacvς input (send audio here to be RVC'd)" [virtual speaker]
  → loopback → "Spartacvς (RVC output)" [virtual mic]
  → Discord / game / any app
```

### 5b. Implementation options (in order of preference)

1. **Bundle a WASAPI virtual audio driver** (kernel-mode audio class driver)
   - Based on open-source: `virtual-audio-cable` (CABLE) is proprietary;
     look at `wasapi-virtual-audio` or write a minimal KMAudio driver.
   - Requires: Windows Driver Kit (WDK), driver signing (EV cert or
     test-signing mode), installer (MSI or NSIS) to register the driver.
   - The driver creates a paired virtual endpoint (render + capture)
     that loops back: whatever is rendered to endpoint A is captured
     from endpoint B.
   - Spartacvς renders RVC output to endpoint A; apps capture from B.
   - **This is the correct solution.** Significant work (driver + installer).

2. **WASAPI loopback + Process Loopback API** (Windows 10 2004+)
   - `IAudioClient3::InitializeAudioClient3` with process loopback.
   - Can capture audio from a specific process without a virtual device.
   - **Does NOT create a virtual device** — doesn't solve the problem.

3. **WASAPI shared-mode render to a named stream** (Windows 11+)
   - `IAudioClient8` with `AUDCLNT_STREAMFLAGS_...` — no, this doesn't
     exist for creating virtual endpoints.

### 5c. Realistic path

- **Short-term (this release):** Ship with VB-Cable / VoiceMeeter
  auto-detection. The tray app scans for known virtual cables
  ("CABLE Input", "VoiceMeeter", "Steam Audio") and tags them
  "(virtual)" in the device list. If none found, show a setup dialog
  with one-click VB-Cable download link.
- **Medium-term (next release):** Bundle a signed WASAPI virtual
  audio driver. "Install virtual device" menu item runs the installer.
  "Uninstall virtual device" removes it. Portable mode skips this.
- **The driver work is the critical path for "just works" UX.**

### 5d. Device naming
- Default names:
  - Output (virtual speaker): `Spartacvς input (send audio here to be RVC'd)`
  - Input (virtual mic): `Spartacvς (RVC output)`
- User can rename via:
  - Spartacvς menu → Devices → Rename devices...
  - Windows Sound settings → Right-click → Rename
- Rename is stored in the registry
  (`HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\...`)
  or via the device property store (`DEVPKEY_Device_FriendlyName`).

---

## 6. Configuration

### 6a. Development: `spartacvc.cfg` (next to exe)

```ini
backend=3            # 0=cpu, 1=cuda, 2=dml, 3=vulkan
model=nahida.gguf    # filename in models/voices/
model_dir=models\voices\
dev_in=0             # index into WASAPI capture list
dev_out=0            # index into WASAPI render list
pitch=0              # semitones, -12..+12
resident=1           # 1=keep in VRAM, 0=on-demand
```

### 6b. Release: Windows Registry (primary) + .cfg (fallback)

- **Installed mode** (default): config in
  `HKCU\Software\Spartacvc\` (per-user, no admin needed).
  - `Backend` (DWORD), `Model` (SZ), `ModelDir` (SZ)
  - `DevIn` (DWORD), `DevOut` (DWORD), `Pitch` (DWORD)
  - `Resident` (DWORD), `Portable` (DWORD)
- **Portable mode** (thumb drive): `Portable=1` in registry OR
  `spartacvc.cfg` next to exe. All config stays in the .cfg file.
  No registry writes. No virtual device installation.
- **Precedence:** registry (if `Portable=0`) → .cfg file → defaults.
- **Write:** always write to both (registry if installed, .cfg if portable).
  This way, copying the folder to a USB drive carries the config.

### 6c. Installer options (NSIS or MSI)

- **[✓] Install virtual audio device** (default: checked)
  - Installs the WASAPI virtual audio driver (when available).
  - Requires admin (UAC prompt).
- **[✓] Add to startup** (default: unchecked)
  - Auto-launch Spartacvς on user logon.
- **[ ] Portable mode** (default: unchecked)
  - If checked: no registry, no virtual device, no startup.
  - App runs from any folder (USB drive, etc.).
  - Runtimes still loaded from beside the exe.
- **Runtime selection** (checkboxes):
  - [✓] CUDA runtime (NVIDIA GPU) — installs `ggml-cuda.dll`
  - [✓] Vulkan runtime (AMD/Intel/NVIDIA) — installs `ggml-vulkan.dll`
  - CPU is always included (in the main exe).
  - User without NVIDIA can uncheck CUDA → smaller download, no junk.

---

## 7. Runtime DLLs (separate from main exe)

> User directive: "separate the runtimes into DLLs which will sit beside
> the app and load only on demand, of course with graceful fallbacks if
> the file is missing/corrupt; there should also be an option in the
> installer for a portable installation... (another reason to pull the
> runtimes out into DLLs: smaller file size for submission to virus
> checking websites)"

### 7a. Layout

```
$spartacvc/                    # install folder (or USB drive)
├── spartacvc_tray.exe         # main app (CPU backend built-in)
├── spartacvc.exe              # CLI (CPU backend built-in)
├── ggml-cuda.dll              # CUDA runtime (optional, ~30MB)
├── ggml-vulkan.dll            # Vulkan runtime (optional, ~15MB)
├── cublas64_12.dll            # CUDA dependency (if CUDA installed)
├── cublasLt64_12.dll          # CUDA dependency
├── cudart64_12.dll            # CUDA runtime
├── models/
│   ├── base/                  # HuBERT + RMVPE
│   └── voices/                # voice models
├── spartacvc.cfg              # config (portable mode)
└── LICENSE
```

### 7b. Loading strategy

- Main exe contains: ggml-base, ggml-cpu, pipeline code, WASAPI, tray.
- **CUDA:** `LoadLibraryW("ggml-cuda.dll")` from exe directory.
  - If missing → CUDA unavailable, menu shows "CUDA (not installed)".
  - If corrupt → `LoadLibraryW` fails, same fallback.
  - Resolve `ggml_backend_reg` via `GetProcAddress`.
- **Vulkan:** `LoadLibraryW("ggml-vulkan.dll")` from exe directory.
  - Same pattern.
  - Vulkan SDK DLL (`vulkan-1.dll`) must also be present or in PATH.
- **CPU:** always available (statically linked).
- **Selection:** user picks backend in Runtime submenu.
  If the chosen backend's DLL is missing, show a warning and fall back
  to CPU. Never crash.

### 7c. Build changes

- `build_tray.bat` / `build_spcv.bat`: link only `ggml-base.lib` +
  `ggml-cpu.lib` statically. Do NOT link `ggml-cuda.lib` / `ggml-vulkan.lib`.
- New: `build_runtimes.bat` — builds `ggml-cuda.dll` and
  `ggml-vulkan.dll` as shared libraries (CMake `BUILD_SHARED_LIBS=ON`).
- The DLLs export the standard ggml backend registration symbols:
  - `ggml_backend_cuda_reg` (or `ggml_backend_reg_cuda`)
  - `ggml_backend_vk_reg`
- Main exe calls `ggml_backend_load_all_from_library()` or manually
  calls the reg function after `LoadLibraryW`.

### 7d. Virus scanner benefit

- Main exe: ~50MB (CPU only, no CUDA/Vulkan bloat).
- CUDA DLL: ~30MB (separate file, easy to scan).
- Vulkan DLL: ~15MB (separate file, easy to scan).
- Total on VirusTotal: 3 files instead of one 200MB monolith.
- Faster scanning, less likely to hit size limits.

## 8. Implementation Priority (revised)

1. **Icon** — generate .ico from user PNG + 3 state variants (§1)
2. **Pitch submenu** — fix the 25-item menu height bug (§4a)
3. **Menu reorganization** — Devices/Runtime submenus (§4b, §4c)
4. **Registry config** — HKCU with .cfg fallback (§6b)
5. **Model handle refactor** — load/infer/free (§2f)
6. **Load on launch + cancel + resident toggle + unload** (§2a-2e)
7. **Virtual device** — short-term: cable detection (§5c); long-term: driver
8. **Runtime DLLs** — separate CUDA/Vulkan into DLLs (§7)
9. **Conversion dialog** — Win32 dialog with progress (§3)
10. **Installer** — NSIS with options (§6c)
