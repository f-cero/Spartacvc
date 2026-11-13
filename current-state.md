# Current State — Spartacvς

Written: 2026-11-03 (session handoff). Read this first, then `PROMPT.md`,
`AGENTS.md`, `ROADMAP.md`, `docs/tray-polish-spec.md`.

---

## 1. What the project is

Spartacvς = pure-C RVC voice converter for Windows. Two deliverables:

- **CLI** (`spartacvc.exe`) — offline WAV→WAV conversion, GPU (CUDA/Vulkan/CPU)
- **Tray app** (`spartacvc_tray.exe`) — the main release product: real-time
  mic → RVC → output device, system tray UI

End goal (user's words): other apps (Discord, games) see the user's
RVC-converted voice as a normal microphone. That requires the **virtual audio
driver** (Phase 4), which is the current critical path.

GitHub: `f-cero/Spartacvc` (public). Stylized name "Spartacvς" in docs/UI,
ASCII "spartacvc"/"Spartacvc" in code/paths/filenames.

---

## 2. Overall status at a glance

| Component | State |
|-----------|-------|
| CLI `spartacvc.exe` (file conversion) | ✅ Working (CUDA 2.4s/4.1s audio, Vulkan 13.8s, CPU ok) |
| `.pth → .gguf` conversion | ✅ Working (in-app + standalone, parallelized 0.24s) |
| Tray app core (menu, devices, models, start/stop) | ✅ Working (batch RVC via CLI round-trip, ~2–3s latency) |
| Tray polish: state icons, pitch submenu, device/runtime submenus, resident toggle, model state | ✅ In code (commit 50fe61b) — **not yet user-verified** |
| Virtual audio driver `.sys` | ✅ **LOADING** (2026-11-10, KMDF 1.15 bind; branded Spartacvc) |
| Driver install pipeline (devcon batch) | ✅ Working manually (`C:\temp\install_driver.bat`); tray menu wiring ⬜ |
| Registry config (HKCU) + portable mode | ⬜ Not done (.cfg persistence exists) |
| Model handle refactor (load/infer/free, load-on-launch, cancel, unload) | ⬜ Not done — ggml pipeline reloads from disk every call |
| Conversion dialog (Win32 window w/ progress) | ⬜ Not done (menu item works via CLI) |
| Runtime DLL split (ggml-cuda.dll / ggml-vulkan.dll beside exe) | ⬜ Not done — exes are ~213MB static |
| Installer (NSIS/MSI with options) | ⬜ Not started |
| True real-time RVC in tray (pipeline linked directly, <100ms budget) | ⬜ Not done — currently shells out to spartacvc.exe per chunk |
| Git push | 🔴 **36+ commits ahead of origin/main** — unpushed! |

---

## 3. Completed work (verified)

### CLI / engine
- Full RVC pipeline in C: resample→16k → HuBERT → RMVPE f0 → pitch shift →
  voice decoder. Per-model fps = `sr / prod(upsample_rates)` (Nahida 48k/100fps).
- ggml GPU path: encoder + flow + decoder all verified on CUDA (PG500/V100)
  and Vulkan (Vega FE). `--voice model.gguf` single-file mode.
- `.pth` loading: ZIP+pickle reader in C; `pth2gguf` converter (32 threads);
  GGUF load into ggml context (`src/gguf_load.c`).
- Pitch shift −12…+12 semitones, passed to CLI worker via `--pitch`.
- Config persistence: backend/model/device saved & restored.

### Tray app (`src/tray.c`, `src/tray_main.c`, `src/wasapi.c`, `src/pipeline_rt.c`)
- Win32 tray icon with **3 state icons** (loaded / loading / unloaded —
  shape-differentiated, colorblind-safe; PNGs in `scratch/icon_*.png`).
- Menu: Model submenu (list + Convert models... + Resident toggle + Unload),
  Pitch submenu (−12…+12, fixes the 25-item menu-height bug), Devices submenu
  (In/Out selection), Runtime submenu (Vulkan/CUDA/CPU with GPU names).
- WASAPI capture/render in pure C (COM vtable calls; manual GUIDs in
  `src/com_guids.c` since SDK lib doesn't export them).
- Real-time worker: chunked mic → CLI round-trip → output device. Works,
  but latency ~2–3s (not the <100ms target — that needs pipeline linked in).
- All strings wide/UTF-16. No console window.

### Models on disk (`models/`)
- `base/rmvpe.onnx`, `base/vec-768-layer-12.onnx` (HuBERT 768)
- `voices/nahida.gguf` ← **the standard test voice**, plus `nahida-jp.pth`,
  faiss index files. (Ozada 32k models are gone from disk; re-download via
  hf-cli if needed — see `tools/download_models.md`.)

### Driver prerequisites (this session)
- WDK 10.0.22000.0 installed at `C:\Program Files (x86)\Windows Kits\10`.
- Spectre-mitigated CRT libs for MSVC 14.44 present:
  `...\MSVC\Lib\Spectre\onecore\x64` (installed via `install_spectre.bat`).
- Driver source cloned: `third_party/virtual-audio-driver/` — "Virtual Audio
  Driver by MikeTheTech", standard Microsoft KMDF PortCls sample. Creates a
  **paired** virtual speaker + microphone endpoint (exactly what we need).
  Beta, requires test-signing enabled on the target machine.

---

## 4. Driver .sys — SOLVED (2026-11-10)

Build + install pipeline fully working. Driver loads ("Driver is running"),
endpoints live: "Spartacvc Mic" (capture) / "Spartacvc (RVC output)" (render).

### The complete saga (so nobody re-treads this)
1. Link never ran → fixed via `Directory.Build.targets` (OutputType=sys,
   TargetExt=.sys, GenerateManifest=false, LinkCompiled for Driver config;
   clear @(Link) for StaticLibrary; kernel lib deps + EntryPointSymbol=FxDriverEntry).
2. INF fixes: strip UTF-8 BOM (breaks pnputil/devcon!), drop the
   NTamd64.10.0...22000 decoration (Win11-only — the OFFICIAL MTT release has
   this bug too and does NOT install on Win10!), drop DRM/PETrust
   SignatureAttributes.
3. Signing: self-signed "Spartacvc Test" cert (thumbprint D56B20B6...) in
   Root + TrustedPublisher; testsigning ON; Secure Boot disabled in firmware
   (blocked bcdedit before).
4. **Bind failure (problem 37) root cause: KMDF bind protocol changed above
   1.15.** Every stub >=1.25 (any kit generation, incl. era-correct EWDK
   19041 ones) fails WdfVersionBind on the 19041 runtime with a MISLEADING
   event ("Bind Minor version is greater" even when driver < runtime).
   DriverEntry never ran (registry-marker proof). A/B: minimal KMDF driver
   failed identically; VB Voicemeeter VAIO binds 1.15 and works (user hunch).
   FIX: headers Include\wdf\kmdf\1.15 + stubs Lib\wdf\kmdf\x64\1.15
   (EWDK-19041 copies at C:\temp\ewdk_wdf15 + C:\temp\ewdk_inc15),
   stampinf -k 1.15.

### Build/install pipeline (reproduce end to end)
- Build: `cmd /c C:\temp\build_driver2.bat` (retry loop; "DONE 1" is
  findstr's exit code — success = "Build succeeded" in C:\temp\bld.log)
- Stage: copy .sys -> virtualaudiodriver.sys + .inx -> stampinf -a amd64
  -d * -v 1.0.0.0 -k 1.15 -> inf2cat /os:10_x64 -> signtool sign (test cert)
- Install: `C:\temp\install_driver.bat` elevated (devcon, ROOT\VirtualAudioDriver)
- 22000 stub backup: C:\temp\wdf131_22000_backup; EWDK ISO: C:\temp\ewdk19041.iso

### Next driver steps
- Wire tray Install/Uninstall virtual device menu (reuse staging pipeline)
- For release: MS attestation signing (EV cert + partner portal) so end users
  dont need testsigning; INX branding done (Spartacvc).
## 4b. FAISS retrieval — wired (2026-11-13)

- CLI: `--index <path>` + `--index-rate <r>` — verified E2E (output differs
  with/without; af_heart -> nahida @ cpu, 8.4s). Blend feeds all paths
  (ggml/ONNX/CPU) since it happens on emb_model before the decoder.
- Tray + rt worker: auto-discover `<model>*.index.bin` next to the voice
  model (src/index_discover.h) and pass `--index`. Dim mismatch is safe
  (main.c skips with warning).
- nahida index: models/voices/nahida-jp.index.bin (SPCVIX01 format, d=768).
  Re-extract from .pth via tools/extract_index.py if lost.
- Distribution note: spartacvc.exe needs beside it: onnxruntime.dll,
  DirectML.dll, onnxruntime_providers_shared.dll (third_party/onnxruntime),
  cublas64_12/cublasLt64_12/cudart64_12 (kokoro-tts folder has copies), and
  system VC redist (VCOMP140/MSVCP140/VCRUNTIME140). Fold into installer.
## 5. Environment quirks & rules (learned the hard way)

1. **X: drive is flaky** — intermittent "No such device" / MSB4025 on reads
   and writes. Retry loops work. Don't panic; don't assume file corruption.
2. **Batch files MUST be CRLF.** LF-only .bat makes cmd.exe silently exit.
   After any tool-write: `sed -i 's/$/\r/' file.bat` (or verify with od).
3. **git-bash quirks:** `/v:normal` gets mangled → use `-verbosity:normal`;
   `printf` escapes eat `\t`/`\v` in Windows paths → build .bat files with
   `echo` lines + sed CRLF, not printf; MSYS sometimes can't read a file that
   cmd.exe reads fine (and vice versa) — test both.
4. **MSBuild env:** WDK vars (`DDK_INC_PATH`, `KM_IncludePath`) are NOT set
   outside the WDK dev prompt → Directory.Build.props supplies them. MSBuild's
   C++ targets override inherited INCLUDE/LIB anyway; use props files, not env.
5. **Kernel builds force /WX** — suppress header warnings via `/wd` flags in
   AdditionalOptions (already done in Directory.Build.props).
6. **Solution files reject `-getProperty`** (MSB1063) — query the .vcxproj instead.
7. **Rules from AGENTS.md:** pure C; heavy documentation; debug tracing
   (compile flag + runtime `--debug`); structure before code; frequent git
   commits; ask user when blocked; timeouts on everything; never put files on
   desktop; Python only in tools/ (dev-time), never at runtime.
8. **TEST PROTOCOL (sacrosanct):** input = kokoro af_heart WAV, voice = nahida.
   No deviations — they invalidate the test.
9. Git identity: `f-cero`. Push when work is verified.

---

## 6. TODO list (release order)

1. ✅ **Driver .sys** — BUILT 2026-11-08 (§4), verified x64 native. Next: test-sign + install-test on this machine
2. **Tray verification pass** — user-test the polish features in commit 50fe61b
   (state icons, submenus, resident toggle); fix anything broken
3. **Model handle refactor** (`spcv_model_load/infer/free`) + load-on-launch +
   cancel-in-progress + unload — spec §2f; needed before real-time works well
4. **True real-time RVC in tray** — link pipeline into tray exe instead of CLI
   round-trip (spec Phase 3: <100ms budget on GPU)
5. **Registry config** HKCU\Software\Spartacvc + portable mode fallback (§6b)
6. **Runtime DLL split** — ggml-cuda.dll / ggml-vulkan.dll loaded on demand (§7);
   shrinks main exe from 213MB, helps virus scanners
7. **Conversion dialog** with progress (§3)
8. **Driver installer** (NSIS/MSI): install/uninstall virtual device, startup
   option, portable mode, runtime checkboxes (§6c)
9. **E2E test everything:** CLI conversion, tray real-time, Discord-hears-converted-voice
10. **Push 34 unpushed commits** to origin/main (do this as each item lands!)

---

## 7. Git state

- Branch `main`, HEAD = `50fe61b` "Tray: state icons, menu submenus, model state, device/runtime menus"
- **34 commits ahead of origin/main** — all tray/ggml work is unpushed
- Uncommitted: `docs/session-work.md` (M), `src/spartacvc.res` (M),
  `install_spectre.bat` + `install_wdk.bat` (untracked)

## 8. Build commands

```bat
cmd //c build_spcv.bat    :: CLI → spartacvc.exe
cmd //c build_tray.bat    :: Tray → spartacvc_tray.exe
:: Driver: see §4 retry loop (msbuild VirtualAudioDriver.sln Release x64)
```

Run: `spartacvc.exe input.wav --voice models/voices/nahida.gguf --backend cuda --out out.wav`

## 9. GPU status

| GPU | Backend | Perf (4.1s audio) |
|-----|---------|-------------------|
| Tesla PG500-216 (V100, sm_70, TCC, 32GB) | CUDA | ✅ 2.4s |
| Radeon Vega Frontier Edition (16GB) | Vulkan | ✅ 13.8s |
| CPU (48 cores) | ggml-cpu | ✅ works |
