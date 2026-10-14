# hf CLI — quick reference

Binary: `hf` (on PATH) — installed at `C:\Users\fluffy\.local\bin\hf.exe`
Auth: logged in as **fluffendiferous**
Installed via: `powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"`
Self-update: `hf update`

## Download

```bash
# Specific files (preferred — don't pull whole repos)
hf download ozada/onnx_rvc rmvpe.onnx vec-768-layer-12.onnx --local-dir ./models/ozada

# Glob patterns
hf download ozada/onnx_rvc "*.onnx" --local-dir ./models/ozada

# File in a subfolder
hf download ArkanDash/rvc-genshin-impact "prezipped/v2/nahida-jp 102 epochs 48k v2.zip" --local-dir ./downloads

# Dataset
hf download --type dataset some-org/some-dataset --local-dir ./data

# Dry run (see what would download without doing it)
hf download ozada/onnx_rvc --dry-run
```

## List repo files (no download)

```bash
hf repo files ozada/onnx_rvc
hf repo files ArkanDash/rvc-genshin-impact
```

## Copy / move

```bash
# Local → local
hf cp ./src.wav ./dst.wav

# Local → Hub repo
hf cp ./model.onnx my-user/my-repo/model.onnx

# Hub → local
hf cp ozada/onnx_rvc/rmvpe.onnx ./rmvpe.onnx
```

## Upload

```bash
hf upload my-user/my-repo ./local-folder/ --commit-message "add models"
```

## Cache management

```bash
hf cache list                    # see what's cached + sizes
hf cache prune --dry-run         # preview cleanup
hf cache prune --yes             # actually clean up
hf cache rm ozada/onnx_rvc      # remove one repo from cache
```

Default cache: `C:\Users\fluffy\.cache\huggingface\hub\`

## Auth

```bash
hf auth whoami        # current user
hf auth login         # re-login / switch token
hf auth token         # print token to stdout
hf auth logout
```

## Be a good citizen

- **Download only what you need** — use file filters, not whole repos
- **Don't re-download** — hf skips complete files by default
- **No tight-loop hammering** — space out bulk downloads
- **Large repos**: prefer individual files over `--include "*"`
- Authenticated requests get higher rate limits; keep the token active

## Uninstall

```bash
rm -rf C:\Users\fluffy\.hf-cli
rm C:\Users\fluffy\.local\bin\hf.exe
```
