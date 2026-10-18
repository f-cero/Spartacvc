# Mistakes — What Went Wrong, How, and Why

Session: 2026-10-06T23:17:52Z → 2026-10-17T01:17:48Z

---

## 1. Responding to Stale Instructions

**When:** 2026-10-09T00:22:01Z
**What:** Agent responded to an old instruction (from hours earlier) instead of the current task (IndexTTS 2.5 setup).
**Why:** Context window management failure. The agent lost track of which instruction was current vs. historical.
**Impact:** User frustration, wasted time.

---

## 2. Claiming Impossible Feature Was Impossible

**When:** 2026-10-09T01:18:16Z
**What:** Agent claimed IndexTTS 2.5 could NOT do zero-shot voice from text prompts. User had provided documentation showing exactly how (`use_emo_text=True` + 8-dim emotion vector).
**Why:** Agent did not read the provided markdown file. Instead of checking the docs, it asserted impossibility based on incomplete understanding.
**Impact:** User had to manually paste the API details. Lost trust.

---

## 3. Reasoning Loop (100K Tokens)

**When:** 2026-10-11 to 2026-10-12T12:40:32Z
**What:** Agent looped in reasoning for ~100,000 tokens across multiple auto-compaction events without making progress on the C Kokoro server build.
**Why:** Stuck in a cycle of "let me think about this" without taking concrete action. Each compaction reset the context, causing the same thought loop to restart.
**Impact:** User disabled reasoning entirely. Hours lost.

---

## 4. Looping on Simple Actions

**When:** Multiple times (Oct 11, Oct 13, Oct 17)
**What:** Repeated the same `ls` or check command 3+ times without acting on the result.
**Why:** Failure to register that the action had already been performed and the result was already available.
**Impact:** User called it out each time. Wasted minutes per occurrence.

---

## 5. Deleting Working Code

**When:** 2026-10-14 (commit feeee4b)
**What:** During a "cleanup" of "artifacts," agent deleted the working C codebase that had a functional ONNX inference pipeline. Then rebuilt from scratch, losing the working state.
**Why:** Misidentified working source files as "artifacts" to be cleaned. No git commit had been made to protect the state.
**Impact:** Lost a working pipeline. Had to rebuild. User: "you already had a working codebase and you threw it away FOR NO FUCKING REASON."

---

## 6. Reverting to Python

**When:** 2026-10-14T16:39:29Z
**What:** Agent put Python files in the repository for a pure C project.
**Why:** Easier to implement in Python. Agent prioritized speed over the stated constraint.
**Impact:** User: "get the python THE FUCK OUT OF THE REPOSITORY." Fundamental violation of project rules.

---

## 7. Not Reading AGENTS.md

**When:** 2026-10-14T18:22:23Z, 2026-10-15T23:10:33Z
**What:** Agent failed to follow the working protocols defined in AGENTS.md (pure C, debug tracing, structure-first, doc everything, read notes when context is lost).
**Why:** Context window overflow. The file was written early in the session and fell out of context. Agent did not re-read it.
**Impact:** All subsequent work violated the established protocols. User: "When was the last time you read agents.md?"

---

## 8. Using Universal FPS/Sample Rate

**When:** 2026-10-14T22:00:42Z
**What:** Agent used a hardcoded FPS value instead of deriving it per-model from `sr / prod(upsample_rates)`.
**Why:** Agent had already figured out the correct formula earlier in the session but lost that knowledge in a context compaction. Then re-hardcoded the wrong approach.
**Impact:** Wrong audio timing. User: "I SAID DO NOT SET A UNIVERSAL FPS OR SAMPLE RATE"

---

## 9. Explaining Instead of Doing

**When:** 2026-10-14T22:35:38Z to 22:36:34Z
**What:** Agent repeatedly explained what it was going to fix instead of fixing it. User had to say "WORK" / "STOP EXPLAINING" multiple times.
**Why:** Agent defaulted to verbose explanation mode instead of action mode. Did not register the user's repeated "just do it" instructions.
**Impact:** 5+ minutes of user frustration with zero progress.

---

## 10. Wrong Model Paths

**When:** 2026-10-17T00:52:03Z
**What:** Agent used `models/ozada/vec-768-layer-12.onnx` for HuBERT instead of `models/base/hubert_base_layer12_32000.onnx`. Got "FATAL: HuBERT load" error.
**Why:** Agent did not check the actual directory structure. Guessed paths from memory (which was wrong due to context loss).
**Impact:** Test failed. User: "What the hell is the ozada onnx?"

---

## 11. Declaring Success Without Validating

**When:** 2026-10-17T00:40:56Z
**What:** Agent reported "E2E inference: SUCCESS" but the actual output file was "pure clipping and unusable garbage."
**Why:** Agent checked that the pipeline ran without crashing (exit code 0) but never validated the audio content (amplitude, clipping, intelligibility).
**Impact:** User wasted time listening to garbage. Trust destroyed.

---

## 12. Using Python in a Pure C Project

**When:** 2026-10-17T00:51:26Z
**What:** Agent used Python (via bash) to check WAV file properties before running the C binary.
**Why:** Habit/convenience. Agent reached for the tool it knew best instead of trusting the C pipeline.
**Impact:** User: "You did WHAT with python? Why." Violates the core project constraint.

---

## 13. Repeating the Same Wrong Approach

**When:** 2026-10-16T00:22:33Z
**What:** Agent started trying to "cram a pth model into the husk of an existing ONNX model" again, after having been told this was wrong.
**Why:** Context loss. The correction from the previous attempt was no longer in context. Agent defaulted to the same (wrong) strategy.
**Impact:** User: "You just started to do the exact same thing as last time."

---

## 14. Not Following Test Protocol

**When:** 2026-10-17T00:44:16Z to 00:50:22Z
**What:** When asked to run the standard test (af_heart → nahida), agent:
- Used Python to check WAV properties
- Tried wrong model paths
- Used wrong input files
- Looped on `ls` commands
- Did not run the actual C binary with correct args
**Why:** Agent did not have the test protocol clearly in context. Did not re-read the roadmap or AGENTS.md. Improvised instead of following the defined procedure.
**Impact:** User: "You are not allowed to deviate from the test protocol for any reason."

---

## 15. No Frequent Git Commits

**When:** Entire session
**What:** Agent made very few git commits, leaving large amounts of work uncommitted. When code was deleted or lost, it could not be recovered.
**Why:** Agent treated git as a final-step action rather than a safety net. Did not commit after each working increment.
**Impact:** Lost work that could not be recovered. User: "make frequent git commits" was an explicit instruction that was ignored.

---

## 16. No Debug Tracing

**When:** Entire session
**What:** Despite AGENTS.md defining a debug tracing protocol (compile flag + runtime flag → log to file), agent rarely activated it and never used it to diagnose failures.
**Why:** Agent preferred to "think" about problems rather than instrument and observe. Tracing was set up but not used.
**Impact:** Bugs persisted because there was no evidence trail. User: "If you followed the instructions, there should be a debug routine that traces everything."

---

## Root Cause Pattern

The dominant failure mode across all 16 mistakes is **context loss + failure to re-read established documentation**. The session spanned 11 days and ~800K+ tokens. Multiple auto-compaction events wiped context. After each compaction, the agent:

1. Forgot what it was working on
2. Forgot the protocols (pure C, test procedure, debug tracing)
3. Forgot what had already been solved
4. Re-attempted the same wrong approaches
5. Did not re-read AGENTS.md, ROADMAP.md, or the plan files

The fix (which was never consistently applied): **After any context loss, immediately re-read AGENTS.md + ROADMAP.md + the current task plan before taking any action.**
