# AGENTS.md

## Project
AI Talking Photo

## Source of truth
Read in this order:
1. `AGENTS.md`
2. `docs/TASKS.md`
3. `docs/SPEC.md` only when needed

## Workflow
Work only on the Current task from `docs/TASKS.md`.
Before marking a task complete:
1. Implement the task.
2. Run relevant tests.
3. Fix failures.
4. Update docs if behavior changed.
5. Update `docs/TASKS.md`.

Do not fake integrations. Do not commit model weights, private photos, generated videos, secrets, or temp files.

## Product language
All end-user UI text must use Traditional Chinese with Taiwan terminology.

## Hardware
Primary development constraint: NVIDIA GTX 1050 2GB.
Google Colab GPU is the preferred inference environment.
Local CPU / low-VRAM CUDA is fallback only.
Future target: RTX 3060 12GB.
