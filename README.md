# Index interview exercise

The sample preview loads the first finance example from PRBench and prints each
of its prompts in order.

From a clone, use Python 3.12 and uv:

```sh
uv sync --locked
uv run --locked pytest -k 'not Functional and not CodexMultiturnMemory'
```

This test selection uses local fixtures and mocks. The other tests exercise
Hugging Face or the installed Codex CLI.

Run the sample preview with `uv run --locked python main.py`. When the local
cache is missing, this command downloads the dataset before printing the prompts.
