# GPU Vector Search

## Build and run on Euler

TODO: document the Euler modules, build commands, dataset setup, and benchmark invocation.

## Versions

| Version | Implementation | Status |
| --- | --- | --- |
| v1 | Serial CPU | TODO |
| v2 | OpenMP CPU | TODO |
| v3 | Naive CUDA | In Progress |
| v4 | Tiled CUDA | In Progress |
| v5 | Fused CUDA | TODO |
| v6 | WMMA CUDA | TODO |

## Results

TODO: add the benchmark results and final plots.



## Playground

The `zplayground/` directory is intended for individual learning, experimentation, and testing without affecting the main project implementations. The `z` prefix is simply to keep the directory at the bottom of the repository listing.

**Commits containing only playground changes should begin with `Learning:`**

I hope this directory encourages us to experiment freely, explore ideas, and use it as a scratchpad without worrying about writing polished code.

Since the ideas and experiments developed along the way are valuable to the project, and our evaluation also considers commit history, I believe maintaining this directory will help document our individual learning and contributions.

### Directory Structure

Each team member should create a personal subdirectory.

Experiments can be organized into daily folders using the `YYYY-MM-DD` format, as I have done in my directory. However, **feel free to organize your playground in whatever way feels most natural to you.**

```text
zplayground/
├── member1/
│   ├── 2026-10-21/
│   │   └── experiment.ipynb
│   └── 2026-10-22/
│       └── test.cpp
├── member2/
└── member3/
```

### Guidelines

1. **Personal workspace:** Work within your own directory and avoid modifying other members' experiments.

2. **Commit convention:** Commits containing only playground changes should begin with `Learning:`.

   * Example: `Learning: Exploring SIFT1M dataset structure`

3. **Experiment freely:** Code in the playground does not need to be optimized, polished, or production-ready. The purpose is to encourage exploration and learning.

4. **Integration:** Successful experiments can be moved into the appropriate main project directories after team review.

5. **Data management:** Avoid committing large datasets, compiled binaries, or unnecessary temporary files.

6. **Jupyter notebooks:** Use `nbstripout` to prevent notebook outputs from being committed.

**Note:** The playground is a shared learning space and is not part of the official benchmarking pipeline.
