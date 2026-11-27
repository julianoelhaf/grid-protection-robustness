# Release procedure

GitHub is the source of truth for this clean-history project. Use a new FAU
GitLab mirror that can accept the exact GitHub release commit by fast-forward.
The existing GitLab project has divergent work. Preserve it as a separate
archive or agree on a migration before reconfiguring it.

Historical refs belong to their original repositories.

| Location | Ref | Target |
| --- | --- | --- |
| GitHub legacy project | `eusipco-arxiv-v2` | `1f6eedb83ced6cf713a41d863e3f8080afe9ef1e` |
| GitHub legacy project | `v1.0.0` | Existing tag target |
| FAU GitLab legacy project | `v0.1.0` | Existing tag target |

The EUSIPCO benchmark records the original source baseline, executed
runner checksum, audited tables, and environment. Record the new package commit
with the wheel and evidence supplement checksums at release. The archived run
predates the package commit.

Privately review the clean-history source and evidence supplement. Check
dataset and supplement redistribution and credits before publication. Use
`v2.0.0` for EUSIPCO and a study-specific tag for future DPSP releases. Before
mirroring, compare the exact release tree and commit on both remotes with
`git ls-remote`. Preserve existing refs and use a fast-forward mirror. Remote
renames, pushes, and publication require Julian's release authorization.

- [Legacy GitHub repository](https://github.com/julianoelhaf/eusipco2025-sparse-fault-detect)
- [Legacy FAU GitLab repository](https://git5.cs.fau.de/juoelhaf/sparse_fault_detect)
- [EUSIPCO arXiv v2](https://arxiv.org/abs/2505.15560v2)
