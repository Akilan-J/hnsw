| latent (intrinsic) dimension | HNSW | flat + heuristic |
|---:|---:|---:|
| 4 | &le; 134 | &le; 221 |
| 8 | &le; 194 | &le; 256 |
| 16 | 303 | 721 |
| 32 | 465 | 892 |
| 64 | 576 | 927 |
| 128 | 653 | 1,257 |

distances per query needed to reach recall@10 = 0.95, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
