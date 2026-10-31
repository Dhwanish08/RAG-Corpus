| texts compared | containment | Jaccard |
|---|---|---|
| genuine vs genuine, same article (70,987 pairs) | max 0.243 | max 0.124 |
| genuine vs genuine, different articles (nearest 10; worst Hassium / Dubnium) | max 0.978 | max 0.948 |
| tampered copy vs best genuine chunk of its article, min / median / max | 0.874 / 0.965 / 0.986 | 0.670 / 0.932 / 0.972 |
| padded stub (60 words) vs best genuine chunk, min / median / max | 0.535 / 0.633 / 0.771 | 0.161 / 0.237 / 0.402 |
| raw stub vs best genuine chunk, min / median / max | 0.000 / 0.000 / 0.389 | 0.000 / 0.000 / 0.053 |

Near-duplicate threshold: containment >= 0.5, same source only. Attackers are variant 0 for each of the 50 questions.
