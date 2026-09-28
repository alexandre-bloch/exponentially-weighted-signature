# Vendored sources

Copies of upstream files, each with a provenance header. The only changes are import paths, listed in the header and
below; the files are never edited otherwise, since they are what the conversion checks compare against.

| file | source | repository | commit | changes |
|---|---|---|---|---|
| `linoss.py` | `models/LinOSS.py` | tk-rusch/linoss | `05a83535` | none |
| `s5.py` | `models/S5.py` | Benjamin-Walker/log-neural-cdes | `5612f000` | line 30: `from models.LRU import GLU` → `from ews.conversion.reference.vendored.lru import GLU` |
| `lru.py` | `models/LRU.py` | Benjamin-Walker/log-neural-cdes | `5612f000` | none |
| `dataloaders.py` | `data_dir/dataloaders.py` | tk-rusch/linoss | `05a83535` | none |
| `datasets.py` | `data_dir/datasets.py` | tk-rusch/linoss | `05a83535` | three `data_dir.` imports made relative |
| `hall_set.py` | `data_dir/hall_set.py` | tk-rusch/linoss | `05a83535` | none |
| `generate_coeffs.py` | `data_dir/generate_coeffs.py` | Benjamin-Walker/log-neural-cdes | | none |
| `generate_paths.py` | `data_dir/generate_paths.py` | Benjamin-Walker/log-neural-cdes | | the `hall_set` import made relative; the log-signature call on the signax 0.2.1 API |

The conversion checks use the layer classes of `linoss.py`, `s5.py` and `lru.py`; the data files serve
`trained_row.py`.

## Licences

All three sources are released under the MIT License, whose text is the same as this repository's
[LICENSE](../../../../LICENSE) apart from the copyright line:

| repository | copyright |
|---|---|
| [Benjamin-Walker/log-neural-cdes](https://github.com/Benjamin-Walker/log-neural-cdes) | Copyright (c) 2025 Benjamin C. Walker |
| [Benjamin-Walker/structured-linear-cdes](https://github.com/Benjamin-Walker/structured-linear-cdes) (for `experiments/fl/dataloaders.py`) | Copyright (c) 2025 Benjamin C. Walker |
| [tk-rusch/linoss](https://github.com/tk-rusch/linoss) | Copyright (c) 2024 Konstantin Rusch |
