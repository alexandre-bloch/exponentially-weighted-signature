# Experiments

One folder per experiment family, each with a config-driven trainer.

| family | data | model | results |
|---|---|---|---|
| `uea/` | six UEA classification datasets | one EWS layer | `uea_results/` |
| `fl/` | four formal-language tasks | stacked EWS blocks | `formal_language_results/` |
| `sde/` | synthetic regression: the expressivity study, the coupled oscillator and the oscillatory kernel | one EWS layer | `sde_results/` |
| `weather/` | Informer weather, 720 steps ahead | stacked EWS blocks | `weather_results/` |
| `ppg/` | PPG-DaLiA heart rate | stacked EWS blocks | `ppg_results/` |

`regression_common.py` holds the model, optimiser and batching of the `sde` trainer.

```bash
python experiments/uea/train.py --config uea_results/eigenworms/config.json --seed 0 --dataset_file <split.npz> --gpu 0
python experiments/fl/train.py --config formal_language_results/parity/config.json --run 0 --gpu 0
python experiments/sde/train.py --config sde_results/oscillator/ews/config.json --seed 100 --gpu 0
python experiments/weather/train.py --config weather_results/config.json --seed 2345 --gpu 0
python experiments/ppg/train.py --config ppg_results/config.json --seed 2345 --gpu 0
```

Every trainer takes `--config`, `--gpu` and the run identity (`--seed`, or `--run` for `fl`). All but `ppg` also take
the two XLA switches `--autotune {on,off}` and `--deterministic-ops {on,off}` (see [docs/training.md](../docs/training.md));
the PPG trainer runs with autotuning off. The data comes from the commands in [data/README.md](../data/README.md). The
config keys and their defaults are in [docs/configs.md](../docs/configs.md); every banked `config.json` is a complete
example, and `uea`, `fl` and `sde` each have a `configs/template.json` to start from.
