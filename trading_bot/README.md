# Fail-fast evolutionary trading bot

A reinforcement-learning trading agent that learns entry and exit signals (short/flat/long). It is judged on held-out validation data during training. When it runs a **losing strategy** or breaches a **drawdown limit**, the current generation is **terminated**: its state is wiped and a new generation is spawned. This repeats until a generation reaches the target Sharpe or `max_generations` runs out.

```bash
pip install -r requirements.txt
python main.py                                  # synthetic data, full wipe on death
python main.py --csv prices.csv                 # your data (needs a `close` column)
python main.py --inherit elite --kill-drawdown 0.35
python main.py --help                           # every knob in evolve.Config
```

| file | role |
|---|---|
| `data.py` | CSV loader, synthetic regime-switching prices, chronological train/val/test split |
| `env.py` | features (no look-ahead), position simulation with costs, metrics |
| `agent.py` | linear softmax policy trained with policy gradient |
| `evolve.py` | generation loop: train → judge → terminate & wipe → respawn |
| `main.py` | CLI; writes `runs/generations.csv`, `champion.npz`, `result.json` |

## Kill rules (`evolve.Config`)
- `kill_drawdown`: terminate if the validation max drawdown is above this
- `kill_on_loss`: terminate if the validation net return is below zero
- `grace_episodes`: no kills before this many episodes, because an untrained agent always loses
- `inherit`: `scratch` = full wipe (the original design); `elite` = new generation starts from a mutated copy of the best survivor

## Read before trusting results
- **TEST is the only honest number.** The champion is selected on validation data from many generations, so its validation score is inflated by luck. `main.py` reports results on a test slice that no generation ever saw.
- Costs default to 5 bps per unit of position change. Set `--cost` to match your broker's fees and slippage.
- Paper-trade before going live.
