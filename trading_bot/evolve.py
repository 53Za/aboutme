"""Fail-fast generational loop: train, judge, kill & wipe, respawn."""
from dataclasses import dataclass, asdict

import numpy as np

from agent import Agent
from env import run, metrics


@dataclass
class Config:
    max_generations: int = 50
    train_episodes: int = 300        # episodes a generation gets before it is judged "complete"
    episode_len: int = 250
    grace_episodes: int = 60         # no kills before this; an untrained agent always loses
    check_every: int = 20            # judge on validation data every N episodes
    kill_drawdown: float = 0.15      # kill if validation max drawdown exceeds this
    kill_on_loss: bool = True        # kill if validation net return < 0 ("losing strategy")
    target_sharpe: float = 1.0       # a surviving generation at/above this ends evolution
    cost: float = 0.0005             # per unit of position change (fees + slippage)
    lr: float = 0.01
    inherit: str = "scratch"         # "scratch" = full wipe; "elite" = mutate best survivor
    mutation: float = 0.02
    seed: int = 0


def evaluate(agent, X, rets, cost):
    return metrics(*run(lambda f, p: agent.act(f, p, greedy=True), X, rets, cost))


def death_cause(m, cfg):
    if m["max_drawdown"] > cfg.kill_drawdown:
        return f"drawdown {m['max_drawdown']:.1%} > {cfg.kill_drawdown:.0%}"
    if cfg.kill_on_loss and m["return"] < 0:
        return f"losing strategy ({m['return']:+.1%})"
    return None


def evolve(X_train, r_train, X_val, r_val, cfg: Config, log=print):
    rng = np.random.default_rng(cfg.seed)
    champion, champion_m, history = None, None, []
    n_feat = X_train.shape[1]
    max_start = len(r_train) - cfg.episode_len

    for gen in range(1, cfg.max_generations + 1):
        # Spawn. "scratch" honours the full-wipe rule; "elite" carries the best survivor forward.
        seed_w = None
        if cfg.inherit == "elite" and champion is not None:
            seed_w = champion.W + rng.normal(0, cfg.mutation, champion.W.shape)
        agent = Agent(n_feat, np.random.default_rng(rng.integers(1 << 32)), cfg.lr, seed_w)

        cause, m = None, None
        for ep in range(1, cfg.train_episodes + 1):
            start = int(rng.integers(90, max_start))
            agent.train_episode(X_train, r_train, cfg.cost, start, cfg.episode_len)
            if ep >= cfg.grace_episodes and ep % cfg.check_every == 0:
                m = evaluate(agent, X_val, r_val, cfg.cost)
                cause = death_cause(m, cfg)
                if cause:
                    break
        m = m or evaluate(agent, X_val, r_val, cfg.cost)

        status = f"TERMINATED at ep {ep}: {cause}" if cause else "SURVIVED"
        log(f"gen {gen:3d} | val ret {m['return']:+7.1%} dd {m['max_drawdown']:6.1%} "
            f"sharpe {m['sharpe']:5.2f} | {status}")
        history.append({"generation": gen, "survived": cause is None,
                        "died_at_episode": ep if cause else None, "cause": cause, **m})

        if cause:
            del agent  # self-termination: state is discarded, nothing carries over
            continue
        if champion_m is None or m["sharpe"] > champion_m["sharpe"]:
            champion, champion_m = agent, m
        if m["sharpe"] >= cfg.target_sharpe:
            log(f"gen {gen} reached target Sharpe {cfg.target_sharpe}; stopping.")
            break

    return champion, champion_m, history, asdict(cfg)
