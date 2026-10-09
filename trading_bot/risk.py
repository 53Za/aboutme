"""Risk rules enforced in code, outside the model's control (idea from NoFxAiOS/nofx).

The model proposes a position; RiskGuard decides what is actually held. The same guard runs
in backtests (evolve.evaluate) and in live alerts (signals.py), so tests match live behaviour.
Prices are checked at candle close; there is no intrabar stop.
"""
from dataclasses import dataclass, asdict


@dataclass
class RiskRules:
    stop_loss: float = 0.03       # close if the position is down 3%
    protect_after: float = 0.05   # once up 5% at peak...
    giveback: float = 0.40        # ...close if 40% of that peak profit is given back
    min_hold_bars: int = 2        # model may not exit/flip before this many candles
    cooldown_bars: int = 4        # no new entry for this many candles after any close


class RiskGuard:
    def __init__(self, rules: RiskRules, state: dict | None = None):
        self.r = rules
        s = state or {}
        self.pos = s.get("pos", 0)
        self.entry = s.get("entry")
        self.peak_gain = s.get("peak_gain", 0.0)
        self.held = s.get("held", 0)
        self.cool = s.get("cool", 0)

    def state(self) -> dict:
        return {"pos": self.pos, "entry": self.entry, "peak_gain": self.peak_gain,
                "held": self.held, "cool": self.cool}

    def _set(self, new_pos: int, price: float):
        if self.pos != 0 and new_pos != self.pos:
            self.cool = self.r.cooldown_bars
        if new_pos != 0 and new_pos != self.pos:
            self.entry, self.peak_gain, self.held = price, 0.0, 0
        if new_pos == 0:
            self.entry, self.peak_gain, self.held = None, 0.0, 0
        self.pos = new_pos

    def force_close(self, price: float):
        """Close outside the candle loop (e.g. on news); starts the re-entry cooldown."""
        self._set(0, price)

    def step(self, desired: int, price: float) -> tuple[int, str | None]:
        """Call once per closed candle. Returns (position to hold, reason if a rule overrode the model)."""
        r = self.r
        if self.pos != 0:
            self.held += 1
            gain = (price / self.entry - 1.0) * self.pos
            self.peak_gain = max(self.peak_gain, gain)
            if gain <= -r.stop_loss:
                self._set(0, price)
                return 0, f"stop-loss hit ({gain:+.1%})"
            if self.peak_gain >= r.protect_after and self.peak_gain - gain >= r.giveback * self.peak_gain:
                peak = self.peak_gain
                self._set(0, price)
                return 0, f"profit protection (peak {peak:+.1%}, now {gain:+.1%})"
            if desired != self.pos and self.held < r.min_hold_bars:
                return self.pos, None
        elif self.cool > 0:
            self.cool -= 1
            return 0, None
        if desired != self.pos and self.pos != 0 and desired != 0:
            desired = 0  # a flip closes first; the cooldown then delays the new entry
        self._set(desired, price)
        return desired, None


def rules_dict(rules: RiskRules) -> dict:
    return asdict(rules)
