from dataclasses import dataclass


@dataclass
class Counters:

    global_backups: int = 0 
    aggregate_backups: int = 0 
    rebin_ops: int = 0 
    lift_ops: int = 0 

    @property
    def billed(self) -> int:
        return self.global_backups + self.aggregate_backups

    @property
    def actual(self) -> int:
        return self.billed + self.rebin_ops + self.lift_ops

    @property
    def overhead_fraction(self) -> float:
        if self.billed == 0:
            return 0.0
        return (self.actual - self.billed) / self.billed


# slots, so a mistyped field name in the loop's setattr raises rather than
# quietly creating a fifth bucket that nothing ever reads.
@dataclass(slots=True)
class PhaseTimes:

    global_ns: int = 0
    aggregate_ns: int = 0
    rebin_ns: int = 0
    lift_ns: int = 0

    @property
    def total_ns(self) -> int:
        return self.global_ns + self.aggregate_ns + self.rebin_ns + self.lift_ns

    def share(self) -> dict[str, float]:
        total = self.total_ns
        if total == 0:
            return {"global": 0.0, "aggregate": 0.0, "rebin": 0.0, "lift": 0.0}

        return {
            "global": self.global_ns / total,
            "aggregate": self.aggregate_ns / total,
            "rebin": self.rebin_ns / total,
            "lift": self.lift_ns / total,
        }

