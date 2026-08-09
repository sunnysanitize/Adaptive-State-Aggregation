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

