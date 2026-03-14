from dataclasses import dataclass, field


@dataclass
class ScorePipelineConfig:
    strategy_weights: dict[str, float] = field(default_factory=dict)