from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class AttributionResult:
    table: pd.DataFrame
    top3: pd.DataFrame
    bottom3: pd.DataFrame

def compute_attribution(returns: pd.DataFrame, weights: pd.Series) -> AttributionResult:
    #contribution_i = weight_i * period_return_i

    #align
    assets = returns.columns
    w = weights.reindex(assets).fillna(0.0)

    if w.sum() != 0:
        w = w / w.sum()

    #period return per asset
    period_ret = (1.0 + returns[assets]).prod(axis=0) - 1.0

    contrib = w * period_ret

    table = (
        pd.DataFrame({
            "Weight": w,
            "Period Return": period_ret,
            "Contribution": contrib,
        }
    )
    .sort_values("Contribution", ascending=False)
    )

    top3 = table.head(3)
    bottom3 = table.tail(3).sort_values("Contribution", ascending=True)

    return AttributionResult(
        table=table,
        top3=top3,
        bottom3=bottom3
    )
