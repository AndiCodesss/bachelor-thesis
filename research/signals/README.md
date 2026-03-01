# Research Signal Contract

Every strategy in `research/signals/` must expose:

```python
def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:
    ...
```

## Rules

- Return shape must be `(len(df),)`.
- Values must be in `{-1, 0, 1}`.
- No `NaN`.
- Function must be pure (no hidden state).
- Use only current/past data in `df` (no lookahead).

## Optional Metadata

```python
STRATEGY_METADATA = {
    "name": "example",
    "version": "1.0",
    "features_required": ["close", "ema_8", "ema_21"],
    "description": "Brief hypothesis",
}
```

