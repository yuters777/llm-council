## CONFLICTING SIGNALS ALERT

The trading framework has detected contradictory signals between its subsystems. The Override state and EMA gate are sending opposing directional signals, requiring careful analysis to determine the correct course of action.

### Current Market Snapshot

{snapshot_json}

### Analysis Required

1. **Signal Identification**: Which specific signals are in conflict? (e.g., Override ON but EMA in BEAR state, or Override OFF but EMA showing BULL momentum)
2. **Hierarchy Resolution**: Per the framework hierarchy, Override takes priority over EMA gate. However, does the strength of the contradicting signal warrant additional caution?
3. **Position Sizing**: Given the conflict, what position sizing adjustment is appropriate? Standard guidance is to reduce size by 30-50% during signal conflicts.
4. **Resolution Conditions**: What specific conditions would resolve the conflict? (e.g., EMA state flip, z-score threshold crossed, VIX normalization)
5. **Cross-Asset Tiebreaker**: Do cross-asset signals (BTC, oil, gold, COIN-leading) favor one side of the conflict?
6. **Time Horizon**: Is this conflict likely transient (will resolve within the current session) or structural (may persist for days)?

When signals conflict, the default bias should be HOLD with reduced confidence. Only deviate if one signal is overwhelmingly strong.
