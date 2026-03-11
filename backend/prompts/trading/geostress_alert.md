## GEOSTRESS ALERT

The GeoStress monitor has detected elevated cross-asset stress signals. When GeoStress is active, the framework mandates trading SUSPENSION or significant risk reduction.

### Current Market Snapshot

{snapshot_json}

### Analysis Required

1. **Stress Assessment**: Review the GeoStress score and individual components (z_dvix_5, z_dvvix_5, z_gold_5, z_oil_5_abs, z_jpy_dxy, breadth_shock). Which components are driving the alert?
2. **Severity Classification**: Is this a minor stress spike (score 3.0-4.0), moderate (4.0-6.0), or severe (6.0+)?
3. **Cross-Asset Confirmation**: Do gold, oil, and JPY/DXY moves confirm a genuine risk-off event, or is this a single-asset anomaly?
4. **VIX/VVIX Dynamics**: Are VIX and VVIX both elevated? Is the VIX term structure inverted (indicating near-term panic)?
5. **Position Action**: Should all positions be exited immediately, or can some be held with reduced sizing?
6. **Duration Estimate**: Based on the stress components, is this likely a flash event (minutes) or sustained (hours/days)?

When GeoStress is active, the default decision should be HOLD or EXIT. BUY decisions require extraordinary justification.
