# Cuba hurricane trigger — methodology

End-to-end flow from raw data sources to trigger candidates, as
implemented in `exploration/wsp_trigger.py`.

## Overview

```mermaid
flowchart LR
    %% ── Data sources ─────────────────────────────────────────────────
    subgraph SOURCES["Data sources (Postgres + blob)"]
        OBS["storms.nhc_tracks_obsv_exposure<br/><i>cumulative observed</i>"]
        FCO["storms.nhc_tracks_fcastonly_exposure<br/><i>forecast-only, no t=0 obs snapshot</i>"]
        IBT["storms.ibtracs_wind_exposure<br/><i>historical fallback</i>"]
        RAIN["fcast_obsv_combined_stats.parquet<br/><i>IMERG q80_obsv</i>"]
        IMP["emdat_cerf_upto2024.parquet<br/><i>impact + CERF labels</i>"]
    end

    %% ── Per-storm metrics ────────────────────────────────────────────
    subgraph METRICS["Per-storm metrics (64 kt)"]
        OBS64["<b>final_obsv_64</b><br/>last cumul. observed value"]
        FC64["<b>max_fcast_64</b><br/>max over forecast issuances"]
        MT64["<b>max_total_64</b><br/>max over issuances of<br/>(fcastonly_T + cumul_obs_at_T)"]
        Q80["<b>q80_obsv</b><br/>IMERG 2-day"]
        OLD_F["<b>Old fcast_trig</b><br/>ZMA wind ≥ 120"]
        OLD_O["<b>Old obsv_trig</b><br/>ZMA wind ≥ 105 AND q80 ≥ 96.2"]
    end

    OBS --> OBS64
    OBS --> MT64
    FCO --> FC64
    FCO --> MT64
    IBT --> OBS64
    IBT --> FC64
    RAIN --> Q80
    RAIN -.->|legacy| OLD_F
    RAIN -.->|legacy| OLD_O

    %% ── Trigger formulations ─────────────────────────────────────────
    subgraph TRIGS["Trigger formulations &nbsp;<i>(all locked to n=10 storms, RP ≈ 2.6 yrs)</i>"]
        T_OLD["<b>Old option 1b</b><br/>fcast_trig <i>OR</i> obsv_trig"]
        T_64EXP["<b>64 exp</b> &nbsp;<i>proposed</i><br/>max_total_64 ≥ 616,778"]
        T_2DOBS["<b>2-dim, obs-dominant</b><br/>obs ≥ 229,427<br/><i>OR</i> fcast ≥ 3,957,370"]
        T_2DFC["<b>2-dim, fcast-dominant</b><br/>obs ≥ 1,123,760<br/><i>OR</i> fcast ≥ 616,778"]
    end

    OLD_F --> T_OLD
    OLD_O --> T_OLD
    MT64 --> T_64EXP
    OBS64 --> T_2DOBS
    FC64 --> T_2DOBS
    OBS64 --> T_2DFC
    FC64 --> T_2DFC

    %% ── Evaluation ───────────────────────────────────────────────────
    subgraph EVAL["Evaluation criterion"]
        CRIT["1. max <b>CERF</b> storms caught (of 9)<br/>2. then max Total Affected<br/>3. then min threshold value"]
    end

    IMP --> CRIT
    T_OLD --> CRIT
    T_64EXP --> CRIT
    T_2DOBS --> CRIT
    T_2DFC --> CRIT

    %% ── Styling ──────────────────────────────────────────────────────
    classDef src fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef met fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef trig fill:#f3e5f5,stroke:#8e24aa,color:#4a148c
    classDef eval fill:#e8f5e9,stroke:#43a047,color:#1b5e20
    class OBS,FCO,IBT,RAIN,IMP src
    class OBS64,FC64,MT64,Q80,OLD_F,OLD_O met
    class T_OLD,T_64EXP,T_2DOBS,T_2DFC trig
    class CRIT eval
```

## The n = 10 lock

Every trigger is calibrated to fire on exactly **10 storms** over the
2002–2025 window — a return period of about 2.6 years. With that count
fixed, the optimization searches the threshold space for the
combination that catches the most CERF-funded storms, then maximises
Total Affected (EM-DAT), then takes the lowest (most sensitive)
threshold value.

For triggers with two arms (Old option 1b, both 2-dim variants), the
optimiser sweeps one threshold and sets the other deterministically to
exactly fill the remaining n-quota among storms not already triggered.

## Two methodology notes worth highlighting

### Observed exposure is cumulative-to-date

`storms.nhc_tracks_obsv_exposure.pop_exposed` at each `valid_time` is
the **union** of all wind-radii buffers from storm genesis up to that
time, intersected with population. Verified empirically: across Melissa
2025 and Ian 2022, all wind levels, 50+ data points — zero
non-monotone steps. So `final_obsv_64` = `max(cumul_obs)` = the last
recorded value, and `cummax()` in the code is a defensive no-op rather
than a correction.

### Forecast-only, not forecast-full

`storms.nhc_tracks_fcast_exposure` (the "fcast-full" table) bundles a
snapshot of the **current observed position** into each forecast row.
Summing `fcast_full(T) + cumul_obs(T)` would then double-count that
buffer — once in the forecast, once in the cumulative observed series
that already contains it.

`storms.nhc_tracks_fcastonly_exposure` strips that snapshot. All trigger
computations here use `fcastonly`, matching the convention in the live
alerts pipeline (`monorepo-hurricane-monitoring/ds-storms-alerts`).

Empirical impact of the swap: peak `max_total_64` values drop by 0–80%
across historical storms. For the proposed `64 exp` trigger
specifically, the n=10 threshold moves 697,573 → 616,778, and the
composition of the 10 triggered storms shifts (Rafael 2024 swaps in for
Ian 2022; both CERF).

## Where each piece lives in the notebook

| Block | Cells in `exploration/wsp_trigger.py` |
|---|---|
| Data sources | `load_obsv_exposure`, `load_total_exposure`, `load_storm_meta`, `load_impact`, `load_old_trigger`, `load_max_fcast` |
| Per-storm metrics | `trigger_table` (display), `load_total_exposure` (compute), `load_max_fcast` (compute) |
| `Old option 1b` trigger | `load_old_trigger` |
| `64 exp` trigger optimisation | `rain_trigger_opt` (the `64x exp only` row of the Summary) |
| `2-dim` optimisations | `two_d_64kt_opt` (obs-dominant) and `two_d_64kt_opt_fcast_dom` (fcast-dominant) |
| Head-to-head comparisons | `compare_old_vs_new`, `compare_proposals` |

See [`../review/01_review.md`](../review/01_review.md) for the deeper
code review notes and empirical verification details (review file is
gitignored — local only).
