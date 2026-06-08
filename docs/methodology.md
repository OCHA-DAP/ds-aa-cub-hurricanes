# Cuba hurricane trigger — old vs new

Two indicators, AND/OR logic across forecast and observational arms —
replaced by a single population-exposure threshold.

```mermaid
flowchart LR
    subgraph OLD [" "]
        direction TB
        OLD_T["<b>OLD METHOD</b>"]
        F["<b>Forecast</b><br/>ZMA wind ≥ 120 kt"]
        O["<b>Observational</b><br/>ZMA wind ≥ 105 kt<br/>+ IMERG q80 ≥ 96.2 mm"]
        OR(("OR"))
        OLD_TRIG["Trigger fires"]
        OLD_T ~~~ F
        OLD_T ~~~ O
        F --> OR
        O --> OR
        OR --> OLD_TRIG
    end

    subgraph NEW [" "]
        direction TB
        NEW_T["<b>NEW METHOD</b>"]
        EXP["<b>64 kt exposure</b><br/>≥ 616,778 people"]
        NEW_TRIG["Trigger fires"]
        NEW_T ~~~ EXP
        EXP --> NEW_TRIG
    end

    classDef title fill:none,stroke:none,color:#37474f,font-size:18px
    classDef arm fill:#fff8e1,stroke:#f57c00,stroke-width:2px,color:#5d4037
    classDef new fill:#e0f2f1,stroke:#00796b,stroke-width:3px,color:#004d40
    classDef gate fill:#fff,stroke:#f57c00,stroke-width:2px,color:#5d4037
    classDef trigOld fill:#5d4037,stroke:#3e2723,stroke-width:2px,color:#fff
    classDef trigNew fill:#00796b,stroke:#004d40,stroke-width:2px,color:#fff

    class OLD_T,NEW_T title
    class F,O arm
    class EXP new
    class OR gate
    class OLD_TRIG trigOld
    class NEW_TRIG trigNew

    style OLD fill:#fafafa,stroke:#e0e0e0,stroke-width:1px
    style NEW fill:#fafafa,stroke:#e0e0e0,stroke-width:1px
```

Same n = 10 storms triggered over 2002–2025 (RP ≈ 2.6 yrs), same
6 CERF-funded storms caught — but the new method is a single
threshold against one indicator, rather than two AND-gated arms
combined by OR.

---

## How the 64 kt exposure number is computed

For each storm, "people exposed to 64 kt winds" is computed by
overlaying the hurricane's wind footprint on a population map.
Two streams contribute and are combined at each NHC advisory.

```mermaid
flowchart LR
    %% Inputs
    OBS(["<b>Storm track</b><br/>where it has been<br/><i>(observed positions)</i>"])
    FCT(["<b>Storm track</b><br/>where it's going<br/><i>(forecast positions)</i>"])
    POP[("<b>Population<br/>map</b>")]

    %% Per-stream processing
    XO["intersect with<br/>64-kt wind buffers"]
    XF["intersect with<br/>64-kt wind buffers"]
    OBS --> XO
    FCT --> XF
    POP -.-> XO
    POP -.-> XF

    %% Stream outputs
    OE(["<b>Observed exposure</b><br/>cumulative people<br/>already affected"])
    FE(["<b>Forecast exposure</b><br/>people projected<br/>to be affected"])
    XO --> OE
    XF --> FE

    %% Combine
    OE --> SUM(("&plus;"))
    FE --> SUM
    SUM --> TOT(["<b>Total exposure</b><br/>recomputed every<br/>NHC advisory"])
    TOT --> PEAK(["<b>Peak total</b><br/>maximum across all<br/>advisories during the storm"])

    %% Decision
    PEAK --> THR{{"≥ 616,778 people?"}}
    THR -->|yes| FIRE["Trigger fires"]

    classDef input fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#01579b
    classDef proc fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,color:#424242,font-style:italic
    classDef result fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#004d40
    classDef gate fill:#fff,stroke:#00796b,stroke-width:2px,color:#004d40
    classDef decide fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#bf360c
    classDef fire fill:#00796b,stroke:#004d40,stroke-width:2px,color:#fff

    class OBS,FCT,POP input
    class XO,XF proc
    class OE,FE,TOT,PEAK result
    class SUM gate
    class THR decide
    class FIRE fire
```

**Two important details:**

- *Observed exposure is **cumulative**.* Once a populated area has been
  inside the 64-kt wind buffer, those people stay counted for the rest
  of the storm — even after the storm has moved on. This makes the
  observed series monotone non-decreasing through time.
- *Forecast exposure is **forward-only**.* It counts people in the
  wind buffers along the forecast track from now onwards, not back to
  storm genesis. That avoids double-counting the people already in the
  observed series.

The trigger metric is the **peak total exposure** observed at any
single NHC advisory during the storm's lifetime — so a storm that
ramps up gradually and a storm that strengthens suddenly can both
trigger if their peak combined exposure crosses the threshold at any
moment.
