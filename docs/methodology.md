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
