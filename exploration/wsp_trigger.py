import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def imports():
    import geopandas as gpd
    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import ocha_stratus as stratus
    import pandas as pd
    from sqlalchemy import text

    from src.constants import PROJECT_PREFIX
    from src.datasources.codab import load_codab_from_blob

    return (
        PROJECT_PREFIX,
        gpd,
        load_codab_from_blob,
        mo,
        mpatches,
        pd,
        plt,
        stratus,
        text,
    )


@app.cell
def load_obsv_exposure(pd, stratus, text):
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        # ibtracs_wind_exposure: full historical coverage
        _df_ibtracs = pd.read_sql(
            text(
                """
                SELECT sid, wind_speed_kt, pop_exposed
                FROM storms.ibtracs_wind_exposure
                WHERE iso3 = 'CUB' AND admin_level = 0
            """
            ),
            _conn,
        )
        # nhc_tracks_obsv_exposure: recent storms (fills gaps / Melissa 2025)
        _df_nhc = pd.read_sql(
            text(
                """
                SELECT atcf_id, valid_time, wind_speed_kt, pop_exposed
                FROM storms.nhc_tracks_obsv_exposure
                WHERE iso3 = 'CUB' AND admin_level = 0
            """
            ),
            _conn,
        )
        _df_sid = pd.read_sql(
            text("SELECT sid, atcf_id FROM storms.ibtracs_storms"),
            _conn,
        )
    _engine.dispose()

    # Convert nhc table to (sid, wind_speed_kt, pop_exposed)
    # keeping only the last valid_time per storm
    if not _df_nhc.empty:
        _df_nhc = _df_nhc.merge(_df_sid, on="atcf_id", how="left").dropna(
            subset=["sid"]
        )
        _df_nhc = (
            _df_nhc.sort_values("valid_time")
            .groupby(["sid", "wind_speed_kt"], sort=False)
            .last()
            .reset_index()[["sid", "wind_speed_kt", "pop_exposed"]]
        )

    # Union: nhc takes precedence, ibtracs fills historical gaps
    _nhc_sids = set(_df_nhc["sid"].unique()) if not _df_nhc.empty else set()
    _df_ibtracs_fill = _df_ibtracs[~_df_ibtracs["sid"].isin(_nhc_sids)]
    df_exp_raw = pd.concat(
        (
            [_df_nhc, _df_ibtracs_fill]
            if not _df_nhc.empty
            else [_df_ibtracs_fill]
        ),
        ignore_index=True,
    )

    # Restrict to 2002 onwards (sid starts with year)
    df_exp_raw = df_exp_raw[df_exp_raw["sid"].str[:4].astype(int) >= 2002]

    return (df_exp_raw,)


@app.cell
def load_storm_meta(df_exp_raw, pd, stratus, text):
    _sids = df_exp_raw["sid"].unique().tolist()
    _placeholders = ", ".join(f"'{s}'" for s in _sids)
    _engine = stratus.get_engine(stage="prod")
    with _engine.connect() as _conn:
        _df_meta = pd.read_sql(
            text(
                "SELECT sid, season, name FROM storms.ibtracs_storms"
                f" WHERE sid IN ({_placeholders})"
            ),
            _conn,
        )
    _engine.dispose()
    df_exp = df_exp_raw.merge(_df_meta, on="sid", how="left")
    return (df_exp,)


@app.cell
def load_impact(PROJECT_PREFIX, pd, stratus):
    _blob = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
    _df_all = stratus.load_parquet_from_blob(_blob)
    _keep = [
        "sid",
        "Event Name",
        "Start Year",
        "Total Affected",
        "Amount in US$",
    ]
    df_impact = _df_all[
        [c for c in _keep if c in _df_all.columns]
    ].drop_duplicates("sid")
    # Ike 2008 impact significantly underreported in EM-DAT
    df_impact.loc[df_impact["sid"] == "2008245N17323", "Total Affected"] = (
        2_600_000
    )
    # Melissa 2025 not yet in EM-DAT
    _MELISSA_SID = "2025291N11319"
    if _MELISSA_SID not in df_impact["sid"].values:
        _mel_impact = pd.DataFrame(
            [
                {
                    "sid": _MELISSA_SID,
                    "Total Affected": 2_200_017,
                    "Amount in US$": 3_500_015 + 4_000_001,
                }
            ]
        )
        df_impact = pd.concat([df_impact, _mel_impact], ignore_index=True)
    else:
        df_impact.loc[df_impact["sid"] == _MELISSA_SID, "Total Affected"] = (
            2_200_017
        )
        df_impact.loc[df_impact["sid"] == _MELISSA_SID, "Amount in US$"] = (
            3_500_015 + 4_000_001
        )
    return (df_impact,)


@app.cell
def load_old_trigger(PROJECT_PREFIX, pd, stratus, text):
    _MELISSA_SID = "2025291N11319"

    # Look up Melissa's actual atcf_id from DB
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        _df_mel_atcf = pd.read_sql(
            text("SELECT atcf_id FROM storms.ibtracs_storms WHERE sid = :sid"),
            _conn,
            params={"sid": _MELISSA_SID},
        )
    _engine.dispose()
    _melissa_atcf = (
        _df_mel_atcf["atcf_id"].iloc[0] if not _df_mel_atcf.empty else None
    )

    # Historical combined stats (one row per storm, 2000-2024)
    _blob = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
    _df = stratus.load_parquet_from_blob(_blob)

    # Option 1b thresholds (index 13136)
    _df["fcast_trig"] = (_df["wind"] >= 120.0) & (_df["q80"] >= 35.698547)
    _df["obsv_trig"] = (_df["wind_obsv"] >= 105.0) & (
        _df["q80_obsv"] >= 96.217003
    )
    _df = _df[["sid", "fcast_trig", "obsv_trig", "q80_obsv"]].copy()

    # Add Melissa 2025 if not already in historical parquet
    if _MELISSA_SID not in _df["sid"].values:
        _q80_obsv_mel = None
        _obsv_trig_mel = True
        _fcast_trig_mel = True

        if _melissa_atcf is not None:
            try:
                _obsv_blob = (
                    f"{PROJECT_PREFIX}/monitoring/cub_obsv_monitoring.parquet"
                )
                _df_obsv_mon = stratus.load_parquet_from_blob(_obsv_blob)
                _mel_obsv = _df_obsv_mon[
                    _df_obsv_mon["atcf_id"].str.upper()
                    == _melissa_atcf.upper()
                ]
                if not _mel_obsv.empty:
                    if "obsv_p" in _mel_obsv.columns:
                        _q80_obsv_mel = float(_mel_obsv["obsv_p"].max())
                    if "obsv_trigger" in _mel_obsv.columns:
                        _obsv_trig_mel = bool(_mel_obsv["obsv_trigger"].any())
            except Exception:
                pass

            try:
                _fcast_blob = (
                    f"{PROJECT_PREFIX}/monitoring/cub_fcast_monitoring.parquet"
                )
                _df_fcast_mon = stratus.load_parquet_from_blob(_fcast_blob)
                _mel_fcast = _df_fcast_mon[
                    _df_fcast_mon["atcf_id"].str.upper()
                    == _melissa_atcf.upper()
                ]
                if (
                    not _mel_fcast.empty
                    and "action_trigger" in _mel_fcast.columns
                ):
                    _fcast_trig_mel = bool(_mel_fcast["action_trigger"].any())
            except Exception:
                pass

        _mel_row = pd.DataFrame(
            [
                {
                    "sid": _MELISSA_SID,
                    "fcast_trig": _fcast_trig_mel,
                    "obsv_trig": _obsv_trig_mel,
                    "q80_obsv": _q80_obsv_mel,
                }
            ]
        )
        _df = pd.concat([_df, _mel_row], ignore_index=True)

    df_old_trig = _df
    return (df_old_trig,)


@app.cell
def load_fcast_exposure(pd, stratus, text):
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        _df_raw = pd.read_sql(
            text(
                """
                SELECT atcf_id,
                       issued_time AS issue_time,
                       wind_speed_kt,
                       pop_exposed
                FROM storms.nhc_tracks_fcast_exposure
                WHERE iso3 = 'CUB' AND admin_level = 0
            """
            ),
            _conn,
        )
        _df_sid = pd.read_sql(
            text("SELECT sid, atcf_id FROM storms.ibtracs_storms"),
            _conn,
        )
        if _df_raw.empty:
            _df_fallback = pd.read_sql(
                text(
                    """
                    SELECT sid, wind_speed_kt, pop_exposed
                    FROM storms.ibtracs_wind_exposure
                    WHERE iso3 = 'CUB' AND admin_level = 0
                """
                ),
                _conn,
            )
        else:
            _df_fallback = None
    _engine.dispose()

    if not _df_raw.empty:
        _df_raw = _df_raw.merge(_df_sid, on="atcf_id", how="left")
        # No cutoff filter — use max forecast exposure across all issuances
        df_fcast_exp = (
            _df_raw.groupby(["sid", "wind_speed_kt"])["pop_exposed"]
            .max()
            .reset_index()
            .dropna(subset=["sid"])
        )
    else:
        df_fcast_exp = _df_fallback

    return (df_fcast_exp,)


@app.cell
def load_monitors(PROJECT_PREFIX, pd, stratus):
    _blob = f"{PROJECT_PREFIX}/processed/nhc/monitors_nhc_chirpsgefs.parquet"
    df_monitors = stratus.load_parquet_from_blob(_blob)
    return (df_monitors,)


@app.cell
def load_codab(load_codab_from_blob):
    gdf_cub = load_codab_from_blob(admin_level=0)
    return (gdf_cub,)


@app.cell
def trigger_table(df_exp, df_impact, df_old_trig, mo, pd):
    _n = 9

    _exp_pivot = (
        df_exp.pivot_table(
            index="sid",
            columns="wind_speed_kt",
            values="pop_exposed",
            aggfunc="first",
        )
        .rename(columns={34: "exp_34", 50: "exp_50", 64: "exp_64"})
        .reset_index()
    )
    for _c in ["exp_34", "exp_50", "exp_64"]:
        if _c not in _exp_pivot.columns:
            _exp_pivot[_c] = 0

    def _get_thresh(col: str) -> float:
        _vals = _exp_pivot[col].fillna(0).sort_values(ascending=False)
        return float(_vals.iloc[_n - 1]) if _n <= len(_vals) else 0.0

    _thresh = {
        34: _get_thresh("exp_34"),
        50: _get_thresh("exp_50"),
        64: _get_thresh("exp_64"),
    }

    _n_years = int(df_exp["season"].max() - df_exp["season"].min() + 1)
    _rp = (_n_years + 1) / _n

    _meta = df_exp[["sid", "season", "name"]].drop_duplicates("sid")
    _df = _meta.merge(_exp_pivot, on="sid", how="outer")
    _df = _df.merge(df_impact, on="sid", how="outer")
    _df = _df.drop_duplicates(subset=["sid"])
    _df["season"] = pd.to_numeric(
        _df["season"].fillna(_df["sid"].str[:4]), errors="coerce"
    ).astype("Int64")
    if "Event Name" in _df.columns:
        _df["name"] = _df["name"].fillna(_df["Event Name"])

    _df = _df.merge(
        df_old_trig[["sid", "fcast_trig", "obsv_trig"]], on="sid", how="left"
    )
    _df["fcast_trig"] = (
        _df["fcast_trig"].astype("boolean").fillna(False).astype(bool)
    )
    _df["obsv_trig"] = (
        _df["obsv_trig"].astype("boolean").fillna(False).astype(bool)
    )

    _has_emdat = _df["Total Affected"].notna() & (_df["Total Affected"] > 0)
    _has_cerf = _df["Amount in US$"].notna() & (_df["Amount in US$"] > 0)
    _has_exp = (
        (_df["exp_34"].fillna(0) > 0)
        | (_df["exp_50"].fillna(0) > 0)
        | (_df["exp_64"].fillna(0) > 0)
    )
    _df = _df[_has_emdat | _has_cerf | _has_exp].copy()

    _df["trig_34"] = _df["exp_34"].fillna(0) >= _thresh[34]
    _df["trig_50"] = _df["exp_50"].fillna(0) >= _thresh[50]
    _df["trig_64"] = _df["exp_64"].fillna(0) >= _thresh[64]

    def _cerf_str(row):
        if pd.notna(row["Amount in US$"]) and row["Amount in US$"] > 0:
            return f"${row['Amount in US$']:,.0f}"
        if pd.notna(row["season"]) and int(row["season"]) >= 2006:
            return "—"
        return "pre-"

    _df["CERF"] = _df.apply(_cerf_str, axis=1)
    _df["Old fcast."] = _df["fcast_trig"].map({True: "✓", False: "—"})
    _df["Old obsv."] = _df["obsv_trig"].map({True: "✓", False: "—"})

    def _storm_label(row):
        _nm = (
            str(row["name"]).strip().title()
            if pd.notna(row["name"])
            else "Unnamed"
        )
        return f"{_nm} ({row['season']})"

    _df["Storm"] = _df.apply(_storm_label, axis=1)
    _df = _df.sort_values(
        ["Total Affected", "Amount in US$", "exp_64", "exp_50", "exp_34"],
        ascending=False,
        na_position="last",
    )

    _display = (
        _df[
            [
                "Storm",
                "exp_34",
                "exp_50",
                "exp_64",
                "trig_34",
                "trig_50",
                "trig_64",
                "Total Affected",
                "CERF",
                "Old fcast.",
                "Old obsv.",
            ]
        ]
        .rename(
            columns={"exp_34": "34 kt", "exp_50": "50 kt", "exp_64": "64 kt"}
        )
        .reset_index(drop=True)
    )

    def _style_row(row):
        _styles = [""] * len(row)
        _idx = list(row.index)
        for _col, _trig in [
            ("34 kt", "trig_34"),
            ("50 kt", "trig_50"),
            ("64 kt", "trig_64"),
        ]:
            if _trig in _idx and row[_trig]:
                _styles[_idx.index(_col)] = (
                    "background-color: gold; font-weight: bold"
                )
        return _styles

    def _style_check(val):
        if val == "✓":
            return (
                "background-color: #fff0b3; color: #888; font-weight: normal"
            )
        return "color: #ccc"

    def _style_cerf(val):
        if isinstance(val, str) and val.startswith("$"):
            return "background-color: crimson; color: white; font-weight: bold"
        if val == "—":
            return "background-color: #cce5ff; color: #555"
        return "color: #aaa"

    _styled = (
        _display.style.apply(_style_row, axis=1)
        .map(_style_check, subset=["Old fcast.", "Old obsv."])
        .map(_style_cerf, subset=["CERF"])
        .bar(subset=["Total Affected"], color="#b39ddb", vmin=0)
        .format(
            {
                "34 kt": lambda x: (
                    f"{int(x):,}" if pd.notna(x) and x > 0 else "—"
                ),
                "50 kt": lambda x: (
                    f"{int(x):,}" if pd.notna(x) and x > 0 else "—"
                ),
                "64 kt": lambda x: (
                    f"{int(x):,}" if pd.notna(x) and x > 0 else "—"
                ),
                "Total Affected": lambda x: (
                    f"{x:,.0f}" if pd.notna(x) else "—"
                ),
            }
        )
        .hide(axis="columns", subset=["trig_34", "trig_50", "trig_64"])
        .hide(axis="index")
    )

    _summary = mo.md(
        f"**Return period: {_rp:.1f} yrs** ({_n} storms / {_n_years} yrs)  \n"
        f"Thresholds: **34 kt** ≥ {int(_thresh[34]):,} · "
        f"**50 kt** ≥ {int(_thresh[50]):,} · "
        f"**64 kt** ≥ {int(_thresh[64]):,} people exposed"
    )

    mo.output.replace(mo.vstack([_summary, mo.Html(_styled.to_html())]))
    df_triggers = _df
    thresh = _thresh
    return df_triggers, thresh


@app.cell
def storm_selector(df_exp, mo, pd):
    _exposed_sids = df_exp[df_exp["pop_exposed"] > 0]["sid"].unique()
    _storms = (
        df_exp[df_exp["sid"].isin(_exposed_sids)][["sid", "season", "name"]]
        .drop_duplicates("sid")
        .copy()
    )

    def _label(row):
        _name = (
            str(row["name"]).strip().title() if pd.notna(row["name"]) else ""
        )
        _yr = int(row["season"]) if pd.notna(row["season"]) else row["sid"][:4]
        return (
            f"{_name} ({_yr}) — {row['sid']}"
            if _name
            else f"({_yr}) — {row['sid']}"
        )

    _storms["label"] = _storms.apply(_label, axis=1)
    _storms = _storms.sort_values("season", ascending=False)
    _storm_map_dict = dict(zip(_storms["label"], _storms["sid"]))

    _default_melissa = "2025291N11319"
    _default_key = next(
        (k for k, v in _storm_map_dict.items() if v == _default_melissa), None
    )

    storm_sel = mo.ui.dropdown(
        options=_storm_map_dict,
        value=_default_key,
        label="Select storm for map",
    )
    storm_sel
    return (storm_sel,)


@app.cell
def storm_map(
    df_exp,
    df_fcast_exp,
    df_monitors,
    df_old_trig,
    gdf_cub,
    gpd,
    mo,
    mpatches,
    pd,
    plt,
    storm_sel,
    stratus,
    text,
):
    _sid = storm_sel.value
    mo.stop(
        _sid is None,
        mo.md("Select a storm above to view the wind buffer map."),
    )

    _dev_engine = stratus.get_engine(stage="dev")
    with _dev_engine.connect() as _con:
        _gdf_bufs = gpd.read_postgis(
            text(
                "SELECT sid, wind_speed_kt, geometry"
                " FROM storms.ibtracs_wind_buffers"
                " WHERE sid = :sid"
            ),
            _con,
            geom_col="geometry",
            params={"sid": _sid},
        )
        _df_atcf = pd.read_sql(
            text("SELECT atcf_id FROM storms.ibtracs_storms WHERE sid = :sid"),
            _con,
            params={"sid": _sid},
        )
        _atcf_id = _df_atcf["atcf_id"].iloc[0] if not _df_atcf.empty else None
        if _atcf_id is not None:
            _df_fexp_ts = pd.read_sql(
                text(
                    """
                    SELECT issued_time, wind_speed_kt, pop_exposed
                    FROM storms.nhc_tracks_fcast_exposure
                    WHERE UPPER(atcf_id) = UPPER(:atcf_id)
                      AND iso3 = 'CUB' AND admin_level = 0
                    ORDER BY issued_time
                """
                ),
                _con,
                params={"atcf_id": _atcf_id},
            )
        else:
            _df_fexp_ts = pd.DataFrame(
                columns=["issued_time", "wind_speed_kt", "pop_exposed"]
            )
    _dev_engine.dispose()

    _row = df_exp[df_exp["sid"] == _sid]
    _name = (
        str(_row.iloc[0]["name"]).strip().title()
        if not _row.empty and pd.notna(_row.iloc[0]["name"])
        else ""
    )
    _yr = (
        int(_row.iloc[0]["season"])
        if not _row.empty and pd.notna(_row.iloc[0]["season"])
        else _sid[:4]
    )
    _title = f"{_name} ({_yr})" if _name else _sid[:4]
    _exp_by_wt = (
        _row[_row["pop_exposed"] > 0]
        .set_index("wind_speed_kt")["pop_exposed"]
        .to_dict()
    )

    # Rainfall evolution from monitors parquet
    _storm_mon = pd.DataFrame()
    if _atcf_id is not None and not df_monitors.empty:
        _storm_mon = df_monitors[
            df_monitors["atcf_id"].str.upper() == _atcf_id.upper()
        ]

    _prec_ts = pd.DataFrame()
    if not _storm_mon.empty and "lt_name" in _storm_mon.columns:
        _prec_ts = _storm_mon[_storm_mon["lt_name"] == "action"][
            ["issue_time", "q80"]
        ].sort_values("issue_time")

    # "Storm time" reference: last issue_time in monitors for this storm
    _storm_time = None
    if not _storm_mon.empty and "issue_time" in _storm_mon.columns:
        _storm_time = _storm_mon["issue_time"].max()

    # Observed rainfall reference from df_old_trig
    _obs_rain_val = None
    _trig_row = df_old_trig[df_old_trig["sid"] == _sid]
    if not _trig_row.empty and "q80_obsv" in _trig_row.columns:
        _v = _trig_row["q80_obsv"].iloc[0]
        if pd.notna(_v):
            _obs_rain_val = float(_v)

    # ── Figure layout ─────────────────────────────────────────────────────
    _cmap = plt.get_cmap("YlOrRd")
    _norm = plt.Normalize(vmin=20, vmax=80)
    _WIND_SPEEDS = [34, 50, 64]
    _WKT_COLORS = {
        34: _cmap(_norm(34)),
        50: _cmap(_norm(50)),
        64: _cmap(_norm(64)),
    }

    _fig = plt.figure(figsize=(22, 7), dpi=120)
    _gs = _fig.add_gridspec(
        2, 2, width_ratios=[1.1, 1.9], hspace=0.45, wspace=0.3
    )
    _ax_map = _fig.add_subplot(_gs[:, 0])
    _ax_exp = _fig.add_subplot(_gs[0, 1])
    _ax_prec = _fig.add_subplot(_gs[1, 1])

    # ── Map ───────────────────────────────────────────────────────────────
    gdf_cub.boundary.plot(ax=_ax_map, linewidth=0.8, color="k")
    _patches = []
    for _wt in _WIND_SPEEDS:
        if "wind_speed_kt" not in _gdf_bufs.columns:
            continue
        _buf = _gdf_bufs[
            (_gdf_bufs["wind_speed_kt"] == _wt)
            & _gdf_bufs.geometry.notna()
            & ~_gdf_bufs.geometry.is_empty
        ]
        if not _buf.empty:
            _color = _cmap(_norm(_wt))
            _buf.plot(
                ax=_ax_map,
                alpha=0.4,
                color=_color,
                edgecolor=_color,
                linewidth=0.5,
            )
            _exp_str = (
                f" — {int(_exp_by_wt[_wt]):,}" if _wt in _exp_by_wt else ""
            )
            _patches.append(
                mpatches.Patch(color=_color, label=f"{_wt} kt{_exp_str}")
            )
    if _patches:
        _ax_map.legend(handles=_patches[::-1], loc="lower left", fontsize=8)
    else:
        _ax_map.text(
            0.5,
            0.5,
            "No wind buffer data available",
            transform=_ax_map.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
    _minx, _miny, _maxx, _maxy = gdf_cub.total_bounds
    _pad = 3
    _ax_map.set_xlim(_minx - _pad, _maxx + _pad)
    _ax_map.set_ylim(_miny - _pad, _maxy + _pad)
    _ax_map.set_title(_title, fontsize=11)
    _ax_map.set_axis_off()

    # ── Forecast exposure evolution ────────────────────────────────────────
    # Max forecast exposure per wind speed for this storm (from df_fcast_exp)
    _fexp_by_wt = (
        df_fcast_exp[df_fcast_exp["sid"] == _sid]
        .set_index("wind_speed_kt")["pop_exposed"]
        .to_dict()
        if _sid in df_fcast_exp["sid"].values
        else {}
    )

    if not _df_fexp_ts.empty:
        # Full time-series available (nhc_tracks_fcast_exposure has CUB data)
        for _wt in _WIND_SPEEDS:
            _sub = _df_fexp_ts[
                _df_fexp_ts["wind_speed_kt"] == _wt
            ].sort_values("issued_time")
            if not _sub.empty:
                _ax_exp.plot(
                    _sub["issued_time"],
                    _sub["pop_exposed"],
                    color=_WKT_COLORS[_wt],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=f"{_wt} kt fcast",
                )
        # Observed exposure reference lines
        for _wt in _WIND_SPEEDS:
            if _wt in _exp_by_wt:
                _ax_exp.axhline(
                    _exp_by_wt[_wt],
                    color=_WKT_COLORS[_wt],
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
    elif _fexp_by_wt:
        # No time-series — show max forecast exposure as dashed lines
        for _wt in _WIND_SPEEDS:
            if _wt in _fexp_by_wt:
                _ax_exp.axhline(
                    _fexp_by_wt[_wt],
                    color=_WKT_COLORS[_wt],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{_wt} kt fcast max: {int(_fexp_by_wt[_wt]):,}",
                )
            if _wt in _exp_by_wt:
                _ax_exp.axhline(
                    _exp_by_wt[_wt],
                    color=_WKT_COLORS[_wt],
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{_wt} kt obsv: {int(_exp_by_wt[_wt]):,}",
                )
        _ax_exp.set_xlabel("No time-series — max fcast (--) vs observed (:)")
    else:
        _ax_exp.text(
            0.5,
            0.5,
            "No forecast exposure data",
            transform=_ax_exp.transAxes,
            ha="center",
            va="center",
            color="grey",
        )
    if _storm_time is not None:
        _ax_exp.axvline(
            _storm_time,
            color="#333",
            linestyle="-",
            linewidth=1.2,
            label="Storm time",
        )
    _ax_exp.set_title(
        "Forecast exposure evolution (dotted = observed)", fontsize=9
    )
    _ax_exp.set_ylabel("Pop. exposed")
    _ax_exp.tick_params(axis="x", labelsize=7, rotation=20)
    _exp_handles, _exp_labels = _ax_exp.get_legend_handles_labels()
    if _exp_handles:
        _ax_exp.legend(fontsize=7, loc="upper left")
    _ax_exp.grid(True, alpha=0.25, linestyle="--")
    _ax_exp.set_ylim(bottom=0)

    # ── Forecast rainfall evolution ────────────────────────────────────────
    if not _prec_ts.empty and "q80" in _prec_ts.columns:
        _ax_prec.plot(
            _prec_ts["issue_time"],
            _prec_ts["q80"],
            color="steelblue",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="Forecast q80 (CHIRPS-GEFS)",
        )
    if _obs_rain_val is not None:
        _ax_prec.axhline(
            _obs_rain_val,
            color="darkorange",
            linestyle=":",
            linewidth=1.5,
            label=f"Observed q80 ({_obs_rain_val:.0f} mm)",
        )
    if _storm_time is not None:
        _ax_prec.axvline(
            _storm_time,
            color="#333",
            linestyle="-",
            linewidth=1.2,
            label="Storm time",
        )
    _ax_prec.set_title(
        "Forecast rainfall q80 CHIRPS-GEFS (dotted = observed IMERG)",
        fontsize=9,
    )
    _ax_prec.set_ylabel("2-day rainfall q80 (mm)")
    _ax_prec.tick_params(axis="x", labelsize=7, rotation=20)
    _prec_handles, _prec_labels = _ax_prec.get_legend_handles_labels()
    if _prec_handles:
        _ax_prec.legend(fontsize=7, loc="upper left")
    _ax_prec.grid(True, alpha=0.25, linestyle="--")
    _ax_prec.set_ylim(bottom=0)

    _fig.suptitle(_title, fontsize=12, y=1.01)
    _fig
    return


@app.cell
def corr_plots(df_triggers, plt, thresh):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, sharey=True)

    for _ax, (_wt, _col, _trig_col, _label) in zip(
        _axes,
        [
            (34, "exp_34", "trig_34", "34 kt"),
            (50, "exp_50", "trig_50", "50 kt"),
            (64, "exp_64", "trig_64", "64 kt"),
        ],
    ):
        _sub = df_triggers[
            (df_triggers[_col].fillna(0) > 0)
            & df_triggers["Total Affected"].notna()
            & (df_triggers["Total Affected"] > 0)
        ][["Storm", _col, "Total Affected", _trig_col]].copy()

        _r = _sub[_col].corr(_sub["Total Affected"])

        _ax.scatter(_sub[_col], _sub["Total Affected"], alpha=0)

        for _, _row in _sub.iterrows():
            _ax.annotate(
                _row["Storm"],
                xy=(_row[_col], _row["Total Affected"]),
                ha="center",
                va="center",
                fontsize=7,
                color="crimson" if _row[_trig_col] else "#999999",
            )

        _ax.axvline(
            thresh[_wt],
            color="crimson",
            linewidth=1,
            linestyle="--",
            label=f"Threshold: {int(thresh[_wt]):,}",
        )
        _ax.legend(fontsize=7, loc="upper left")
        _ax.set_xlim(left=0)
        _ax.set_ylim(bottom=0)
        _ax.set_xlabel(f"Pop. exposed ({_label})")
        _ax.set_title(f"{_label} exposure vs. Total Affected  (r = {_r:.2f})")
        _ax.grid(True, alpha=0.3, linestyle="--")
        if _ax is _axes[0]:
            _ax.set_ylabel("Total Affected (EM-DAT)")

    _fig.suptitle("Wind exposure vs. EM-DAT Total Affected", y=1.01)
    plt.tight_layout()
    _fig
    return


@app.cell
def trigger_corr_table(df_rain_opt, mo, pd, plt):
    _cols = {
        "fcast_exp_34": "Fcast exp 34",
        "fcast_exp_50": "Fcast exp 50",
        "fcast_exp_64": "Fcast exp 64",
        "exp_34": "Obsv exp 34",
        "exp_50": "Obsv exp 50",
        "exp_64": "Obsv exp 64",
        "max_obs_rain": "Obs rain (q80)",
        "fcast_trig_old": "Old fcast",
        "obsv_trig_old": "Old obsv",
        "Total Affected": "Impact",
        "has_cerf": "CERF",
    }
    _df = df_rain_opt[[c for c in _cols if c in df_rain_opt.columns]].copy()
    for _c in ["fcast_trig_old", "obsv_trig_old", "has_cerf"]:
        if _c in _df.columns:
            _df[_c] = _df[_c].astype(float)
    _df = _df.rename(columns=_cols)
    _corr = _df.corr(numeric_only=True).round(2)

    _fig, _ax = plt.subplots(figsize=(10, 8), dpi=120)
    _mat = _corr.values
    _im = _ax.imshow(_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    _fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
    _labels = list(_corr.columns)
    _ax.set_xticks(range(len(_labels)))
    _ax.set_yticks(range(len(_labels)))
    _ax.set_xticklabels(_labels, rotation=45, ha="right", fontsize=8)
    _ax.set_yticklabels(_labels, fontsize=8)
    for _i in range(len(_labels)):
        for _j in range(len(_labels)):
            _v = _mat[_i, _j]
            _ax.text(
                _j,
                _i,
                f"{_v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(_v) > 0.5 else "#333",
            )
    _ax.set_title(
        "Pearson correlations — trigger indicators vs. CERF & impact",
        fontsize=10,
    )
    plt.tight_layout()
    _fig


@app.cell
def rain_trigger_opt(df_exp, df_fcast_exp, df_impact, df_old_trig, mo, pd):
    _n = 11  # RP ≈ 2.4 yrs over 2000-2025

    # ── Build base data frame ─────────────────────────────────────────────
    _exp_pivot = (
        df_exp.pivot_table(
            index="sid",
            columns="wind_speed_kt",
            values="pop_exposed",
            aggfunc="first",
        )
        .rename(columns={34: "exp_34", 50: "exp_50", 64: "exp_64"})
        .reset_index()
    )
    for _c in ["exp_34", "exp_50", "exp_64"]:
        if _c not in _exp_pivot.columns:
            _exp_pivot[_c] = 0

    _fexp_pivot = (
        df_fcast_exp.pivot_table(
            index="sid",
            columns="wind_speed_kt",
            values="pop_exposed",
            aggfunc="max",
        )
        .rename(
            columns={
                34: "fcast_exp_34",
                50: "fcast_exp_50",
                64: "fcast_exp_64",
            }
        )
        .reset_index()
    )
    for _c in ["fcast_exp_34", "fcast_exp_50", "fcast_exp_64"]:
        if _c not in _fexp_pivot.columns:
            _fexp_pivot[_c] = 0

    _meta = df_exp[["sid", "season", "name"]].drop_duplicates("sid")
    _opt = _meta.merge(_exp_pivot, on="sid", how="outer")
    _opt = _opt.merge(_fexp_pivot, on="sid", how="outer")
    _opt = _opt.merge(
        df_old_trig[["sid", "q80_obsv", "fcast_trig", "obsv_trig"]],
        on="sid",
        how="left",
    )
    _opt = _opt.merge(
        df_impact[["sid", "Total Affected", "Amount in US$"]],
        on="sid",
        how="left",
    )
    _opt = _opt.drop_duplicates("sid")
    for _c in [
        "exp_34",
        "exp_50",
        "exp_64",
        "fcast_exp_34",
        "fcast_exp_50",
        "fcast_exp_64",
    ]:
        _opt[_c] = _opt[_c].fillna(0)
    _opt["season"] = pd.to_numeric(
        _opt["season"].fillna(_opt["sid"].str[:4]), errors="coerce"
    ).astype("Int64")
    _opt["max_obs_rain"] = _opt["q80_obsv"]
    _opt["has_cerf"] = _opt["Amount in US$"].notna() & (
        _opt["Amount in US$"] > 0
    )
    _opt["fcast_trig_old"] = (
        _opt["fcast_trig"].astype("boolean").fillna(False).astype(bool)
    )
    _opt["obsv_trig_old"] = (
        _opt["obsv_trig"].astype("boolean").fillna(False).astype(bool)
    )

    def _storm_label_opt(row):
        _nm = (
            str(row["name"]).strip().title()
            if pd.notna(row["name"])
            else "Unnamed"
        )
        _yr = row["season"] if pd.notna(row["season"]) else row["sid"][:4]
        return f"{_nm} ({_yr})"

    _opt["Storm"] = _opt.apply(_storm_label_opt, axis=1)

    # Old combined trigger
    _opt["old_combined"] = _opt["fcast_trig_old"] | _opt["obsv_trig_old"]

    # Lookup dicts for speed
    _cerf_lookup = dict(zip(_opt["sid"], _opt["has_cerf"]))
    _affected_lookup = dict(zip(_opt["sid"], _opt["Total Affected"].fillna(0)))

    # ── Double sweep ──────────────────────────────────────────────────────
    _results = []
    for _wkt, _fcol, _ocol in [
        (34, "fcast_exp_34", "exp_34"),
        (50, "fcast_exp_50", "exp_50"),
        (64, "fcast_exp_64", "exp_64"),
    ]:
        _exp_f_vals = sorted(_opt[_fcol].dropna().unique())
        _exp_o_vals = sorted(_opt[_ocol].dropna().unique())

        for _e_thresh_f in _exp_f_vals:
            _pool_fcast = _opt[_opt[_fcol] >= _e_thresh_f]
            _n_f = len(_pool_fcast)
            if _n_f > _n:
                continue
            _n_o = _n - _n_f
            _fcast_sids_set = set(_pool_fcast["sid"])
            _not_fcast = _opt[~_opt["sid"].isin(_fcast_sids_set)]

            if _n_o == 0:
                _combined_sids = frozenset(_fcast_sids_set)
                _results.append(
                    {
                        "wkt": _wkt,
                        "exp_thresh_f": _e_thresh_f,
                        "exp_thresh_o": None,
                        "r_obs_exact": None,
                        "r_obs": None,
                        "n_fcast": _n_f,
                        "n_obsv": 0,
                        "cerf_count": sum(
                            1
                            for _s in _combined_sids
                            if _cerf_lookup.get(_s, False)
                        ),
                        "total_affected": sum(
                            _affected_lookup.get(_s, 0)
                            for _s in _combined_sids
                        ),
                        "_combined_sids": _combined_sids,
                        "_fcast_sids": frozenset(_fcast_sids_set),
                    }
                )
            else:
                for _e_thresh_o in _exp_o_vals:
                    _obs_eligible = _not_fcast[
                        _not_fcast[_ocol] >= _e_thresh_o
                    ]
                    if len(_obs_eligible) < _n_o:
                        continue
                    _oe_s = _obs_eligible[
                        _obs_eligible["max_obs_rain"].notna()
                    ].sort_values("max_obs_rain", ascending=False)
                    if len(_oe_s) < _n_o:
                        continue
                    _r_obs = float(_oe_s.iloc[_n_o - 1]["max_obs_rain"])
                    _obs_new_sids = set(
                        _obs_eligible[
                            _obs_eligible["max_obs_rain"].notna()
                            & (_obs_eligible["max_obs_rain"] >= _r_obs)
                        ]["sid"]
                    )
                    if len(_obs_new_sids) != _n_o:
                        continue
                    _combined_sids = frozenset(_fcast_sids_set | _obs_new_sids)
                    if len(_combined_sids) != _n:
                        continue
                    _results.append(
                        {
                            "wkt": _wkt,
                            "exp_thresh_f": _e_thresh_f,
                            "exp_thresh_o": _e_thresh_o,
                            "r_obs_exact": _r_obs,
                            "r_obs": round(_r_obs, 1),
                            "n_fcast": _n_f,
                            "n_obsv": _n_o,
                            "cerf_count": sum(
                                1
                                for _s in _combined_sids
                                if _cerf_lookup.get(_s, False)
                            ),
                            "total_affected": sum(
                                _affected_lookup.get(_s, 0)
                                for _s in _combined_sids
                            ),
                            "_combined_sids": _combined_sids,
                            "_fcast_sids": frozenset(_fcast_sids_set),
                        }
                    )

    df_rain_opt = _opt.copy()
    rain_opt_thresh = {}

    if not _results:
        mo.output.replace(mo.md("⚠ No valid combinations found."))
    else:
        _df_res = pd.DataFrame(_results)

        # Best per (wkt, n_fcast): max CERF → max affected → min thresh
        _df_options = (
            _df_res.sort_values(
                [
                    "wkt",
                    "n_fcast",
                    "cerf_count",
                    "total_affected",
                    "exp_thresh_f",
                ],
                ascending=[True, True, False, False, True],
            )
            .groupby(["wkt", "n_fcast"], sort=True)
            .first()
            .reset_index()
        )
        # Best overall per wkt
        _df_best = (
            _df_res.sort_values(
                ["wkt", "cerf_count", "total_affected", "exp_thresh_f"],
                ascending=[True, False, False, True],
            )
            .groupby("wkt", sort=True)
            .first()
            .reset_index()
        )

        _best_keys = set(zip(_df_best["wkt"], _df_best["n_fcast"]))
        _df_options["best"] = [
            (r["wkt"], r["n_fcast"]) in _best_keys
            for _, r in _df_options.iterrows()
        ]

        # Apply best triggers to base dataframe
        for _, _brow in _df_best.iterrows():
            _wkt = int(_brow["wkt"])
            _opt[f"fcast_trig_{_wkt}"] = _opt["sid"].isin(_brow["_fcast_sids"])
            _opt[f"combined_{_wkt}"] = _opt["sid"].isin(
                _brow["_combined_sids"]
            )

        for _wkt in [34, 50, 64]:
            if f"fcast_trig_{_wkt}" not in _opt.columns:
                _opt[f"fcast_trig_{_wkt}"] = False
            if f"combined_{_wkt}" not in _opt.columns:
                _opt[f"combined_{_wkt}"] = False

        # Build best_thresh dict
        _best_thresh = {}
        for _, _r in _df_best.iterrows():
            _wkt = int(_r["wkt"])
            _best_thresh[_wkt] = {
                "exp_f": int(_r["exp_thresh_f"]),
                "exp_o": (
                    int(_r["exp_thresh_o"])
                    if pd.notna(_r["exp_thresh_o"])
                    else 0
                ),
                "r_obs": (
                    float(_r["r_obs_exact"])
                    if pd.notna(_r["r_obs_exact"])
                    else None
                ),
            }
        rain_opt_thresh = _best_thresh

        # Condition columns (using exact thresholds to avoid rounding issues)
        _bool_hide = []
        for _wkt, _fcol, _ocol in [
            (34, "fcast_exp_34", "exp_34"),
            (50, "fcast_exp_50", "exp_50"),
            (64, "fcast_exp_64", "exp_64"),
        ]:
            _t = _best_thresh.get(_wkt, {})
            _ef = _t.get("exp_f", float("inf"))
            _eo = _t.get("exp_o", float("inf"))
            _ro = _t.get("r_obs")
            _opt[f"_fexp_flag_{_wkt}"] = _opt[_fcol] >= _ef
            _opt[f"_oexp_flag_{_wkt}"] = _opt[_ocol] >= _eo
            _opt[f"_obs_rf_{_wkt}"] = (
                (_opt["max_obs_rain"] >= _ro)
                if _ro is not None
                else pd.Series(False, index=_opt.index)
            )
            # For Cuba: forecast trigger = just exposure (no rainfall)
            _opt[f"{_wkt} fexp"] = _opt[f"_fexp_flag_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt} oexp"] = _opt[f"_oexp_flag_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt} obs"] = _opt[f"_obs_rf_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt} kt"] = _opt[f"fcast_trig_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt}+O"] = _opt[f"combined_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _bool_hide += [
                f"_fexp_flag_{_wkt}",
                f"_oexp_flag_{_wkt}",
                f"_obs_rf_{_wkt}",
                f"fcast_trig_{_wkt}",
                f"combined_{_wkt}",
            ]

        df_rain_opt = _opt

        # ── Summary table ─────────────────────────────────────────────────
        _n_yrs = int(_opt["season"].max() - _opt["season"].min() + 1)
        _rp = (_n_yrs + 1) / _n

        _old_fcast_sids = set(_opt.loc[_opt["fcast_trig_old"], "sid"])
        _old_obsv_sids = set(
            _opt.loc[_opt["obsv_trig_old"] & ~_opt["fcast_trig_old"], "sid"]
        )
        _old_combined_sids = _old_fcast_sids | _old_obsv_sids
        _n_old = len(_old_combined_sids)
        _cerf_old = sum(
            1 for _s in _old_combined_sids if _cerf_lookup.get(_s, False)
        )
        _aff_old = int(
            sum(_affected_lookup.get(_s, 0) for _s in _old_combined_sids)
        )
        _rp_old = (_n_yrs + 1) / _n_old if _n_old else float("inf")

        _summary_rows = [
            {
                "Trigger": "Old option 1b",
                "Wind kt": "ZMA ≥120 (f) / ≥105 (o)",
                "Fcast exp thresh": "—",
                "Obs exp thresh": "—",
                "Obs rain (q80) mm": "≥96.2",
                "# Fcast": len(_old_fcast_sids),
                "# Obs": len(_old_obsv_sids),
                "CERF storms": _cerf_old,
                "Total Affected": _aff_old,
                "RP yrs": round(_rp_old, 1),
            }
        ]
        for _, _r in _df_best.iterrows():
            _wkt_b = int(_r["wkt"])
            _summary_rows.append(
                {
                    "Trigger": f"New {_wkt_b} kt ★",
                    "Wind kt": f"{_wkt_b}",
                    "Fcast exp thresh": f"{int(_r['exp_thresh_f']):,}",
                    "Obs exp thresh": (
                        f"{int(_r['exp_thresh_o']):,}"
                        if pd.notna(_r["exp_thresh_o"])
                        else "—"
                    ),
                    "Obs rain (q80) mm": (
                        f"{_r['r_obs']:.1f}" if pd.notna(_r["r_obs"]) else "—"
                    ),
                    "# Fcast": int(_r["n_fcast"]),
                    "# Obs": int(_r["n_obsv"]),
                    "CERF storms": int(_r["cerf_count"]),
                    "Total Affected": int(_r["total_affected"]),
                    "RP yrs": round(_rp, 1),
                }
            )

        _df_summary = pd.DataFrame(_summary_rows)

        def _style_summary(row):
            _styles = [""] * len(row)
            if "★" in str(row.get("Trigger", "")):
                _styles = ["background-color: #fffde7"] * len(row)
            return _styles

        _styled_summary = (
            _df_summary.style.apply(_style_summary, axis=1)
            .bar(subset=["Total Affected"], color="#b39ddb", vmin=0)
            .format(
                {
                    "Total Affected": lambda x: (
                        f"{int(x):,}" if pd.notna(x) else "—"
                    )
                }
            )
            .set_properties(**{"text-align": "center"})
            .set_properties(subset=["Trigger"], **{"text-align": "left"})
            .set_table_styles(
                [{"selector": "th", "props": [("text-align", "center")]}]
            )
            .hide(axis="index")
        )

        # ── Options table ─────────────────────────────────────────────────
        _df_opts_disp = _df_options[
            [
                "wkt",
                "n_fcast",
                "exp_thresh_f",
                "exp_thresh_o",
                "r_obs",
                "n_obsv",
                "cerf_count",
                "total_affected",
                "best",
            ]
        ].copy()
        _df_opts_disp["★"] = _df_opts_disp["best"].map({True: "★", False: ""})
        _df_opts_disp = _df_opts_disp.rename(
            columns={
                "wkt": "Wind kt",
                "n_fcast": "# Fcast",
                "exp_thresh_f": "Fcast exp",
                "exp_thresh_o": "Obs exp",
                "r_obs": "Obs rain mm",
                "n_obsv": "# Obs",
                "cerf_count": "CERF",
                "total_affected": "Total Aff.",
            }
        ).drop(columns=["best"])

        _styled_opts = (
            _df_opts_disp.style.format(
                {
                    "Fcast exp": lambda x: (
                        f"{int(x):,}" if pd.notna(x) else "—"
                    ),
                    "Obs exp": lambda x: (
                        f"{int(x):,}" if pd.notna(x) else "—"
                    ),
                    "Obs rain mm": lambda x: (
                        f"{x:.1f}" if pd.notna(x) else "—"
                    ),
                    "Total Aff.": lambda x: (
                        f"{int(x):,}" if pd.notna(x) else "—"
                    ),
                }
            )
            .set_properties(**{"text-align": "center"})
            .set_table_styles(
                [{"selector": "th", "props": [("text-align", "center")]}]
            )
            .hide(axis="index")
        )

        # ── Storm conditions table ────────────────────────────────────────
        def _cerf_str_opt(row):
            if pd.notna(row.get("Amount in US$")) and row["Amount in US$"] > 0:
                return f"${row['Amount in US$']:,.0f}"
            if pd.notna(row.get("season")) and int(row["season"]) >= 2006:
                return "—"
            return "pre-"

        _opt["CERF"] = _opt.apply(_cerf_str_opt, axis=1)
        _opt["Old fcast."] = _opt["fcast_trig_old"].map(
            {True: "✓", False: "—"}
        )
        _opt["Old obsv."] = _opt["obsv_trig_old"].map({True: "✓", False: "—"})

        _cond_cols = [
            f"{w} {c}" for w in [34, 50, 64] for c in ["fexp", "oexp", "obs"]
        ]
        _kt_cols = [f"{w} kt" for w in [34, 50, 64]]
        _comb_cols = [f"{w}+O" for w in [34, 50, 64]]

        _any_triggered = _opt[[f"combined_{w}" for w in [34, 50, 64]]].any(
            axis=1
        )
        _show = (
            _any_triggered
            | (_opt["Total Affected"].fillna(0) > 0)
            | _opt["has_cerf"]
        )
        _storm_table = (
            _opt[_show][
                [
                    "Storm",
                    "34 fexp",
                    "34 oexp",
                    "34 obs",
                    "34 kt",
                    "34+O",
                    "50 fexp",
                    "50 oexp",
                    "50 obs",
                    "50 kt",
                    "50+O",
                    "64 fexp",
                    "64 oexp",
                    "64 obs",
                    "64 kt",
                    "64+O",
                    *_bool_hide,
                    "old_combined",
                    "Total Affected",
                    "CERF",
                    "Old fcast.",
                    "Old obsv.",
                ]
            ]
            .sort_values("Total Affected", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

        def _sc(val):
            return (
                "background-color: #e8f5e9; color: #2e7d32"
                if val == "✓"
                else "color: #ddd"
            )

        def _st(val):
            return (
                "background-color: gold; font-weight: bold"
                if val == "✓"
                else "color: #ccc"
            )

        def _sco(val):
            return (
                "background-color: #ffa040; color: white; font-weight: bold"
                if val == "✓"
                else "color: #ccc"
            )

        def _sch(val):
            return (
                "background-color: #fff0b3; color: #888; font-weight: normal"
                if val == "✓"
                else "color: #ccc"
            )

        def _scerf(val):
            if isinstance(val, str) and val.startswith("$"):
                return "background-color: crimson; color: white; font-weight: bold"
            if val == "—":
                return "background-color: #cce5ff; color: #555"
            return "color: #aaa"

        _styled_storms = (
            _storm_table.style.map(_sc, subset=_cond_cols)
            .map(_st, subset=_kt_cols)
            .map(_sco, subset=_comb_cols)
            .map(_sch, subset=["Old fcast.", "Old obsv."])
            .map(_scerf, subset=["CERF"])
            .bar(subset=["Total Affected"], color="#b39ddb", vmin=0)
            .format(
                {
                    "Total Affected": lambda x: (
                        f"{x:,.0f}" if pd.notna(x) else "—"
                    )
                }
            )
            .hide(axis="columns", subset=_bool_hide + ["old_combined"])
            .hide(axis="index")
        )

        _rp_note = mo.md(
            f"**n = {_n} storms** — return period {_rp:.1f} yrs "
            f"({_n_yrs} seasons 2000–{int(_opt['season'].max())})"
        )
        mo.output.replace(
            mo.vstack(
                [
                    _rp_note,
                    mo.md("### Summary"),
                    mo.Html(_styled_summary.to_html()),
                    mo.md(
                        "### All options (best per wind level + fcast count)"
                    ),
                    mo.Html(_styled_opts.to_html()),
                    mo.md("### Storm conditions"),
                    mo.Html(_styled_storms.to_html()),
                ]
            )
        )

    return df_rain_opt, rain_opt_thresh


@app.cell
def rain_scatter(df_rain_opt, mpatches, mo, pd, plt, rain_opt_thresh):
    mo.stop(not len(df_rain_opt) or not rain_opt_thresh)
    _fig, _axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)

    for _col_idx, _wkt in enumerate([34, 50, 64]):
        _t = rain_opt_thresh.get(_wkt, {})
        _ef = _t.get("exp_f", 0)
        _eo = _t.get("exp_o", 0)
        _ro = _t.get("r_obs")

        for _row_idx, (_xcol, _exp_thresh, _row_label) in enumerate(
            [
                (f"fcast_exp_{_wkt}", _ef, "Fcast exp"),
                (f"exp_{_wkt}", _eo, "Obs exp"),
            ]
        ):
            _ax = _axes[_row_idx, _col_idx]
            _sub = df_rain_opt[
                (df_rain_opt[_xcol].fillna(0) > 0)
                & df_rain_opt["max_obs_rain"].notna()
            ].copy()

            # Color by trigger type
            def _color_point(row, wkt=_wkt):
                if row["has_cerf"]:
                    return "crimson"
                if row.get(f"combined_{wkt}", False):
                    if row.get(f"fcast_trig_{wkt}", False):
                        return "gold"
                    return "darkorange"
                return "#aaaaaa"

            _colors = [_color_point(r) for _, r in _sub.iterrows()]
            _sizes = [
                (
                    max(
                        20,
                        (float(v) ** 0.5)
                        * 500
                        / (df_rain_opt["Total Affected"].max() ** 0.5),
                    )
                    if pd.notna(v) and v > 0
                    else 20
                )
                for v in _sub["Total Affected"]
            ]

            _ax.scatter(
                _sub[_xcol],
                _sub["max_obs_rain"],
                c=_colors,
                s=_sizes,
                alpha=0.7,
                edgecolors="none",
                zorder=2,
            )
            for _, _row in _sub.iterrows():
                _triggered = bool(_row.get(f"combined_{_wkt}", False))
                _ax.annotate(
                    _row["Storm"],
                    xy=(_row[_xcol], _row["max_obs_rain"]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold" if _triggered else "normal",
                    zorder=3,
                )
            if _exp_thresh:
                _ax.axvline(
                    _exp_thresh,
                    color="steelblue",
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                )
            if _ro is not None:
                _ax.axhline(
                    _ro,
                    color="darkorange",
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                )
            _ax.set_xlabel(f"{_row_label} ({_wkt} kt)")
            _ax.set_ylabel("Obs rain q80 (mm)" if _col_idx == 0 else "")
            _ax.set_title(
                f"{_wkt} kt — {'Fcast' if _row_idx == 0 else 'Obs'} exposure"
            )
            _ax.grid(True, alpha=0.25, linestyle="--")
            _ax.set_xlim(left=0)
            _ax.set_ylim(bottom=0)

    _legend_patches = [
        mpatches.Patch(color="crimson", label="CERF"),
        mpatches.Patch(color="gold", label="Fcast trig"),
        mpatches.Patch(color="darkorange", label="Obs trig"),
        mpatches.Patch(color="#aaaaaa", label="Not triggered"),
    ]
    _fig.legend(
        handles=_legend_patches,
        loc="upper center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.01),
    )
    _fig.suptitle(
        "Exposure vs. observed rainfall (q80 IMERG) — bubble size ∝ impact",
        fontsize=11,
        y=1.04,
    )
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
