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
def load_total_exposure(pd, stratus, text):
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        _df_fcast = pd.read_sql(
            text(
                """
                SELECT atcf_id, issued_time, wind_speed_kt,
                       pop_exposed AS fcast_exp
                FROM storms.nhc_tracks_fcast_exposure
                WHERE iso3 = 'CUB' AND admin_level = 0
            """
            ),
            _conn,
        )
        _df_obsv = pd.read_sql(
            text(
                """
                SELECT atcf_id, valid_time, wind_speed_kt,
                       pop_exposed AS obsv_exp
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
        _df_ibtracs = pd.read_sql(
            text(
                """
                SELECT sid, wind_speed_kt,
                       pop_exposed AS max_total_exposure
                FROM storms.ibtracs_wind_exposure
                WHERE iso3 = 'CUB' AND admin_level = 0
            """
            ),
            _conn,
        )
    _engine.dispose()

    _df_fcast = _df_fcast.merge(_df_sid, on="atcf_id", how="left").dropna(
        subset=["sid"]
    )
    _df_obsv = _df_obsv.merge(_df_sid, on="atcf_id", how="left").dropna(
        subset=["sid"]
    )

    # For each (sid, wind_speed_kt): max(fcast_t + cumulative_obsv_t)
    _results = []
    for (_sid_v, _wkt), _fg in _df_fcast.groupby(["sid", "wind_speed_kt"]):
        _og = _df_obsv[
            (_df_obsv["sid"] == _sid_v) & (_df_obsv["wind_speed_kt"] == _wkt)
        ]
        _fs = _fg.sort_values("issued_time")
        if _og.empty:
            _max_total = float(_fs["fcast_exp"].max())
        else:
            _os = _og.sort_values("valid_time").assign(
                _cm=lambda x: x["obsv_exp"].cummax()
            )
            _m = pd.merge_asof(
                _fs[["issued_time", "fcast_exp"]],
                _os[["valid_time", "_cm"]].rename(
                    columns={"valid_time": "issued_time"}
                ),
                on="issued_time",
                direction="backward",
            )
            _max_total = float((_m["fcast_exp"] + _m["_cm"].fillna(0)).max())
        _results.append(
            {
                "sid": _sid_v,
                "wind_speed_kt": _wkt,
                "max_total_exposure": _max_total,
            }
        )

    _df_nhc = (
        pd.DataFrame(_results)
        if _results
        else pd.DataFrame(
            columns=["sid", "wind_speed_kt", "max_total_exposure"]
        )
    )
    _nhc_sids = set(_df_nhc["sid"].unique()) if not _df_nhc.empty else set()
    _df_fill = _df_ibtracs[~_df_ibtracs["sid"].isin(_nhc_sids)]
    df_total_exp = pd.concat(
        [_df_nhc, _df_fill] if not _df_nhc.empty else [_df_fill],
        ignore_index=True,
    )
    df_total_exp = df_total_exp[
        df_total_exp["sid"].str[:4].astype(int) >= 2002
    ]
    return (df_total_exp,)


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
def doc_wind_table(mo):
    mo.md(
        """
## Wind exposure trigger table

Shows population exposed to hurricane-force winds for all Cuba-affecting
storms since 2002, at three wind speed levels (34 / 50 / 64 kt).

Two exposure metrics are shown per wind level:

| Column | Definition |
|--------|-----------|
| **X kt final obsv** | Final observed population exposure — last valid_time from NHC observed tracks (`nhc_tracks_obsv_exposure`), or IBTrACS for historical storms without NHC data |
| **X kt max total fcast** | Maximum of **(forecast exposure + cumulative observed exposure)** across all NHC forecast issuances. At each `issued_time`, forecast exposure (what NHC predicts will be affected) is added to the largest observed exposure recorded up to that point, capturing the peak combined signal during the approach. |

**Highlighting:** gold = storm triggers the *final obsv* threshold at n = 10
(RP ≈ 2.6 yrs); blue = triggers the *max total fcast* threshold at n = 10.
These are independent thresholds computed separately.

**Old fcast. / Old obsv.** columns show whether each storm triggered under
historical option 1b: forecast = ZMA wind ≥ 120 kt AND CHIRPS-GEFS q80 ≥ 35.7 mm;
observational = ZMA wind ≥ 105 kt AND IMERG q80 ≥ 96.2 mm.
        """
    )


@app.cell
def trigger_table(df_exp, df_impact, df_old_trig, df_total_exp, mo, pd):
    _n = 10

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

    _texp_pivot = (
        df_total_exp.pivot_table(
            index="sid",
            columns="wind_speed_kt",
            values="max_total_exposure",
            aggfunc="max",
        )
        .rename(
            columns={
                34: "total_34",
                50: "total_50",
                64: "total_64",
            }
        )
        .reset_index()
    )
    for _c in ["total_34", "total_50", "total_64"]:
        if _c not in _texp_pivot.columns:
            _texp_pivot[_c] = 0

    def _get_thresh(col: str) -> float:
        _vals = _exp_pivot[col].fillna(0).sort_values(ascending=False)
        return float(_vals.iloc[_n - 1]) if _n <= len(_vals) else 0.0

    _thresh = {
        34: _get_thresh("exp_34"),
        50: _get_thresh("exp_50"),
        64: _get_thresh("exp_64"),
    }

    _meta = df_exp[["sid", "season", "name"]].drop_duplicates("sid")
    _df = _meta.merge(_exp_pivot, on="sid", how="outer")
    _df = _df.merge(_texp_pivot, on="sid", how="left")
    for _c in ["total_34", "total_50", "total_64"]:
        _df[_c] = _df[_c].fillna(0)

    def _get_thresh_total(col: str) -> float:
        _vals = _df[col].fillna(0).sort_values(ascending=False)
        return float(_vals.iloc[_n - 1]) if _n <= len(_vals) else 0.0

    _thresh_total = {
        34: _get_thresh_total("total_34"),
        50: _get_thresh_total("total_50"),
        64: _get_thresh_total("total_64"),
    }

    _n_years = int(df_exp["season"].max() - df_exp["season"].min() + 1)
    _rp = (_n_years + 1) / _n
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
    _df["trig_total_34"] = _df["total_34"].fillna(0) >= _thresh_total[34]
    _df["trig_total_50"] = _df["total_50"].fillna(0) >= _thresh_total[50]
    _df["trig_total_64"] = _df["total_64"].fillna(0) >= _thresh_total[64]

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
                "total_34",
                "exp_50",
                "total_50",
                "exp_64",
                "total_64",
                "trig_34",
                "trig_total_34",
                "trig_50",
                "trig_total_50",
                "trig_64",
                "trig_total_64",
                "Total Affected",
                "CERF",
                "Old fcast.",
                "Old obsv.",
            ]
        ]
        .rename(
            columns={
                "exp_34": "34 kt final obsv",
                "total_34": "34 kt max total fcast",
                "exp_50": "50 kt final obsv",
                "total_50": "50 kt max total fcast",
                "exp_64": "64 kt final obsv",
                "total_64": "64 kt max total fcast",
            }
        )
        .reset_index(drop=True)
    )

    def _style_row(row):
        _styles = [""] * len(row)
        _idx = list(row.index)
        for _obsv_col, _trig_obsv, _total_col, _trig_total in [
            (
                "34 kt final obsv",
                "trig_34",
                "34 kt max total fcast",
                "trig_total_34",
            ),
            (
                "50 kt final obsv",
                "trig_50",
                "50 kt max total fcast",
                "trig_total_50",
            ),
            (
                "64 kt final obsv",
                "trig_64",
                "64 kt max total fcast",
                "trig_total_64",
            ),
        ]:
            if _trig_obsv in _idx and row[_trig_obsv] and _obsv_col in _idx:
                _styles[_idx.index(_obsv_col)] = (
                    "background-color: gold; font-weight: bold"
                )
            if _trig_total in _idx and row[_trig_total] and _total_col in _idx:
                _styles[_idx.index(_total_col)] = (
                    "background-color: #a8d8ea; font-weight: bold"
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
                c: lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "—"
                for c in [
                    "34 kt final obsv",
                    "34 kt max total fcast",
                    "50 kt final obsv",
                    "50 kt max total fcast",
                    "64 kt final obsv",
                    "64 kt max total fcast",
                ]
            }
            | {
                "Total Affected": lambda x: (
                    f"{x:,.0f}" if pd.notna(x) else "—"
                )
            }
        )
        .hide(
            axis="columns",
            subset=[
                "trig_34",
                "trig_total_34",
                "trig_50",
                "trig_total_50",
                "trig_64",
                "trig_total_64",
            ],
        )
        .hide(axis="index")
    )

    _summary = mo.md(
        f"**Return period: {_rp:.1f} yrs** ({_n} storms / {_n_years} yrs)  \n"
        f"Final obsv thresholds (gold): **34 kt** ≥ {int(_thresh[34]):,} · "
        f"**50 kt** ≥ {int(_thresh[50]):,} · "
        f"**64 kt** ≥ {int(_thresh[64]):,}  \n"
        f"Max total fcast thresholds (blue): **34 kt** ≥ {int(_thresh_total[34]):,} · "
        f"**50 kt** ≥ {int(_thresh_total[50]):,} · "
        f"**64 kt** ≥ {int(_thresh_total[64]):,} people exposed"
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
    df_total_exp,
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
            _df_oexp_ts = pd.read_sql(
                text(
                    """
                    SELECT valid_time, wind_speed_kt, pop_exposed
                    FROM storms.nhc_tracks_obsv_exposure
                    WHERE UPPER(atcf_id) = UPPER(:atcf_id)
                      AND iso3 = 'CUB' AND admin_level = 0
                    ORDER BY valid_time
                """
                ),
                _con,
                params={"atcf_id": _atcf_id},
            )
        else:
            _df_fexp_ts = pd.DataFrame(
                columns=["issued_time", "wind_speed_kt", "pop_exposed"]
            )
            _df_oexp_ts = pd.DataFrame(
                columns=["valid_time", "wind_speed_kt", "pop_exposed"]
            )
    _dev_engine.dispose()

    # Compute total exposure time series: fcast_t + cumulative_obsv_t
    _df_total_ts_list = []
    for _wt in [34, 50, 64]:
        _f = _df_fexp_ts[_df_fexp_ts["wind_speed_kt"] == _wt].sort_values(
            "issued_time"
        )
        if _f.empty:
            continue
        _o = (
            _df_oexp_ts[_df_oexp_ts["wind_speed_kt"] == _wt]
            if not _df_oexp_ts.empty
            else pd.DataFrame()
        )
        if _o.empty:
            _f = _f.copy()
            _f["total_exp"] = _f["pop_exposed"]
        else:
            _os = _o.sort_values("valid_time").assign(
                _cm=lambda x: x["pop_exposed"].cummax()
            )
            _m = pd.merge_asof(
                _f[["issued_time", "pop_exposed"]],
                _os[["valid_time", "_cm"]].rename(
                    columns={"valid_time": "issued_time"}
                ),
                on="issued_time",
                direction="backward",
            )
            _f = _f.copy()
            _f["total_exp"] = (
                _m["pop_exposed"].values + _m["_cm"].fillna(0).values
            )
        _f["wind_speed_kt"] = _wt
        _df_total_ts_list.append(
            _f[["issued_time", "wind_speed_kt", "total_exp"]]
        )
    _df_total_ts = (
        pd.concat(_df_total_ts_list, ignore_index=True)
        if _df_total_ts_list
        else pd.DataFrame(
            columns=["issued_time", "wind_speed_kt", "total_exp"]
        )
    )

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

    # ── Total exposure evolution ───────────────────────────────────────────
    # Max total exposure per wind speed for this storm (fallback reference)
    _total_by_wt = (
        df_total_exp[df_total_exp["sid"] == _sid]
        .set_index("wind_speed_kt")["max_total_exposure"]
        .to_dict()
        if _sid in df_total_exp["sid"].values
        else {}
    )

    if not _df_total_ts.empty:
        for _wt in _WIND_SPEEDS:
            _sub = _df_total_ts[
                _df_total_ts["wind_speed_kt"] == _wt
            ].sort_values("issued_time")
            if not _sub.empty:
                _ax_exp.plot(
                    _sub["issued_time"],
                    _sub["total_exp"],
                    color=_WKT_COLORS[_wt],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=f"{_wt} kt",
                )
        for _wt in _WIND_SPEEDS:
            if _wt in _exp_by_wt:
                _ax_exp.axhline(
                    _exp_by_wt[_wt],
                    color=_WKT_COLORS[_wt],
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
    elif _total_by_wt:
        for _wt in _WIND_SPEEDS:
            if _wt in _total_by_wt:
                _ax_exp.axhline(
                    _total_by_wt[_wt],
                    color=_WKT_COLORS[_wt],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=f"{_wt} kt max: {int(_total_by_wt[_wt]):,}",
                )
    else:
        _ax_exp.text(
            0.5,
            0.5,
            "No exposure data",
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
        "Total exposure evolution (fcast + cumul. observed)", fontsize=9
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
        "total_exp_34": "Total exp 34",
        "total_exp_50": "Total exp 50",
        "total_exp_64": "Total exp 64",
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
def doc_optimization(mo):
    mo.md(
        """
## Trigger optimization

Tests a simplified **OR trigger** with two indicators:

1. **Total exposure** — max(forecast exposure + cumulative observed exposure)
   at any NHC forecast issuance during the storm. Forecast and observed
   exposure come from `nhc_tracks_fcast_exposure` and
   `nhc_tracks_obsv_exposure` respectively (iso3 = CUB, admin_level = 0),
   aligned in time via `pd.merge_asof`. For historical storms not in
   those tables, `ibtracs_wind_exposure` is used as a fallback.

2. **Observed rainfall** — q80 (80th percentile) 2-day IMERG rainfall
   aggregated over Cuba during the storm period, from
   `fcast_obsv_combined_stats.parquet`.

A storm **triggers** if *either* indicator meets its threshold. Three wind
levels are tested (34 / 50 / 64 kt) as separate options, plus a
**64 kt exposure-only** option with no rainfall component.

**How thresholds are determined:** for each wind level and exposure
threshold, the rainfall threshold is set *deterministically* as the
(n − n_exp)-th largest q80 value among storms not already triggered by
exposure. This guarantees exactly n = 10 storms trigger overall (RP ≈
2.6 yrs over 2002–2025). The best option per wind level maximises CERF
storm count, then Total Affected, then minimises the exposure threshold.

**Condition columns in the storm table:**

| Column | Meaning |
|--------|---------|
| **X kt exp** | Total exposure ≥ optimised exposure threshold at this wind level |
| **X kt rain** | Observed q80 ≥ optimised rainfall threshold |
| **X kt+O** | Combined: either condition met (the actual trigger) |
| **64x exp / 64x+O** | 64 kt exposure-only option — no rainfall component |
| **Old fcast. / Old obsv.** | Historical option 1b trigger flags |
        """
    )


@app.cell
def rain_trigger_opt(df_exp, df_total_exp, df_impact, df_old_trig, mo, pd):
    _n = 10  # RP ≈ 2.6 yrs over 2002-2025

    # ── Build base data frame ─────────────────────────────────────────────  # noqa: E501
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

    _texp_pivot = (
        df_total_exp.pivot_table(
            index="sid",
            columns="wind_speed_kt",
            values="max_total_exposure",
            aggfunc="max",
        )
        .rename(
            columns={
                34: "total_exp_34",
                50: "total_exp_50",
                64: "total_exp_64",
            }
        )
        .reset_index()
    )
    for _c in ["total_exp_34", "total_exp_50", "total_exp_64"]:
        if _c not in _texp_pivot.columns:
            _texp_pivot[_c] = 0

    _meta = df_exp[["sid", "season", "name"]].drop_duplicates("sid")
    _opt = _meta.merge(_exp_pivot, on="sid", how="outer")
    _opt = _opt.merge(_texp_pivot, on="sid", how="outer")
    _opt = _opt.merge(
        df_old_trig[["sid", "q80_obsv", "fcast_trig", "obsv_trig"]],
        on="sid",
        how="outer",  # ensure all old-triggered storms are present
    )
    _opt = _opt.merge(
        df_impact[["sid", "Total Affected", "Amount in US$"]],
        on="sid",
        how="left",
    )
    _opt = _opt.drop_duplicates("sid")
    # Re-apply 2002+ filter after outer merge (old_trig parquet covers 2000+)
    _opt = _opt[_opt["sid"].str[:4].astype(int) >= 2002]
    for _c in [
        "exp_34",
        "exp_50",
        "exp_64",
        "total_exp_34",
        "total_exp_50",
        "total_exp_64",
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
    _opt["old_combined"] = _opt["fcast_trig_old"] | _opt["obsv_trig_old"]

    _cerf_lookup = dict(zip(_opt["sid"], _opt["has_cerf"]))
    _affected_lookup = dict(zip(_opt["sid"], _opt["Total Affected"].fillna(0)))

    # ── 1D sweep: total exposure OR observed rainfall ─────────────────────  # noqa: E501
    _results = []
    for _wkt, _tcol in [
        (34, "total_exp_34"),
        (50, "total_exp_50"),
        (64, "total_exp_64"),
    ]:
        _exp_vals = sorted(_opt[_tcol].dropna().unique())
        for _e_thresh in _exp_vals:
            _exp_sids = set(_opt[_opt[_tcol] >= _e_thresh]["sid"])
            _n_exp = len(_exp_sids)
            if _n_exp > _n:
                continue
            _n_rain = _n - _n_exp
            _not_exp = _opt[~_opt["sid"].isin(_exp_sids)]
            if _n_rain == 0:
                _combined_sids = frozenset(_exp_sids)
                _results.append(
                    {
                        "wkt": _wkt,
                        "exp_thresh": _e_thresh,
                        "r_obs_exact": None,
                        "r_obs": None,
                        "n_exp": _n_exp,
                        "n_rain": 0,
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
                        "_exp_sids": frozenset(_exp_sids),
                    }
                )
            else:
                _sorted = _not_exp[
                    _not_exp["max_obs_rain"].notna()
                ].sort_values("max_obs_rain", ascending=False)
                if len(_sorted) < _n_rain:
                    continue
                _r_obs = float(_sorted.iloc[_n_rain - 1]["max_obs_rain"])
                _rain_sids = set(
                    _not_exp[
                        _not_exp["max_obs_rain"].notna()
                        & (_not_exp["max_obs_rain"] >= _r_obs)
                    ]["sid"]
                )
                if len(_rain_sids) != _n_rain:
                    continue
                _combined_sids = frozenset(_exp_sids | _rain_sids)
                if len(_combined_sids) != _n:
                    continue
                _results.append(
                    {
                        "wkt": _wkt,
                        "exp_thresh": _e_thresh,
                        "r_obs_exact": _r_obs,
                        "r_obs": round(_r_obs, 1),
                        "n_exp": _n_exp,
                        "n_rain": _n_rain,
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
                        "_exp_sids": frozenset(_exp_sids),
                    }
                )

    df_rain_opt = _opt.copy()
    rain_opt_thresh = {}

    if not _results:
        mo.output.replace(mo.md("⚠ No valid combinations found."))
    else:
        _df_res = pd.DataFrame(_results)

        _df_options = (
            _df_res.sort_values(
                ["wkt", "n_exp", "cerf_count", "total_affected", "exp_thresh"],
                ascending=[True, True, False, False, True],
            )
            .groupby(["wkt", "n_exp"], sort=True)
            .first()
            .reset_index()
        )
        _df_best = (
            _df_res.sort_values(
                ["wkt", "cerf_count", "total_affected", "exp_thresh"],
                ascending=[True, False, False, True],
            )
            .groupby("wkt", sort=True)
            .first()
            .reset_index()
        )

        _best_keys = set(zip(_df_best["wkt"], _df_best["n_exp"]))
        _df_options["best"] = [
            (r["wkt"], r["n_exp"]) in _best_keys
            for _, r in _df_options.iterrows()
        ]

        for _, _brow in _df_best.iterrows():
            _wkt = int(_brow["wkt"])
            _opt[f"exp_trig_{_wkt}"] = _opt["sid"].isin(_brow["_exp_sids"])
            _opt[f"combined_{_wkt}"] = _opt["sid"].isin(
                _brow["_combined_sids"]
            )
        for _wkt in [34, 50, 64]:
            if f"exp_trig_{_wkt}" not in _opt.columns:
                _opt[f"exp_trig_{_wkt}"] = False
            if f"combined_{_wkt}" not in _opt.columns:
                _opt[f"combined_{_wkt}"] = False

        # Best pure 64kt exposure option (no rainfall component)
        _df_64exp_only = _df_res[
            (_df_res["wkt"] == 64) & (_df_res["n_rain"] == 0)
        ]
        _best_64x = None
        if not _df_64exp_only.empty:
            _best_64x = _df_64exp_only.sort_values(
                ["cerf_count", "total_affected", "exp_thresh"],
                ascending=[False, False, True],
            ).iloc[0]
            _opt["exp_trig_64x"] = _opt["sid"].isin(_best_64x["_exp_sids"])
        else:
            _opt["exp_trig_64x"] = False
        _opt["combined_64x"] = _opt["exp_trig_64x"]

        _best_thresh = {}
        for _, _r in _df_best.iterrows():
            _wkt = int(_r["wkt"])
            _best_thresh[_wkt] = {
                "exp_thresh": int(_r["exp_thresh"]),
                "r_obs": (
                    float(_r["r_obs_exact"])
                    if pd.notna(_r["r_obs_exact"])
                    else None
                ),
            }
        if _best_64x is not None:
            _best_thresh["64x"] = {
                "exp_thresh": int(_best_64x["exp_thresh"]),
                "r_obs": None,
            }
        rain_opt_thresh = _best_thresh

        _bool_hide = []
        for _wkt, _tcol in [
            (34, "total_exp_34"),
            (50, "total_exp_50"),
            (64, "total_exp_64"),
        ]:
            _t = _best_thresh.get(_wkt, {})
            _et = _t.get("exp_thresh", float("inf"))
            _ro = _t.get("r_obs")
            _opt[f"_exp_flag_{_wkt}"] = _opt[_tcol] >= _et
            _opt[f"_rain_flag_{_wkt}"] = (
                (_opt["max_obs_rain"] >= _ro)
                if _ro is not None
                else pd.Series(False, index=_opt.index)
            )
            _opt[f"{_wkt} exp"] = _opt[f"_exp_flag_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt} rain"] = _opt[f"_rain_flag_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _opt[f"{_wkt}+O"] = _opt[f"combined_{_wkt}"].map(
                {True: "✓", False: "—"}
            )
            _bool_hide += [
                f"_exp_flag_{_wkt}",
                f"_rain_flag_{_wkt}",
                f"exp_trig_{_wkt}",
                f"combined_{_wkt}",
            ]

        # 64x pure exposure condition columns
        _et64x = _best_thresh.get("64x", {}).get("exp_thresh", float("inf"))
        _opt["_exp_flag_64x"] = _opt["total_exp_64"] >= _et64x
        _opt["64x exp"] = _opt["_exp_flag_64x"].map({True: "✓", False: "—"})
        _opt["64x+O"] = _opt["combined_64x"].map({True: "✓", False: "—"})
        _bool_hide += ["_exp_flag_64x", "exp_trig_64x", "combined_64x"]

        df_rain_opt = _opt

        # ── Summary table ─────────────────────────────────────────────────  # noqa: E501
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
                "Wind kt": "ZMA ≥120/≥105",
                "Exp thresh": "—",
                "Rain thresh (q80) mm": "≥96.2",
                "# Exp": len(_old_fcast_sids),
                "# Rain": len(_old_obsv_sids),
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
                    "Exp thresh": f"{int(_r['exp_thresh']):,}",
                    "Rain thresh (q80) mm": (
                        f"{_r['r_obs']:.1f}" if pd.notna(_r["r_obs"]) else "—"
                    ),
                    "# Exp": int(_r["n_exp"]),
                    "# Rain": int(_r["n_rain"]),
                    "CERF storms": int(_r["cerf_count"]),
                    "Total Affected": int(_r["total_affected"]),
                    "RP yrs": round(_rp, 1),
                }
            )

        if _best_64x is not None:
            _summary_rows.append(
                {
                    "Trigger": "64 kt exp only ★",
                    "Wind kt": "64",
                    "Exp thresh": f"{int(_best_64x['exp_thresh']):,}",
                    "Rain thresh (q80) mm": "—",
                    "# Exp": int(_best_64x["n_exp"]),
                    "# Rain": 0,
                    "CERF storms": int(_best_64x["cerf_count"]),
                    "Total Affected": int(_best_64x["total_affected"]),
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

        # ── Options table ─────────────────────────────────────────────────  # noqa: E501
        _df_opts_disp = _df_options[
            [
                "wkt",
                "n_exp",
                "exp_thresh",
                "r_obs",
                "n_rain",
                "cerf_count",
                "total_affected",
                "best",
            ]
        ].copy()
        _df_opts_disp["★"] = _df_opts_disp["best"].map({True: "★", False: ""})
        _df_opts_disp = _df_opts_disp.rename(
            columns={
                "wkt": "Wind kt",
                "n_exp": "# Exp",
                "exp_thresh": "Exp thresh",
                "r_obs": "Rain mm",
                "n_rain": "# Rain",
                "cerf_count": "CERF",
                "total_affected": "Total Aff.",
            }
        ).drop(columns=["best"])
        _styled_opts = (
            _df_opts_disp.style.format(
                {
                    "Exp thresh": lambda x: (
                        f"{int(x):,}" if pd.notna(x) else "—"
                    ),
                    "Rain mm": lambda x: (f"{x:.1f}" if pd.notna(x) else "—"),
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

        # ── Storm conditions table ────────────────────────────────────────  # noqa: E501
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
            f"{w} {c}" for w in [34, 50, 64] for c in ["exp", "rain"]
        ] + ["64x exp"]
        _comb_cols = [f"{w}+O" for w in [34, 50, 64]] + ["64x+O"]

        _any_triggered = _opt[
            [f"combined_{w}" for w in [34, 50, 64]] + ["combined_64x"]
        ].any(axis=1)
        _show = (
            _any_triggered
            | _opt["fcast_trig_old"]
            | _opt["obsv_trig_old"]
            | (_opt["Total Affected"].fillna(0) > 0)
            | _opt["has_cerf"]
        )
        _storm_table = (
            _opt[_show][
                [
                    "Storm",
                    "34 exp",
                    "34 rain",
                    "34+O",
                    "50 exp",
                    "50 rain",
                    "50+O",
                    "64 exp",
                    "64 rain",
                    "64+O",
                    "64x exp",
                    "64x+O",
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
                return (
                    "background-color: crimson;"
                    " color: white; font-weight: bold"
                )
            if val == "—":
                return "background-color: #cce5ff; color: #555"
            return "color: #aaa"

        _styled_storms = (
            _storm_table.style.map(_sc, subset=_cond_cols)
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
                    mo.md("### All options (best per wind level + exp count)"),
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
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)

    for _col_idx, _wkt in enumerate([34, 50, 64]):
        _t = rain_opt_thresh.get(_wkt, {})
        _e_thresh = _t.get("exp_thresh", 0)
        _ro = _t.get("r_obs")
        _ax = _axes[_col_idx]
        _xcol = f"total_exp_{_wkt}"
        _sub = df_rain_opt[
            (df_rain_opt[_xcol].fillna(0) > 0)
            & df_rain_opt["max_obs_rain"].notna()
        ].copy()

        def _color_point(row, wkt=_wkt):
            if row["has_cerf"]:
                return "crimson"
            if row.get(f"combined_{wkt}", False):
                if row.get(f"exp_trig_{wkt}", False):
                    return "gold"
                return "darkorange"
            return "#aaaaaa"

        _colors = [_color_point(r) for _, r in _sub.iterrows()]
        _max_aff = df_rain_opt["Total Affected"].max()
        _sizes = [
            (
                max(
                    20,
                    (float(v) ** 0.5) * 500 / (_max_aff**0.5),
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
        if _e_thresh:
            _ax.axvline(
                _e_thresh,
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
        _ax.set_xlabel(f"Total exposure ({_wkt} kt)")
        _ax.set_ylabel("Obs rain q80 (mm)" if _col_idx == 0 else "")
        _ax.set_title(f"{_wkt} kt — total exp vs obs rain")
        _ax.grid(True, alpha=0.25, linestyle="--")
        _ax.set_xlim(left=0)
        _ax.set_ylim(bottom=0)

    _legend_patches = [
        mpatches.Patch(color="crimson", label="CERF"),
        mpatches.Patch(color="gold", label="Exp triggered"),
        mpatches.Patch(color="darkorange", label="Rain triggered"),
        mpatches.Patch(color="#aaaaaa", label="Not triggered"),
    ]
    _fig.legend(
        handles=_legend_patches,
        loc="upper center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    _fig.suptitle(
        "Total exposure vs. observed rainfall (q80) — bubble size ∝ impact",
        fontsize=11,
        y=1.05,
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def doc_leadtime(mo):
    mo.md(
        """
## Lead time analysis — 64 kt exposure-only trigger

For each storm triggered under the 64 kt exposure-only option, compares
the earliest time the trigger would have been met against when the storm
arrived at Cuba.

| Column | Definition |
|--------|-----------|
| **Trigger issued_time** | Earliest NHC forecast issuance at which total 64 kt exposure (forecast + cumulative observed) first reached the optimised threshold |
| **Arrival time (max Δ obs 64kt)** | `valid_time` of the largest single-step increase in observed 64 kt population exposure — the moment the storm was most actively sweeping through Cuba |
| **Lead time** | Arrival time − trigger issued_time. Positive = trigger fired before arrival. "no obs data" = storm is in IBTrACS only, no NHC observed exposure time series available. |
        """
    )


@app.cell
def trigger_leadtime(df_rain_opt, mo, pd, rain_opt_thresh, stratus, text):
    mo.stop(
        not len(df_rain_opt) or "64x" not in rain_opt_thresh,
        mo.md("64x option not yet computed."),
    )
    _exp_thresh = rain_opt_thresh["64x"]["exp_thresh"]
    _triggered = df_rain_opt[df_rain_opt["combined_64x"]].copy()
    mo.stop(_triggered.empty, mo.md("No storms triggered by 64x option."))

    _sids = _triggered["sid"].tolist()
    _sid_ph = ", ".join(f"'{s}'" for s in _sids)
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        _df_atcf = pd.read_sql(
            text(
                f"SELECT sid, atcf_id FROM storms.ibtracs_storms"
                f" WHERE sid IN ({_sid_ph})"
            ),
            _conn,
        )
        _atcf_list = _df_atcf["atcf_id"].tolist()
        if _atcf_list:
            _atcf_ph = ", ".join(f"'{a}'" for a in _atcf_list)
            _df_fcast_all = pd.read_sql(
                text(
                    f"""
                    SELECT atcf_id, issued_time, pop_exposed AS fcast_exp
                    FROM storms.nhc_tracks_fcast_exposure
                    WHERE atcf_id IN ({_atcf_ph})
                      AND iso3 = 'CUB' AND admin_level = 0
                      AND wind_speed_kt = 64
                    ORDER BY atcf_id, issued_time
                    """
                ),
                _conn,
            )
            _df_obsv_all = pd.read_sql(
                text(
                    f"""
                    SELECT atcf_id, valid_time, pop_exposed AS obsv_exp
                    FROM storms.nhc_tracks_obsv_exposure
                    WHERE atcf_id IN ({_atcf_ph})
                      AND iso3 = 'CUB' AND admin_level = 0
                      AND wind_speed_kt = 64
                    ORDER BY atcf_id, valid_time
                    """
                ),
                _conn,
            )
        else:
            _df_fcast_all = pd.DataFrame(
                columns=["atcf_id", "issued_time", "fcast_exp"]
            )
            _df_obsv_all = pd.DataFrame(
                columns=["atcf_id", "valid_time", "obsv_exp"]
            )
    _engine.dispose()

    _df_fcast_all = _df_fcast_all.merge(_df_atcf, on="atcf_id", how="left")
    _df_obsv_all = _df_obsv_all.merge(_df_atcf, on="atcf_id", how="left")

    _rows = []
    for _, _storm in _triggered.iterrows():
        _sid = _storm["sid"]
        _label = _storm["Storm"]
        _fcast_s = _df_fcast_all[_df_fcast_all["sid"] == _sid].sort_values(
            "issued_time"
        )
        _obsv_s = _df_obsv_all[_df_obsv_all["sid"] == _sid].sort_values(
            "valid_time"
        )

        _trigger_time = None
        _arrival_time = None
        _lead_hrs = None

        if not _fcast_s.empty:
            _fcast_s = _fcast_s.copy()
            if not _obsv_s.empty:
                _os = _obsv_s.assign(_cm=lambda x: x["obsv_exp"].cummax())
                _m = pd.merge_asof(
                    _fcast_s[["issued_time", "fcast_exp"]],
                    _os[["valid_time", "_cm"]].rename(
                        columns={"valid_time": "issued_time"}
                    ),
                    on="issued_time",
                    direction="backward",
                )
                _fcast_s["total_exp"] = (
                    _m["fcast_exp"].values + _m["_cm"].fillna(0).values
                )
            else:
                _fcast_s["total_exp"] = _fcast_s["fcast_exp"]

            _met = _fcast_s[_fcast_s["total_exp"] >= _exp_thresh]
            if not _met.empty:
                _trigger_time = _met["issued_time"].min()

        # Storm arrival = time of largest jump in observed 64kt exposure
        _arrival_time = None
        if not _obsv_s.empty and len(_obsv_s) > 1:
            _obsv_s = _obsv_s.copy()
            _obsv_s["_diff"] = _obsv_s["obsv_exp"].diff()
            _max_diff_idx = _obsv_s["_diff"].idxmax()
            if (
                _max_diff_idx is not None
                and _obsv_s.loc[_max_diff_idx, "_diff"] > 0
            ):
                _arrival_time = _obsv_s.loc[_max_diff_idx, "valid_time"]
        elif not _obsv_s.empty:
            # Only one obs record — use it directly
            _arrival_time = _obsv_s.iloc[0]["valid_time"]

        if _trigger_time is not None and _arrival_time is not None:
            _delta = pd.Timestamp(_arrival_time) - pd.Timestamp(_trigger_time)
            _lead_hrs = _delta.total_seconds() / 3600

        _rows.append(
            {
                "Storm": _label,
                "Trigger issued_time": _trigger_time,
                "Arrival time (max Δ obs 64kt)": _arrival_time,
                "Lead time": _lead_hrs,
            }
        )

    _df_lt = pd.DataFrame(_rows).sort_values(
        "Lead time", ascending=False, na_position="last"
    )

    def _fmt_time(x):
        if x is None or (hasattr(x, "__class__") and pd.isna(x)):
            return "—"
        return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")

    def _fmt_lead(x):
        if x is None or (hasattr(x, "__class__") and pd.isna(x)):
            return "no obs data"
        days = int(x) // 24
        hrs = int(x) % 24
        return f"{days}d {hrs}h" if days else f"{hrs}h"

    _styled = (
        _df_lt.style.format(
            {
                "Trigger issued_time": _fmt_time,
                "Arrival time (max Δ obs 64kt)": _fmt_time,
                "Lead time": _fmt_lead,
            }
        )
        .set_properties(**{"text-align": "center"})
        .set_properties(subset=["Storm"], **{"text-align": "left"})
        .set_table_styles(
            [{"selector": "th", "props": [("text-align", "center")]}]
        )
        .hide(axis="index")
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    f"**64 kt exposure-only trigger** — threshold:"
                    f" {_exp_thresh:,} people exposed  \n"
                    "Lead time = peak observed 64kt time − earliest trigger"
                    " issued_time"
                ),
                mo.Html(_styled.to_html()),
            ]
        )
    )


if __name__ == "__main__":
    app.run()
