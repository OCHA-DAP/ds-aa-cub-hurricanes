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
def load_wind_exposure(mo, pd, stratus, text):
    _query = text(
        """
        SELECT sid, wind_speed_kt, pop_exposed
        FROM storms.ibtracs_wind_exposure
        WHERE iso3 = 'CUB'
          AND admin_level = 0
    """
    )
    _engine = stratus.get_engine(stage="dev")
    with _engine.connect() as _conn:
        df_exp_raw = pd.read_sql(_query, _conn)
    _engine.dispose()
    mo.output.replace(
        mo.md(f"✓ Wind exposure loaded ({len(df_exp_raw):,} rows)")
    )
    return (df_exp_raw,)


@app.cell
def load_storm_meta(df_exp_raw, mo, pd, stratus, text):
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
    mo.output.replace(
        mo.md(f"✓ Storm metadata loaded ({len(_df_meta):,} storms)")
    )
    return (df_exp,)


@app.cell
def load_impact(PROJECT_PREFIX, mo, stratus):
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
    mo.output.replace(
        mo.md(f"✓ Impact data loaded ({len(df_impact):,} storms)")
    )
    return (df_impact,)


@app.cell
def load_old_trigger(PROJECT_PREFIX, mo, pd, stratus):
    _blob = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
    _df = stratus.load_parquet_from_blob(_blob)

    # Option 1b thresholds (index 13136)
    _df["fcast_trig"] = (_df["wind"] >= 120.0) & (_df["q80"] >= 35.698547)
    _df["obsv_trig"] = (_df["wind_obsv"] >= 105.0) & (
        _df["q80_obsv"] >= 96.217003
    )

    df_old_trig = _df[["sid", "fcast_trig", "obsv_trig"]].copy()

    # Melissa 2025 would have triggered both stages
    _melissa = pd.DataFrame(
        [{"sid": "2025291N11319", "fcast_trig": True, "obsv_trig": True}]
    )
    df_old_trig = pd.concat([df_old_trig, _melissa], ignore_index=True)

    mo.output.replace(
        mo.md(
            f"✓ Old trigger (option 1b) loaded ({len(df_old_trig):,} storms)"
        )
    )
    return (df_old_trig,)


@app.cell
def load_codab(load_codab_from_blob, mo):
    gdf_cub = load_codab_from_blob(admin_level=0)
    mo.output.replace(mo.md("✓ Cuba boundary loaded"))
    return (gdf_cub,)


@app.cell
def controls(mo):
    n_storms_sel = mo.ui.slider(
        start=1, stop=20, value=9, label="Storms triggered (n)"
    )
    n_storms_sel
    return (n_storms_sel,)


@app.cell
def trigger_table(df_exp, df_impact, df_old_trig, mo, n_storms_sel, pd):
    _n = n_storms_sel.value

    # One row per storm, one column per wind threshold
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

    # n-th largest exposure = threshold that selects exactly n storms
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

    # Build combined dataframe
    _meta = df_exp[["sid", "season", "name"]].drop_duplicates("sid")
    _df = _meta.merge(_exp_pivot, on="sid", how="outer")
    _df = _df.merge(df_impact, on="sid", how="outer")
    _df = _df.drop_duplicates(subset=["sid"])
    _df["season"] = _df["season"].fillna(_df["sid"].str[:4]).astype("Int64")
    if "Event Name" in _df.columns:
        _df["name"] = _df["name"].fillna(_df["Event Name"])

    # Impute known missing/updated values (pinned to exact SIDs)
    _imputes = [
        ("2025291N11319", "Total Affected", 2_200_017),  # Melissa 2025
        ("2025291N11319", "Amount in US$", 7_500_000),  # Melissa 2025 CERF
        ("2008245N17323", "Total Affected", 2_600_000),  # Ike 2008
    ]
    for _sid_val, _col, _val in _imputes:
        _df.loc[_df["sid"] == _sid_val, _col] = _val

    # Merge old trigger flags
    _df = _df.merge(df_old_trig, on="sid", how="left")
    _df["fcast_trig"] = _df["fcast_trig"].fillna(False)
    _df["obsv_trig"] = _df["obsv_trig"].fillna(False)

    # Filter: only storms with EM-DAT impact, CERF, or non-zero exposure
    _has_emdat = _df["Total Affected"].notna()
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
    _storm_map = dict(zip(_storms["label"], _storms["sid"]))

    _default_melissa = "2025291N11319"
    _default_key = next(
        (k for k, v in _storm_map.items() if v == _default_melissa), None
    )

    storm_sel = mo.ui.dropdown(
        options=_storm_map, value=_default_key, label="Select storm for map"
    )
    storm_sel
    return (storm_sel,)


@app.cell
def storm_map(
    df_exp,
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
    _exp_by_wt = (
        _row[_row["pop_exposed"] > 0]
        .set_index("wind_speed_kt")["pop_exposed"]
        .to_dict()
    )

    # Colors derived from YlOrRd colormap — same approach as reference app
    _cmap = plt.get_cmap("YlOrRd")
    _norm = plt.Normalize(vmin=20, vmax=80)
    _WIND_SPEEDS = [34, 50, 64]

    _fig, _ax = plt.subplots(figsize=(10, 4), dpi=150)
    gdf_cub.boundary.plot(ax=_ax, linewidth=0.8, color="k")

    # Plot largest buffer first (background) to smallest (foreground)
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
                ax=_ax,
                alpha=0.4,
                color=_color,
                edgecolor=_color,
                linewidth=0.5,
            )
            _exp_str = (
                f" — {int(_exp_by_wt[_wt]):,} exposed"
                if _wt in _exp_by_wt
                else ""
            )
            _patches.append(
                mpatches.Patch(color=_color, label=f"{_wt} kt{_exp_str}")
            )

    if _patches:
        _ax.legend(handles=_patches[::-1], loc="lower left", fontsize=8)
    else:
        _ax.text(
            0.5,
            0.5,
            "No wind buffer data available",
            transform=_ax.transAxes,
            ha="center",
            va="center",
            color="grey",
        )

    _minx, _miny, _maxx, _maxy = gdf_cub.total_bounds
    _pad = 3
    _ax.set_xlim(_minx - _pad, _maxx + _pad)
    _ax.set_ylim(_miny - _pad, _maxy + _pad)
    _ax.set_title(f"{_name} ({_yr})" if _name else _sid[:4])
    _ax.set_axis_off()
    plt.tight_layout()

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
        ][["Storm", _col, "Total Affected", _trig_col]].copy()

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
        _ax.set_title(f"{_label} exposure vs. Total Affected")
        _ax.grid(True, alpha=0.3, linestyle="--")
        if _ax is _axes[0]:
            _ax.set_ylabel("Total Affected")

    _fig.suptitle("Wind exposure vs. EM-DAT Total Affected", y=1.01)
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
