#!/usr/bin/env python3
"""
Generate an interactive HTML dashboard from SpatialCorpus-110M_metadata.csv.
Can be run standalone or imported and called as generate().
"""

import json
from pathlib import Path

import pandas as pd

CSV_PATH  = Path("/Volumes/processing/SpatialCorpus-110M_metadata.csv")
HTML_PATH = Path("/Users/christoffer/work/karolinska/development/KaroSpaceDataWrangling/spatial_corpus_dashboard.html")

ASSAY_COLORS = {
    "Xenium":                           "#4f8ef7",
    "MERFISH":                          "#f97b4f",
    "10x 3' v2":                        "#4fc47e",
    "10x 3' v3":                        "#a78bfa",
    "10x 5' v2":                        "#f472b6",
    "10x transcription profiling":      "#34d399",
    "10x 3' transcription profiling":   "#60a5fa",
    "10x 5' transcription profiling":   "#fb923c",
    "10x 3' v2; 10x 3' v3":            "#94a3b8",
}
DEFAULT_COLOR = "#94a3b8"

SPATIAL_ASSAYS = {
    "xenium", "merfish", "seqfish", "starmap", "codex", "visium",
    "slideseq", "slide-seq", "stereo-seq", "stereoseq", "resolve",
    "cosmx", "merscope", "10x xenium",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in ("n_cells", "n_vars", "n_donors", "n_conditions", "n_tissue_types", "n_sections"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    return df


# ---------------------------------------------------------------------------
# KaroSpace suitability score  (0–5)
# Donors are the primary axis — KaroSpace is built to harness multi-sample data.
# ---------------------------------------------------------------------------

def _score(row: pd.Series) -> int:
    n_donors = int(row.get("n_donors", 0))
    assay_lower = str(row.get("assay", "")).lower()
    is_spatial = any(s in assay_lower for s in SPATIAL_ASSAYS)

    if n_donors >= 10:
        score = 3
    elif n_donors >= 5:
        score = 2
    elif n_donors >= 2:
        score = 1
    else:
        score = 0

    if is_spatial:
        score += 1
    if str(row.get("publication_doi", "")).strip():
        score += 1

    return min(score, 5)


_SCORE_LABEL = {
    5: ("Excellent", "#4fc47e"),
    4: ("Very good", "#86efac"),
    3: ("Good",      "#fbbf24"),
    2: ("Moderate",  "#f97b4f"),
    1: ("Limited",   "#94a3b8"),
    0: ("Limited",   "#64748b"),
}


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------

def build_chart_data(df: pd.DataFrame) -> dict:
    by_assay = df.groupby("assay")["n_cells"].sum().sort_values(ascending=False)
    by_tissue = (
        df.assign(t0=df["tissue"].str.split(";").str[0].str.strip())
        .groupby("t0")["n_cells"].sum()
        .sort_values(ascending=False)
        .head(20)
    )
    by_organism = df.groupby("organism")["n_cells"].sum().sort_values(ascending=False)

    return {
        "assay": {
            "labels": by_assay.index.tolist(),
            "values": by_assay.values.tolist(),
            "colors": [ASSAY_COLORS.get(a, DEFAULT_COLOR) for a in by_assay.index],
        },
        "tissue": {
            "labels": by_tissue.index.tolist(),
            "values": by_tissue.values.tolist(),
        },
        "organism": {
            "labels": by_organism.index.tolist(),
            "values": by_organism.values.tolist(),
            "colors": ["#4f8ef7", "#f97b4f"],
        },
    }


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------

def build_table_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.sort_values("n_cells", ascending=False).iterrows():
        doi       = r.get("publication_doi", "") or ""
        pub_title = r.get("publication_title", "") or ""
        pub_html  = (
            f'<a href="https://doi.org/{doi}" target="_blank" title="{pub_title}">'
            f'{doi}</a>'
            if doi else ""
        )
        assay_color = ASSAY_COLORS.get(str(r.get("assay", "")), DEFAULT_COLOR)
        sc = _score(r)
        sc_label, sc_color = _SCORE_LABEL[sc]
        is_spatial = any(s in str(r.get("assay", "")).lower() for s in SPATIAL_ASSAYS)

        stem = r["filename"].replace(".h5ad", "")
        rows.append({
            "filename":       stem,
            "filepath":       f"/Volumes/processing/SpatialCorpus-110M/{stem}.h5ad",
            "n_cells":        int(r["n_cells"]),
            "n_vars":         int(r["n_vars"]),
            "n_donors":       int(r.get("n_donors", 0)),
            "n_conditions":   int(r.get("n_conditions", 0)),
            "n_tissue_types": int(r.get("n_tissue_types", 0)),
            "n_sections":     int(r.get("n_sections", 0)),
            "organism":       r.get("organism", "") or "",
            "assay":          r.get("assay", "") or "",
            "assay_color":    assay_color,
            "tissue":         r.get("tissue", "") or "",
            "sex":            r.get("sex", "") or "",
            "conditions":     r.get("condition_ids", "") or "",
            "pub_html":       pub_html,
            "pub_title":      pub_title,
            "score":          sc,
            "score_label":    sc_label,
            "score_color":    sc_color,
            "is_spatial":     is_spatial,
        })
    return rows


# ---------------------------------------------------------------------------
# HTML  (uses __PLACEHOLDER__ tokens instead of {format} to avoid escaping JS)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpatialCorpus-110M — Dataset Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<style>
:root {
  --bg:#0f1117; --surface:#1a1d27; --card:#1e2235; --border:#2a2f4a;
  --accent:#4f8ef7; --text:#e2e8f0; --muted:#7a859a; --radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}

header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 26px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
header h1{font-size:16px;font-weight:600;letter-spacing:-.3px}
.updated{color:var(--muted);font-size:12px}
#btnReload{background:none;border:1px solid var(--border);color:var(--muted);padding:4px 11px;border-radius:6px;cursor:pointer;font-size:12px}
#btnReload:hover{border-color:var(--accent);color:var(--accent)}

.main{padding:18px 26px;max-width:1900px;margin:0 auto}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:9px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:13px 15px}
.card .val{font-size:22px;font-weight:700;color:var(--accent);line-height:1.1}
.card .lbl{color:var(--muted);font-size:10px;margin-top:3px;text-transform:uppercase;letter-spacing:.5px}

/* Charts */
.charts-top{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-bottom:11px}
.charts-bot{display:grid;grid-template-columns:1fr 260px;gap:11px;margin-bottom:18px}
.chart-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:13px 15px}
.chart-box.clickable{cursor:pointer}
.chart-box.clickable:hover{border-color:rgba(79,142,247,.4)}
.ch{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
.ch h2{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.ch-hint{font-size:10px;color:#3d4560}
.cw{position:relative;height:230px} .cw-tall{position:relative;height:290px}

/* Chips */
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px;min-height:22px}
.chip{display:inline-flex;align-items:center;gap:4px;background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.3);border-radius:20px;padding:2px 9px;font-size:11px;color:var(--accent)}
.chip button{background:none;border:none;color:var(--accent);cursor:pointer;font-size:13px;line-height:1;padding:0;opacity:.6}
.chip button:hover{opacity:1}

/* Controls */
.controls{display:flex;gap:7px;margin-bottom:9px;flex-wrap:wrap;align-items:center}
.controls input,.controls select{background:var(--card);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:13px;outline:none}
.controls input{flex:1;min-width:160px}
.controls input:focus,.controls select:focus{border-color:var(--accent)}
.cnt{color:var(--muted);font-size:12px;white-space:nowrap;margin-left:auto}

/* Sliders */
.sliders{display:flex;gap:18px;margin-bottom:13px;flex-wrap:wrap}
.sg{display:flex;flex-direction:column;gap:4px;min-width:190px}
.sg label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;display:flex;justify-content:space-between}
.sg label span{color:var(--text);font-weight:600}
input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer}

/* Legend */
.legend{display:flex;gap:11px;flex-wrap:wrap;align-items:center;margin-bottom:13px}
.li{display:flex;align-items:center;gap:4px;font-size:11px;color:var(--muted)}
.ld{width:8px;height:8px;border-radius:50%;flex-shrink:0}

/* Table */
.tw{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse}
thead{background:var(--surface);position:sticky;top:0;z-index:2}
th{padding:7px 10px;text-align:left;font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;cursor:pointer;user-select:none;white-space:nowrap;border-bottom:1px solid var(--border)}
th:hover{color:var(--text)}
th .si{opacity:.3;margin-left:3px;font-size:10px}
th.asc .si::after{content:"↑";opacity:1;color:var(--accent)}
th.desc .si::after{content:"↓";opacity:1;color:var(--accent)}
th:not(.asc):not(.desc) .si::after{content:"↕"}
tbody tr{border-top:1px solid var(--border);transition:background .1s;cursor:pointer}
tbody tr:hover{background:rgba(79,142,247,.05)}
tbody tr.active{background:rgba(79,142,247,.1)!important;outline:1px solid rgba(79,142,247,.35)}
td{padding:6px 10px;vertical-align:middle}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
td.sm{color:var(--muted);font-size:12px}
td.clip{max-width:190px;font-size:12px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:500;white-space:nowrap;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1)}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(0,0,0,.2);border:1px solid;white-space:nowrap}
.sdot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.mono{font-family:"SF Mono","Fira Code",monospace;font-size:12px}
.no-res{text-align:center;padding:44px;color:var(--muted)}
.cp-btn{background:none;border:none;color:#3d4560;cursor:pointer;font-size:12px;padding:0 3px;vertical-align:middle}
.cp-btn:hover{color:var(--accent)}
.mbw{display:flex;align-items:center;gap:5px}
.mb{height:4px;border-radius:2px;min-width:2px}

/* Drawer */
.ov{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100;opacity:0;pointer-events:none;transition:opacity .2s}
.ov.open{opacity:1;pointer-events:all}
.drawer{position:fixed;right:0;top:0;bottom:0;width:400px;max-width:95vw;background:var(--surface);border-left:1px solid var(--border);z-index:101;transform:translateX(100%);transition:transform .25s cubic-bezier(.4,0,.2,1);overflow-y:auto;display:flex;flex-direction:column}
.drawer.open{transform:translateX(0)}
.dh{padding:16px 18px 12px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;gap:10px;position:sticky;top:0;background:var(--surface);z-index:1}
.dt{font-size:13px;font-weight:600;font-family:"SF Mono","Fira Code",monospace;word-break:break-all}
.dc{background:none;border:1px solid var(--border);color:var(--muted);width:26px;height:26px;border-radius:6px;cursor:pointer;font-size:15px;flex-shrink:0;display:flex;align-items:center;justify-content:center}
.dc:hover{color:var(--text);border-color:var(--text)}
.db{padding:16px 18px;flex:1;display:flex;flex-direction:column;gap:14px}
.ds-title{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);font-weight:600;margin-bottom:4px}
.dgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.dstat{background:var(--card);border:1px solid var(--border);border-radius:7px;padding:9px 11px}
.dstat .dv{font-size:19px;font-weight:700}
.dstat .dl{font-size:11px;color:var(--muted);margin-top:2px}
.drow{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:5px 0;border-bottom:1px solid var(--border)}
.drow:last-child{border-bottom:none}
.dk{color:var(--muted);font-size:12px;flex-shrink:0}
.dvt{font-size:12px;text-align:right;word-break:break-word}

@media(max-width:1100px){.charts-top,.charts-bot{grid-template-columns:1fr}}
@media(max-width:700px){.main{padding:11px}}
</style>
</head>
<body>

<header>
  <h1>SpatialCorpus-110M &mdash; Dataset Explorer</h1>
  <div style="display:flex;align-items:center;gap:14px">
    <span class="updated">Updated __UPDATED__ &nbsp;·&nbsp; __N_FILES__ datasets &nbsp;·&nbsp; __N_CELLS__ cells</span>
    <button id="btnReload" onclick="location.reload()">↻ Refresh</button>
  </div>
</header>

<div class="main">

  <div class="cards">
    <div class="card"><div class="val">__N_FILES__</div><div class="lbl">Datasets</div></div>
    <div class="card"><div class="val">__N_CELLS__</div><div class="lbl">Total cells</div></div>
    <div class="card"><div class="val">__N_HUMAN__</div><div class="lbl">Human</div></div>
    <div class="card"><div class="val">__N_MOUSE__</div><div class="lbl">Mouse</div></div>
    <div class="card"><div class="val">__N_SPATIAL__</div><div class="lbl">Spatial assays</div></div>
    <div class="card"><div class="val">__N_ASSAYS__</div><div class="lbl">Assay types</div></div>
    <div class="card"><div class="val">__N_TISSUES__</div><div class="lbl">Tissues</div></div>
    <div class="card"><div class="val" id="cShowing">__N_FILES__</div><div class="lbl">Showing</div></div>
  </div>

  <div class="charts-top">
    <div class="chart-box clickable">
      <div class="ch"><h2>Cells by assay</h2><span class="ch-hint">click bar → filter</span></div>
      <div class="cw"><canvas id="cAssay"></canvas></div>
    </div>
    <div class="chart-box clickable">
      <div class="ch"><h2>Cells by tissue (top 20)</h2><span class="ch-hint">click bar → filter</span></div>
      <div class="cw"><canvas id="cTissue"></canvas></div>
    </div>
  </div>

  <div class="charts-bot">
    <div class="chart-box clickable">
      <div class="ch">
        <h2>Donors vs Cells — KaroSpace landscape</h2>
        <span class="ch-hint">click dot · scroll zoom · drag pan</span>
      </div>
      <div class="cw-tall"><canvas id="cScatter"></canvas></div>
    </div>
    <div class="chart-box">
      <div class="ch"><h2>Cells by organism</h2></div>
      <div class="cw"><canvas id="cOrg"></canvas></div>
    </div>
  </div>

  <div class="legend">
    <span style="font-size:11px;color:var(--muted);margin-right:2px">KaroSpace fit:</span>
    <div class="li"><div class="ld" style="background:#4fc47e"></div>Excellent (≥10 donors + spatial + DOI)</div>
    <div class="li"><div class="ld" style="background:#86efac"></div>Very good (≥10 donors + DOI or spatial)</div>
    <div class="li"><div class="ld" style="background:#fbbf24"></div>Good (≥2 donors + bonuses)</div>
    <div class="li"><div class="ld" style="background:#f97b4f"></div>Moderate</div>
    <div class="li"><div class="ld" style="background:#94a3b8"></div>Limited (1 sample)</div>
  </div>

  <div class="chips" id="chips"></div>

  <div class="controls">
    <input type="text" id="search" placeholder="Search dataset, tissue, assay, condition…">
    <select id="fOrg"><option value="">All organisms</option></select>
    <select id="fAssay"><option value="">All assays</option></select>
    <select id="fSpatial">
      <option value="">Spatial + non-spatial</option>
      <option value="1">Spatial only</option>
      <option value="0">Non-spatial only</option>
    </select>
    <select id="fScore">
      <option value="">All KaroSpace fits</option>
      <option value="5">Excellent (5)</option>
      <option value="4">Very good (4+)</option>
      <option value="3">Good (3+)</option>
      <option value="2">Moderate (2+)</option>
    </select>
    <span class="cnt" id="rowCnt"></span>
  </div>

  <div class="sliders">
    <div class="sg">
      <label>Min donors <span id="dVal">1</span></label>
      <input type="range" id="sDonors" min="1" max="1" value="1">
    </div>
    <div class="sg">
      <label>Min cells <span id="cVal">0</span></label>
      <input type="range" id="sCells" min="0" max="1" value="0">
    </div>
  </div>

  <div class="tw">
    <table>
      <thead><tr>
        <th data-col="score" class="desc">Fit<span class="si"></span></th>
        <th data-col="is_spatial">Spatial<span class="si"></span></th>
        <th data-col="filename">Dataset<span class="si"></span></th>
        <th data-col="n_cells">Cells<span class="si"></span></th>
        <th data-col="n_donors">Donors<span class="si"></span></th>
        <th data-col="n_conditions">Cond.<span class="si"></span></th>
        <th data-col="n_tissue_types">Tissues<span class="si"></span></th>
        <th data-col="n_sections">Sections<span class="si"></span></th>
        <th data-col="n_vars">Genes<span class="si"></span></th>
        <th data-col="organism">Organism<span class="si"></span></th>
        <th data-col="assay">Assay<span class="si"></span></th>
        <th data-col="tissue">Tissue<span class="si"></span></th>
        <th data-col="conditions">Conditions<span class="si"></span></th>
        <th>Publication</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="no-res" id="noRes" style="display:none">No datasets match your filters.</div>
  </div>
</div>

<!-- Drawer -->
<div class="ov" id="ov"></div>
<div class="drawer" id="drawer">
  <div class="dh">
    <div class="dt" id="dTitle"></div>
    <button class="dc" id="dClose">✕</button>
  </div>
  <div class="db" id="dBody"></div>
</div>

<script>
const ROWS = __ROWS_JSON__;
const CD   = __CHART_JSON__;
const MAX_CELLS  = Math.max(...ROWS.map(r=>r.n_cells));
const MAX_DONORS = Math.max(...ROWS.map(r=>r.n_donors));

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = n => n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":String(n);

function mbar(val, max, color) {
  const w = max>0 ? Math.max(3,Math.round((val/max)*60)) : 0;
  return `<div class="mbw"><div class="mb" style="width:${w}px;background:${color||"#4f8ef7"}"></div><span>${fmt(val)}</span></div>`;
}
function dBar(n) {
  if(!n) return '<span style="color:#4a5470">—</span>';
  const c = n>=10?"#4fc47e":n>=5?"#86efac":n>=2?"#fbbf24":"#94a3b8";
  const w = Math.max(3,Math.round((n/MAX_DONORS)*60));
  return `<div class="mbw"><div class="mb" style="width:${w}px;background:${c}"></div><span style="color:${c};font-weight:600">${n}</span></div>`;
}
function cBadge(n) {
  if(!n) return '<span style="color:#4a5470">—</span>';
  const c = n>=10?"#4fc47e":n>=5?"#86efac":n>=3?"#fbbf24":"#94a3b8";
  return `<span style="color:${c};font-weight:600">${n}</span>`;
}

// ── Slider init ───────────────────────────────────────────────────────────────
const sDonors = document.getElementById("sDonors");
const sCells  = document.getElementById("sCells");
sDonors.max = MAX_DONORS;
sCells.max  = MAX_CELLS;
sCells.step = Math.max(1000, Math.round(MAX_CELLS/100));
sDonors.addEventListener("input",()=>{ document.getElementById("dVal").textContent=sDonors.value; render(); });
sCells.addEventListener("input", ()=>{ document.getElementById("cVal").textContent=fmt(+sCells.value); render(); });

// ── Populate selects ──────────────────────────────────────────────────────────
function populate(id, vals) {
  const s = document.getElementById(id);
  [...new Set(vals)].sort().forEach(v=>{ const o=document.createElement("option"); o.value=o.textContent=v; s.appendChild(o); });
}
populate("fOrg",   ROWS.map(r=>r.organism).filter(Boolean));
populate("fAssay", ROWS.map(r=>r.assay).filter(Boolean));

// ── Chart colour lookup ────────────────────────────────────────────────────────
const ACMAP = {};
CD.assay.labels.forEach((l,i)=>ACMAP[l]=CD.assay.colors[i]);

// ── Charts ────────────────────────────────────────────────────────────────────
const GRID = {color:"#222640"};
const TICK = {color:"#6b7591",font:{size:11}};
const YCB  = v=>v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(0)+"k":v;
const HOPTS = {
  maintainAspectRatio:false, indexAxis:"y",
  plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>" "+fmt(c.parsed.x)+" cells"}}},
  scales:{x:{ticks:{...TICK,callback:YCB},grid:GRID}, y:{ticks:TICK,grid:GRID}},
};

// Assay bar — click filters
new Chart(document.getElementById("cAssay"),{
  type:"bar",
  data:{labels:CD.assay.labels, datasets:[{data:CD.assay.values,backgroundColor:CD.assay.colors,borderRadius:4}]},
  options:{...HOPTS, onClick(_,els){
    if(!els.length) return;
    const v=CD.assay.labels[els[0].index];
    const s=document.getElementById("fAssay");
    s.value = s.value===v ? "" : v;
    render();
  }},
});

// Tissue bar — click sets search
new Chart(document.getElementById("cTissue"),{
  type:"bar",
  data:{labels:CD.tissue.labels, datasets:[{data:CD.tissue.values,backgroundColor:"#4f8ef7",borderRadius:4}]},
  options:{...HOPTS, onClick(_,els){
    if(!els.length) return;
    const v=CD.tissue.labels[els[0].index];
    const s=document.getElementById("search");
    s.value = s.value===v ? "" : v;
    render();
  }},
});

// Organism doughnut
new Chart(document.getElementById("cOrg"),{
  type:"doughnut",
  data:{labels:CD.organism.labels, datasets:[{data:CD.organism.values,backgroundColor:CD.organism.colors,borderWidth:0,hoverOffset:6}]},
  options:{maintainAspectRatio:false, plugins:{legend:{display:true,position:"bottom",labels:{color:"#8892a4",font:{size:12},padding:12}}}},
});

// Scatter: donors vs cells — click opens drawer
const scatterData = ROWS.map(r=>({ x:Math.max(r.n_donors,0.5), y:r.n_cells, r:Math.max(4,r.score*2.5), row:r }));
new Chart(document.getElementById("cScatter"),{
  type:"bubble",
  data:{datasets:[{
    data:scatterData,
    backgroundColor:scatterData.map(d=>(ACMAP[d.row.assay]||"#4f8ef7")+"bb"),
    borderColor:scatterData.map(d=>ACMAP[d.row.assay]||"#4f8ef7"),
    borderWidth:1,
  }]},
  options:{
    maintainAspectRatio:false,
    plugins:{
      legend:{display:false},
      zoom:{zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:"xy"}, pan:{enabled:true,mode:"xy"}},
      tooltip:{callbacks:{
        title:c=>c[0].raw.row.filename,
        label:c=>[`  Donors: ${c.raw.row.n_donors}`,`  Cells: ${fmt(c.raw.row.n_cells)}`,`  Assay: ${c.raw.row.assay}`,`  Fit: ${c.raw.row.score_label}`],
      }},
    },
    scales:{
      x:{type:"logarithmic", title:{display:true,text:"Number of donors",color:"#6b7591",font:{size:11}},
         ticks:{...TICK,callback:v=>[1,2,5,10,20,50].includes(v)?v:""}, grid:GRID},
      y:{type:"logarithmic", title:{display:true,text:"Number of cells",color:"#6b7591",font:{size:11}},
         ticks:{...TICK,callback:YCB}, grid:GRID},
    },
    onClick(_,els){ if(els.length) openDrawer(els[0].element.$context.raw.row); },
  },
});

// ── Chips ─────────────────────────────────────────────────────────────────────
function renderChips(){
  const el=document.getElementById("chips"); el.innerHTML="";
  const add=(lbl,clear)=>{
    const c=document.createElement("div"); c.className="chip";
    c.innerHTML=`${lbl}<button>×</button>`; c.querySelector("button").onclick=clear; el.appendChild(c);
  };
  const q=document.getElementById("search").value;
  if(q)                                       add(`"${q}"`,()=>{document.getElementById("search").value="";render();});
  if(document.getElementById("fOrg").value)   add(document.getElementById("fOrg").value,()=>{document.getElementById("fOrg").value="";render();});
  if(document.getElementById("fAssay").value)   add(document.getElementById("fAssay").value,()=>{document.getElementById("fAssay").value="";render();});
  if(document.getElementById("fSpatial").value) add(document.getElementById("fSpatial").value==="1"?"Spatial only":"Non-spatial only",()=>{document.getElementById("fSpatial").value="";render();});
  if(document.getElementById("fScore").value)   add("Fit ≥ "+document.getElementById("fScore").value,()=>{document.getElementById("fScore").value="";render();});
  if(+sDonors.value>1) add(`≥${sDonors.value} donors`,()=>{sDonors.value=1;document.getElementById("dVal").textContent="1";render();});
  if(+sCells.value >0) add(`≥${fmt(+sCells.value)} cells`,()=>{sCells.value=0;document.getElementById("cVal").textContent="0";render();});
}

// ── Table ─────────────────────────────────────────────────────────────────────
let sortCol="score", sortDir=-1, activeRow=null;

function render(){
  const q       = document.getElementById("search").value.toLowerCase();
  const org     = document.getElementById("fOrg").value;
  const assay   = document.getElementById("fAssay").value;
  const spatial = document.getElementById("fSpatial").value;
  const score   = document.getElementById("fScore").value;
  const minD    = +sDonors.value;
  const minC    = +sCells.value;

  let rows = ROWS.filter(r=>{
    if(org     && r.organism!==org)              return false;
    if(assay   && r.assay!==assay)               return false;
    if(spatial==="1" && !r.is_spatial)           return false;
    if(spatial==="0" &&  r.is_spatial)           return false;
    if(score   && r.score< +score)               return false;
    if(r.n_donors<minD)                          return false;
    if(r.n_cells <minC)                          return false;
    if(q){ const h=[r.filename,r.tissue,r.assay,r.organism,r.conditions,r.pub_title].join(" ").toLowerCase(); if(!h.includes(q)) return false; }
    return true;
  });

  rows.sort((a,b)=>{
    const av=a[sortCol],bv=b[sortCol];
    return (typeof av==="number"?av-bv:String(av).localeCompare(String(bv)))*sortDir;
  });

  const tbody=document.getElementById("tbody"); tbody.innerHTML="";
  rows.forEach(r=>{
    const tr=document.createElement("tr");
    if(activeRow?.filename===r.filename) tr.classList.add("active");
    tr.innerHTML=`
      <td><span class="pill" style="color:${r.score_color};border-color:${r.score_color}55"><span class="sdot" style="background:${r.score_color}"></span>${r.score_label}</span></td>
      <td style="text-align:center">${r.is_spatial
        ? '<span title="Spatial" style="color:#4f8ef7;font-size:15px">⬡</span>'
        : '<span title="Non-spatial" style="color:#3d4560;font-size:13px">—</span>'}</td>
      <td class="mono">${r.filename} <button class="cp-btn" title="${r.filepath}" onclick="event.stopPropagation();navigator.clipboard.writeText('${r.filepath}');this.textContent='✓';setTimeout(()=>this.textContent='⎘',1200)">⎘</button></td>
      <td class="num">${mbar(r.n_cells,MAX_CELLS,"#4f8ef7")}</td>
      <td class="num">${dBar(r.n_donors)}</td>
      <td class="num">${cBadge(r.n_conditions)}</td>
      <td class="num">${cBadge(r.n_tissue_types)}</td>
      <td class="num sm">${r.n_sections||"—"}</td>
      <td class="num sm">${r.n_vars.toLocaleString()}</td>
      <td class="sm">${r.organism}</td>
      <td><span class="badge" style="color:${r.assay_color};border-color:${r.assay_color}44">${r.assay}</span></td>
      <td class="clip" title="${r.tissue}">${r.tissue}</td>
      <td class="clip" title="${r.conditions}">${r.conditions}</td>
      <td style="font-size:12px">${r.pub_html}</td>`;
    tr.addEventListener("click",e=>{ if(e.target.tagName==="A") return; openDrawer(r); });
    tbody.appendChild(tr);
  });

  document.getElementById("rowCnt").textContent=`${rows.length} of ${ROWS.length}`;
  document.getElementById("cShowing").textContent=rows.length;
  document.getElementById("noRes").style.display=rows.length?"none":"block";
  renderChips();
}

// ── Detail drawer ─────────────────────────────────────────────────────────────
function openDrawer(r){
  activeRow=r;
  document.getElementById("dTitle").textContent=r.filename;
  const sc=r.score_color;
  const drows=([
    ["Assay",    `<span class="badge" style="color:${r.assay_color};border-color:${r.assay_color}44">${r.assay}</span>`],
    ["Organism", r.organism],
    ["Tissue",   r.tissue||"—"],
    ["Sex",      r.sex||"—"],
    ["Conditions", r.conditions||"—"],
  ]).map(([k,v])=>`<div class="drow"><span class="dk">${k}</span><span class="dvt">${v}</span></div>`).join("");

  document.getElementById("dBody").innerHTML=`
    <div>
      <span class="pill" style="color:${sc};border-color:${sc}55;font-size:13px;padding:4px 12px">
        <span class="sdot" style="background:${sc};width:8px;height:8px"></span>
        ${r.score_label} &nbsp;(${r.score}/5)
      </span>
    </div>

    <div>
      <div class="ds-title">Sample structure</div>
      <div class="dgrid">
        <div class="dstat"><div class="dv" style="color:${r.n_donors>=10?"#4fc47e":r.n_donors>=5?"#86efac":r.n_donors>=2?"#fbbf24":"#94a3b8"}">${r.n_donors||"—"}</div><div class="dl">Donors / samples</div></div>
        <div class="dstat"><div class="dv">${r.n_conditions||"—"}</div><div class="dl">Conditions</div></div>
        <div class="dstat"><div class="dv">${r.n_tissue_types||"—"}</div><div class="dl">Tissue types</div></div>
        <div class="dstat"><div class="dv">${r.n_sections||"—"}</div><div class="dl">Sections</div></div>
      </div>
    </div>

    <div>
      <div class="ds-title">Scale</div>
      <div class="dgrid">
        <div class="dstat"><div class="dv">${fmt(r.n_cells)}</div><div class="dl">Cells</div></div>
        <div class="dstat"><div class="dv">${r.n_vars.toLocaleString()}</div><div class="dl">Genes</div></div>
      </div>
    </div>

    <div>
      <div class="ds-title">File path</div>
      <div style="display:flex;align-items:center;gap:8px;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:8px 10px">
        <code style="font-size:11px;color:var(--muted);flex:1;word-break:break-all">${r.filepath}</code>
        <button onclick="navigator.clipboard.writeText(this.dataset.path);this.textContent='✓';setTimeout(()=>this.textContent='Copy',1200)" data-path="${r.filepath}" style="background:none;border:1px solid var(--border);color:var(--muted);padding:3px 8px;border-radius:5px;cursor:pointer;font-size:11px;flex-shrink:0">Copy</button>
      </div>
    </div>
    <div><div class="ds-title">Metadata</div>${drows}</div>

    ${r.pub_html?`
    <div>
      <div class="ds-title">Publication</div>
      ${r.pub_title?`<div style="font-size:13px;line-height:1.5;margin-bottom:6px">${r.pub_title}</div>`:""}
      <div style="font-size:13px">${r.pub_html}</div>
    </div>`:""}
  `;
  document.getElementById("drawer").classList.add("open");
  document.getElementById("ov").classList.add("open");
  render();
}

function closeDrawer(){
  activeRow=null;
  document.getElementById("drawer").classList.remove("open");
  document.getElementById("ov").classList.remove("open");
  render();
}
document.getElementById("dClose").addEventListener("click",closeDrawer);
document.getElementById("ov").addEventListener("click",closeDrawer);
document.addEventListener("keydown",e=>{ if(e.key==="Escape") closeDrawer(); });

// ── Sort headers ──────────────────────────────────────────────────────────────
document.querySelectorAll("th[data-col]").forEach(th=>{
  th.addEventListener("click",()=>{
    const col=th.dataset.col;
    sortDir=sortCol===col?sortDir*-1:-1; sortCol=col;
    document.querySelectorAll("th").forEach(h=>h.classList.remove("asc","desc"));
    th.classList.add(sortDir===-1?"desc":"asc");
    render();
  });
});
["search","fOrg","fAssay","fSpatial","fScore"].forEach(id=>document.getElementById(id).addEventListener("input",render));

render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate() -> None:
    df = load_data()
    chart_data  = build_chart_data(df)
    table_rows  = build_table_rows(df)

    n_cells_total = int(df["n_cells"].sum())
    n_cells_fmt   = f"{n_cells_total/1e6:.1f}M" if n_cells_total >= 1e6 else f"{n_cells_total:,}"
    n_tissues     = df["tissue"].str.split(";").explode().str.strip().dropna().nunique()
    n_assays      = df["assay"].str.split(";").explode().str.strip().dropna().nunique()
    n_human       = int((df["organism"] == "Homo sapiens").sum())
    n_mouse       = int((df["organism"] == "Mus musculus").sum())
    n_excellent   = sum(1 for r in table_rows if r["score"] >= 4)
    n_spatial     = sum(1 for r in table_rows if any(
        s in str(r.get("assay", "")).lower() for s in SPATIAL_ASSAYS
    ))

    html = HTML_TEMPLATE \
        .replace("__UPDATED__",     pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")) \
        .replace("__N_FILES__",     str(len(df))) \
        .replace("__N_CELLS__",     n_cells_fmt) \
        .replace("__N_HUMAN__",     str(n_human)) \
        .replace("__N_MOUSE__",     str(n_mouse)) \
        .replace("__N_SPATIAL__",   str(n_spatial)) \
        .replace("__N_ASSAYS__",    str(n_assays)) \
        .replace("__N_TISSUES__",   str(n_tissues)) \
        .replace("__N_EXCELLENT__", str(n_excellent)) \
        .replace("__ROWS_JSON__",   json.dumps(table_rows, ensure_ascii=False)) \
        .replace("__CHART_JSON__",  json.dumps(chart_data,  ensure_ascii=False))

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {HTML_PATH}  ({len(df)} datasets, {n_cells_fmt} cells, {n_excellent} very-good+)")


if __name__ == "__main__":
    generate()
