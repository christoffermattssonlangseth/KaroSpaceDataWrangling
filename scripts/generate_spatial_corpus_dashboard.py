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
# ---------------------------------------------------------------------------

def _score(row: pd.Series) -> int:
    """
    KaroSpace fit is primarily about multi-sample data.
    n_donors is the main axis; spatial assay and DOI are secondary bonuses.

    Points:
      n_donors >= 10  → 3 pts   (many biological replicates)
      n_donors >= 5   → 2 pts
      n_donors >= 2   → 1 pt
      is_spatial      → +1 pt   (spatial assay = directly usable in KaroSpace)
      has_doi         → +1 pt   (peer-reviewed)
    Max: 5
    """
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

        rows.append({
            "filename":       r["filename"].replace(".h5ad", ""),
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
        })
    return rows


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpatialCorpus-110M — Dataset Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:      #0f1117;
    --surface: #1a1d27;
    --card:    #22263a;
    --border:  #2e3350;
    --accent:  #4f8ef7;
    --text:    #e2e8f0;
    --muted:   #8892a4;
    --green:   #4fc47e;
    --radius:  10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 18px 32px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
  header h1 {{ font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }}
  header span {{ color: var(--muted); font-size: 12px; }}

  .main {{ padding: 24px 32px; max-width: 1800px; margin: 0 auto; }}

  /* Summary cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; }}
  .card .val {{ font-size: 24px; font-weight: 700; color: var(--accent); line-height: 1.1; }}
  .card .lbl {{ color: var(--muted); font-size: 11px; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* Charts */
  .charts {{ display: grid; grid-template-columns: 1fr 1fr 280px; gap: 14px; margin-bottom: 24px; }}
  .chart-box {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; }}
  .chart-box h2 {{ font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
  .chart-wrap {{ position: relative; height: 260px; }}

  /* Score legend */
  .legend {{ display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  /* Controls */
  .controls {{ display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
  .controls input, .controls select {{
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); padding: 7px 12px; font-size: 13px; outline: none;
  }}
  .controls input {{ flex: 1; min-width: 200px; }}
  .controls input:focus, .controls select:focus {{ border-color: var(--accent); }}
  .count {{ color: var(--muted); font-size: 12px; white-space: nowrap; margin-left: auto; }}

  /* Table */
  .table-wrap {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead {{ background: var(--surface); position: sticky; top: 0; z-index: 2; }}
  th {{ padding: 9px 12px; text-align: left; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; cursor: pointer; user-select: none; white-space: nowrap; border-bottom: 1px solid var(--border); }}
  th:hover {{ color: var(--text); }}
  th .si {{ opacity: 0.35; margin-left: 3px; font-size: 10px; }}
  th.asc  .si::after {{ content: "↑"; opacity: 1; color: var(--accent); }}
  th.desc .si::after {{ content: "↓"; opacity: 1; color: var(--accent); }}
  th:not(.asc):not(.desc) .si::after {{ content: "↕"; }}
  tbody tr {{ border-top: 1px solid var(--border); transition: background 0.1s; }}
  tbody tr:hover {{ background: rgba(79,142,247,0.05); }}
  td {{ padding: 8px 12px; vertical-align: middle; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; }}
  td.muted {{ color: var(--muted); font-size: 12px; }}
  td.tissue-cell {{ max-width: 220px; font-size: 12px; color: var(--muted); }}
  td.cond-cell {{ max-width: 180px; font-size: 11px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px;
    font-weight: 500; white-space: nowrap;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
  }}
  .score-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 600;
    background: rgba(0,0,0,0.25); border: 1px solid;
    white-space: nowrap;
  }}
  .score-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
  .filename {{ font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; }}
  .no-results {{ text-align: center; padding: 48px; color: var(--muted); }}

  /* Stat mini-bars inside cells */
  .mini-bar-wrap {{ display: flex; align-items: center; gap: 6px; }}
  .mini-bar {{ height: 4px; border-radius: 2px; background: var(--accent); opacity: 0.7; min-width: 2px; }}

  @media (max-width: 1100px) {{ .charts {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 700px)  {{ .charts {{ grid-template-columns: 1fr; }} .main {{ padding: 14px; }} }}
</style>
</head>
<body>

<header>
  <h1>SpatialCorpus-110M &mdash; Dataset Explorer</h1>
  <span>Last updated: {updated} &nbsp;·&nbsp; {n_files} datasets &nbsp;·&nbsp; {n_cells_fmt} cells</span>
</header>

<div class="main">

  <div class="cards">
    <div class="card"><div class="val">{n_files}</div><div class="lbl">Datasets</div></div>
    <div class="card"><div class="val">{n_cells_fmt}</div><div class="lbl">Total cells</div></div>
    <div class="card"><div class="val">{n_human}</div><div class="lbl">Human</div></div>
    <div class="card"><div class="val">{n_mouse}</div><div class="lbl">Mouse</div></div>
    <div class="card"><div class="val">{n_spatial}</div><div class="lbl">Spatial assays</div></div>
    <div class="card"><div class="val">{n_assays}</div><div class="lbl">Assay types</div></div>
    <div class="card"><div class="val">{n_tissues}</div><div class="lbl">Tissue types</div></div>
    <div class="card"><div class="val">{n_excellent}</div><div class="lbl">Very good+ fit</div></div>
  </div>

  <div class="charts">
    <div class="chart-box">
      <h2>Cells by assay</h2>
      <div class="chart-wrap"><canvas id="chartAssay"></canvas></div>
    </div>
    <div class="chart-box">
      <h2>Cells by tissue (top 20)</h2>
      <div class="chart-wrap"><canvas id="chartTissue"></canvas></div>
    </div>
    <div class="chart-box">
      <h2>Cells by organism</h2>
      <div class="chart-wrap"><canvas id="chartOrganism"></canvas></div>
    </div>
  </div>

  <div class="legend">
    <span style="font-size:12px;color:var(--muted);margin-right:4px">KaroSpace fit:</span>
    <div class="legend-item"><div class="legend-dot" style="background:#4fc47e"></div>Excellent (≥10 donors + spatial + DOI)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#86efac"></div>Very good (≥5 donors + spatial or DOI)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div>Good (≥2–4 donors + bonuses)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f97b4f"></div>Moderate (few donors)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>Limited (single sample)</div>
  </div>

  <div class="controls">
    <input type="text" id="search" placeholder="Search name, tissue, assay, condition…">
    <select id="filterOrganism"><option value="">All organisms</option></select>
    <select id="filterAssay"><option value="">All assays</option></select>
    <select id="filterScore">
      <option value="">All KaroSpace fits</option>
      <option value="5">Excellent (5)</option>
      <option value="4">Very good (4)</option>
      <option value="3">Good (3)</option>
      <option value="2">Moderate (2)</option>
    </select>
    <span class="count" id="rowCount"></span>
  </div>

  <div class="table-wrap">
    <table id="dataTable">
      <thead><tr>
        <th data-col="score" class="desc">KaroSpace fit<span class="si"></span></th>
        <th data-col="filename">Dataset<span class="si"></span></th>
        <th data-col="n_cells">Cells<span class="si"></span></th>
        <th data-col="n_vars">Genes<span class="si"></span></th>
        <th data-col="n_donors">Donors<span class="si"></span></th>
        <th data-col="n_conditions">Conditions<span class="si"></span></th>
        <th data-col="n_tissue_types">Tissue types<span class="si"></span></th>
        <th data-col="n_sections">Sections<span class="si"></span></th>
        <th data-col="organism">Organism<span class="si"></span></th>
        <th data-col="assay">Assay<span class="si"></span></th>
        <th data-col="tissue">Tissue<span class="si"></span></th>
        <th data-col="conditions">Conditions detail<span class="si"></span></th>
        <th>Publication</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="no-results" id="noResults" style="display:none">No datasets match your filters.</div>
  </div>
</div>

<script>
const ROWS       = {rows_json};
const CHART_DATA = {chart_json};
const MAX_CELLS  = Math.max(...ROWS.map(r => r.n_cells));
const MAX_DONORS = Math.max(...ROWS.map(r => r.n_donors));

// ── Charts ───────────────────────────────────────────────────────────────────
const BASE_OPTS = {{
  maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color:"#8892a4", font:{{size:11}} }}, grid: {{color:"#2e3350"}} }},
    y: {{ ticks: {{ color:"#8892a4", font:{{size:11}},
          callback: v => v>=1e6 ? (v/1e6).toFixed(1)+"M" : v>=1e3 ? (v/1e3).toFixed(0)+"k" : v }},
         grid: {{color:"#2e3350"}} }},
  }},
}};
const H_OPTS = {{ ...BASE_OPTS, indexAxis:"y",
  scales: {{ x: BASE_OPTS.scales.y, y: {{ ...BASE_OPTS.scales.x, ticks: {{ color:"#8892a4", font:{{size:11}} }} }} }} }};

new Chart(document.getElementById("chartAssay"), {{
  type:"bar",
  data:{{ labels: CHART_DATA.assay.labels,
          datasets:[{{ data: CHART_DATA.assay.values, backgroundColor: CHART_DATA.assay.colors, borderRadius:4 }}] }},
  options: H_OPTS,
}});
new Chart(document.getElementById("chartTissue"), {{
  type:"bar",
  data:{{ labels: CHART_DATA.tissue.labels,
          datasets:[{{ data: CHART_DATA.tissue.values, backgroundColor:"#4f8ef7", borderRadius:4 }}] }},
  options: H_OPTS,
}});
new Chart(document.getElementById("chartOrganism"), {{
  type:"doughnut",
  data:{{ labels: CHART_DATA.organism.labels,
          datasets:[{{ data: CHART_DATA.organism.values, backgroundColor: CHART_DATA.organism.colors, borderWidth:0, hoverOffset:6 }}] }},
  options:{{
    maintainAspectRatio:false,
    plugins:{{ legend:{{ display:true, position:"bottom",
      labels:{{ color:"#8892a4", font:{{size:12}}, padding:16 }} }} }},
  }},
}});

// ── Dropdowns ────────────────────────────────────────────────────────────────
function populateSelect(id, values) {{
  const sel = document.getElementById(id);
  [...new Set(values)].sort().forEach(v => {{
    const o = document.createElement("option"); o.value = v; o.textContent = v; sel.appendChild(o);
  }});
}}
populateSelect("filterOrganism", ROWS.map(r => r.organism).filter(Boolean));
populateSelect("filterAssay",    ROWS.map(r => r.assay).filter(Boolean));

// ── Table rendering ──────────────────────────────────────────────────────────
let sortCol = "score", sortDir = -1;

function fmt(n) {{
  return n >= 1e6 ? (n/1e6).toFixed(2)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"k" : String(n);
}}

function miniBar(val, max) {{
  const pct = max > 0 ? Math.max(4, Math.round((val/max)*80)) : 0;
  return `<div class="mini-bar-wrap">
    <div class="mini-bar" style="width:${{pct}}px"></div>
    <span>${{fmt(val)}}</span>
  </div>`;
}}

function donorBar(n) {{
  if (!n) return '<span style="color:#64748b">—</span>';
  const color = n >= 10 ? "#4fc47e" : n >= 5 ? "#86efac" : n >= 2 ? "#fbbf24" : "#94a3b8";
  const pct   = Math.max(4, Math.round((n / MAX_DONORS) * 72));
  return `<div class="mini-bar-wrap">
    <div class="mini-bar" style="width:${{pct}}px;background:${{color}}"></div>
    <span style="color:${{color}};font-weight:600">${{n}}</span>
  </div>`;
}}

function countBadge(n) {{
  if (!n) return '<span style="color:#64748b">—</span>';
  const color = n >= 5 ? "#4fc47e" : n >= 3 ? "#86efac" : n >= 2 ? "#fbbf24" : "#94a3b8";
  return `<span style="color:${{color}};font-weight:600">${{n}}</span>`;
}}

function render() {{
  const q     = document.getElementById("search").value.toLowerCase();
  const org   = document.getElementById("filterOrganism").value;
  const assay = document.getElementById("filterAssay").value;
  const score = document.getElementById("filterScore").value;

  let filtered = ROWS.filter(r => {{
    if (org   && r.organism !== org)           return false;
    if (assay && r.assay    !== assay)         return false;
    if (score && r.score    <  Number(score))  return false;
    if (q) {{
      const hay = [r.filename, r.tissue, r.assay, r.organism, r.conditions, r.pub_title].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});

  filtered.sort((a, b) => {{
    const av = a[sortCol], bv = b[sortCol];
    return (typeof av === "number" ? av - bv : String(av).localeCompare(String(bv))) * sortDir;
  }});

  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";
  filtered.forEach(r => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <span class="score-pill" style="color:${{r.score_color}};border-color:${{r.score_color}}55">
          <span class="score-dot" style="background:${{r.score_color}}"></span>
          ${{r.score_label}}
        </span>
      </td>
      <td class="filename">${{r.filename}}</td>
      <td class="num">${{miniBar(r.n_cells, MAX_CELLS)}}</td>
      <td class="num muted">${{r.n_vars.toLocaleString()}}</td>
      <td class="num">${{donorBar(r.n_donors)}}</td>
      <td class="num">${{countBadge(r.n_conditions)}}</td>
      <td class="num">${{countBadge(r.n_tissue_types)}}</td>
      <td class="num muted">${{r.n_sections || "—"}}</td>
      <td class="muted" style="font-size:12px">${{r.organism}}</td>
      <td><span class="badge" style="color:${{r.assay_color}};border-color:${{r.assay_color}}44">${{r.assay}}</span></td>
      <td class="tissue-cell">${{r.tissue}}</td>
      <td class="cond-cell" title="${{r.conditions}}">${{r.conditions}}</td>
      <td style="font-size:12px">${{r.pub_html}}</td>
    `;
    tbody.appendChild(tr);
  }});

  document.getElementById("rowCount").textContent = `${{filtered.length}} of ${{ROWS.length}} datasets`;
  document.getElementById("noResults").style.display = filtered.length ? "none" : "block";
}}

document.querySelectorAll("th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = th.dataset.col;
    sortDir = sortCol === col ? sortDir * -1 : -1;
    sortCol = col;
    document.querySelectorAll("th").forEach(h => h.classList.remove("asc","desc"));
    th.classList.add(sortDir === -1 ? "desc" : "asc");
    render();
  }});
}});

["search","filterOrganism","filterAssay","filterScore"].forEach(id =>
  document.getElementById(id).addEventListener("input", render)
);

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
    n_with_doi    = int((df["publication_doi"].notna() & (df["publication_doi"] != "")).sum())
    n_human       = int((df["organism"] == "Homo sapiens").sum())
    n_mouse       = int((df["organism"] == "Mus musculus").sum())
    n_excellent   = sum(1 for r in table_rows if r["score"] >= 4)   # "Very good" or better
    n_spatial     = sum(1 for r in table_rows if any(
        s in str(r.get("assay", "")).lower() for s in SPATIAL_ASSAYS
    ))

    html = HTML_TEMPLATE.format(
        updated     = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        n_files     = len(df),
        n_cells_fmt = n_cells_fmt,
        n_human     = n_human,
        n_mouse     = n_mouse,
        n_spatial   = n_spatial,
        n_assays    = n_assays,
        n_tissues   = n_tissues,
        n_excellent = n_excellent,
        rows_json   = json.dumps(table_rows, ensure_ascii=False),
        chart_json  = json.dumps(chart_data,  ensure_ascii=False),
    )

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {HTML_PATH}  ({len(df)} datasets, {n_cells_fmt} cells, {n_excellent} excellent)")


if __name__ == "__main__":
    generate()
