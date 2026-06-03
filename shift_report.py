"""
shift_report.py
───────────────
Generates a self-contained HTML Shift Intelligence Report.

All CSS is inline. No external dependencies. No new pip installs.
PDF export is done via browser's native print (window.print()) —
no reportlab, no wkhtmltopdf needed.

Usage (from flask_app.py):
    from shift_report import generate_report
    html = generate_report(alert_log, streamer_stats, session_start_time)
    # Save to file or return as Response
"""

import datetime
import json
from collections import Counter, defaultdict


def generate_report(alert_log: list, stats: dict, session_start: float) -> str:
    """
    Generate a self-contained HTML shift report.

    Args:
        alert_log:     list of alert dicts from _alert_log
        stats:         streamer.stats dict
        session_start: epoch time when session started

    Returns:
        Complete HTML string — save as .html or serve directly.
    """
    now          = datetime.datetime.now()
    start_dt     = datetime.datetime.fromtimestamp(session_start)
    duration_sec = int((now - start_dt).total_seconds())
    duration_str = _fmt_duration(duration_sec)

    # ── Compute analytics ─────────────────────────────────────────────────────
    type_counts        = Counter()
    severity_counts    = Counter({"high": 0, "medium": 0, "low": 0})
    zone_counts        = Counter()
    guard_alert_counts = Counter()
    guard_zones        = defaultdict(set)
    hourly_counts      = defaultdict(int)
    high_incidents     = []

    for a in alert_log:
        atype    = a.get("type", "")
        severity = a.get("severity", "low")
        zone     = a.get("zone", "")
        guard    = a.get("guard_id", "")
        ts       = a.get("timestamp", "")

        if atype.startswith("Patrol:"):
            path = atype.replace("Patrol:", "").strip()
            for z in path.replace(" ", "").split("->"):
                if z in ("A", "B", "C", "D"):
                    guard_zones[guard].add(z)
            type_counts["Patrol"] += 1
        else:
            type_counts[atype] += 1

        severity_counts[severity] += 1
        if zone and zone not in ("—", "-", ""):
            zone_counts[zone] += 1
        if guard and guard not in ("Post", ""):
            guard_alert_counts[guard] += 1
        if ts and len(ts) >= 5:
            hour_key = ts[:2] + ":00"
            hourly_counts[hour_key] += 1
        if severity == "high":
            high_incidents.append(a)

    total         = len(alert_log)
    high_count    = severity_counts["high"]
    high_pct      = round(high_count / total * 100) if total else 0
    top_zone      = zone_counts.most_common(1)[0] if zone_counts else ("—", 0)
    guards_list   = [g for g in guard_alert_counts if g not in ("Camera", "Post", "")]

    # Compliance scores from stats
    compliance_scores = stats.get("compliance_scores", {})
    dwell_times       = stats.get("dwell_times", {})

    # Guard patrol coverage
    guard_compliance = {
        g: int(len(guard_zones[g]) / 4 * 100)
        for g in guards_list
    }

    # ── Build HTML ────────────────────────────────────────────────────────────
    guard_rows = _build_guard_rows(
        guards_list, guard_alert_counts, guard_compliance,
        compliance_scores, dwell_times, guard_zones)

    alert_rows = _build_alert_rows(alert_log[-100:])  # last 100 alerts

    # Bar chart SVG for alert types
    top_types   = [(k, v) for k, v in type_counts.most_common(8) if k != "Patrol"]
    types_svg   = _build_bar_svg(top_types, width=420, height=200)

    # Hourly chart SVG
    hourly_svg  = _build_hourly_svg(hourly_counts, width=640, height=110)

    # Severity pie SVG
    severity_svg = _build_severity_svg(severity_counts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GMS Shift Report — {now.strftime('%d %b %Y')}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, 'Segoe UI', Arial, sans-serif;
    background: #f4f6f9;
    color: #1a2233;
    font-size: 13px;
    line-height: 1.5;
  }}

  /* Print styles */
  @media print {{
    body {{ background: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .no-print {{ display: none !important; }}
    .page-break {{ page-break-before: always; }}
    .card {{ break-inside: avoid; margin-bottom: 12px; }}
    .report-wrap {{ max-width: 100%; padding: 0; }}
    @page {{ margin: 1.5cm; size: A4; }}
  }}

  .report-wrap {{
    max-width: 960px;
    margin: 0 auto;
    padding: 24px 20px 48px;
  }}

  /* ── Header ── */
  .report-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0d1117;
    color: #fff;
    padding: 24px 32px;
    border-radius: 8px;
    margin-bottom: 20px;
  }}
  .rh-left {{ display: flex; flex-direction: column; gap: 4px; }}
  .rh-logo {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .rh-logo-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
  }}
  .rh-subtitle {{ font-size: 11px; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; }}
  .rh-right {{ text-align: right; }}
  .rh-date {{ font-size: 18px; font-weight: 600; }}
  .rh-meta {{ font-size: 11px; color: #8b949e; margin-top: 3px; }}

  /* ── Print button ── */
  .print-bar {{
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-bottom: 16px;
  }}
  .print-btn {{
    padding: 8px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
  }}
  .btn-pdf {{
    background: #1d4ed8;
    color: #fff;
  }}
  .btn-pdf:hover {{ background: #1e40af; }}

  /* ── Summary cards ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi {{
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
  }}
  .kpi-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 6px;
  }}
  .kpi-val {{
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
    color: #0f172a;
  }}
  .kpi-val.danger {{ color: #dc2626; }}
  .kpi-val.ok     {{ color: #16a34a; }}
  .kpi-val.warn   {{ color: #d97706; }}
  .kpi-sub {{
    font-size: 10px;
    color: #94a3b8;
    margin-top: 4px;
  }}

  /* ── Section title ── */
  .section-title {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 10px;
    font-weight: 600;
  }}

  /* ── Card ── */
  .card {{
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
  }}
  .card-title {{
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
  }}

  /* ── Two-column row ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }}

  /* ── Guard table ── */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .data-table th {{
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    padding: 0 10px 8px 0;
    border-bottom: 2px solid #f1f5f9;
    font-weight: 600;
  }}
  .data-table td {{
    padding: 10px 10px 10px 0;
    border-bottom: 1px solid #f8fafc;
    color: #374151;
    vertical-align: middle;
  }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover td {{ background: #fafafa; }}

  /* Compliance bar */
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar-bg {{
    flex: 1;
    height: 5px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
    min-width: 60px;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 3px;
  }}
  .bar-fill.ok     {{ background: #22c55e; }}
  .bar-fill.warn   {{ background: #f59e0b; }}
  .bar-fill.danger {{ background: #ef4444; }}
  .bar-pct {{ font-size: 10px; color: #64748b; min-width: 28px; text-align: right; }}

  /* Severity badge */
  .sev-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .sev-badge.high   {{ background: #fef2f2; color: #dc2626; }}
  .sev-badge.medium {{ background: #fffbeb; color: #d97706; }}
  .sev-badge.low    {{ background: #f0fdf4; color: #16a34a; }}

  /* Zone badges */
  .zone-tag {{
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
    margin: 1px 2px;
  }}

  /* ── Zone heatmap ── */
  .zone-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 8px;
    height: 160px;
  }}
  .zh {{
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
  }}
  .zh-lbl {{ font-size: 22px; font-weight: 700; color: #cbd5e1; }}
  .zh-cnt {{ font-size: 14px; font-weight: 600; color: #374151; }}
  .zh-sub {{ font-size: 9px; color: #94a3b8; }}

  /* ── Alert timeline table ── */
  .timeline-wrap {{
    max-height: 400px;
    overflow-y: auto;
  }}

  /* ── Footer ── */
  .report-footer {{
    text-align: center;
    font-size: 10px;
    color: #94a3b8;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
  }}

  /* ── High incident highlight ── */
  .incident-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #f8fafc;
  }}
  .incident-item:last-child {{ border-bottom: none; }}
  .incident-icon {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #dc2626;
    flex-shrink: 0;
    margin-top: 4px;
  }}
  .incident-type {{ font-weight: 600; font-size: 12px; color: #0f172a; }}
  .incident-meta {{ font-size: 11px; color: #64748b; margin-top: 1px; }}
</style>
</head>
<body>
<div class="report-wrap">

  <!-- Action bar -->
  <div class="print-bar no-print">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div style="font-size:11px;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;
                  border-radius:6px;padding:8px 14px;line-height:1.5;">
        <strong style="color:#374151;">📄 Save as PDF:</strong>
        Click <strong>Print → Destination → Save as PDF</strong> → Save
      </div>
      <button class="print-btn btn-pdf" onclick="window.print()">
        🖨 Print / Save as PDF
      </button>
      <a class="print-btn" id="btn-download-html"
         style="background:#0f766e;color:#fff;text-decoration:none;display:inline-block;"
         href="#" onclick="downloadHTML(event)">
        ⬇ Download HTML Report
      </a>
    </div>
  </div>

  <!-- Header -->
  <div class="report-header">
    <div class="rh-left">
      <div class="rh-logo">
        <div class="rh-logo-dot"></div>
        GMS
      </div>
      <div class="rh-subtitle">Guard Monitoring System — Shift Intelligence Report</div>
    </div>
    <div class="rh-right">
      <div class="rh-date">{now.strftime('%d %B %Y')}</div>
      <div class="rh-meta">
        Generated {now.strftime('%H:%M:%S')}<br>
        Session: {start_dt.strftime('%H:%M')} – {now.strftime('%H:%M')} ({duration_str})
      </div>
    </div>
  </div>

  <!-- KPI Cards -->
  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">Total Alerts</div>
      <div class="kpi-val">{total}</div>
      <div class="kpi-sub">this session</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">High Severity</div>
      <div class="kpi-val {'danger' if high_pct >= 40 else 'warn' if high_pct >= 20 else 'ok'}">{high_pct}%</div>
      <div class="kpi-sub">{high_count} critical events</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Top Alert Zone</div>
      <div class="kpi-val warn">{top_zone[0]}</div>
      <div class="kpi-sub">{top_zone[1]} alerts in zone</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Guards Monitored</div>
      <div class="kpi-val ok">{len(guards_list)}</div>
      <div class="kpi-sub">{stats.get('guards_detected', 0)} currently in frame</div>
    </div>
  </div>

  <!-- High Priority Incidents -->
  {'<div class="card"><div class="card-title">⚠ High Severity Incidents</div>' + _build_incidents(high_incidents[-10:]) + '</div>' if high_incidents else ''}

  <!-- Charts row -->
  <div class="two-col">
    <div class="card">
      <div class="card-title">Alert Type Breakdown</div>
      {types_svg}
    </div>
    <div class="card">
      <div class="card-title">Severity Distribution</div>
      {severity_svg}
    </div>
  </div>

  <!-- Hourly distribution -->
  <div class="card">
    <div class="card-title">Hourly Alert Distribution</div>
    {hourly_svg}
  </div>

  <!-- Guard performance + Zone heatmap -->
  <div class="two-col">
    <div class="card">
      <div class="card-title">Zone Activity Heatmap</div>
      <div class="zone-grid">
        {_build_zone_cells(zone_counts)}
      </div>
    </div>
    <div class="card">
      <div class="card-title">Guard Performance</div>
      {guard_rows}
    </div>
  </div>

  <!-- Alert Timeline -->
  <div class="card page-break">
    <div class="card-title">Alert Timeline (Last {min(len(alert_log), 100)} Events)</div>
    <div class="timeline-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Alert</th>
            <th>Guard</th>
            <th>Zone</th>
            <th>Severity</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {alert_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <div class="report-footer">
    GMS — Guard Monitoring System &nbsp;·&nbsp;
    Report generated {now.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp;
    Session duration: {duration_str} &nbsp;·&nbsp;
    Total frames processed: {stats.get('frame_count', '—')}
  </div>

</div>
<script>
function downloadHTML(e) {{
  e.preventDefault();
  const html = document.documentElement.outerHTML;
  const blob = new Blob([html], {{type: 'text/html'}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  const ts   = new Date().toISOString().slice(0,16).replace(/[T:]/g,'-');
  a.href     = url;
  a.download = 'GMS_ShiftReport_' + ts + '.html';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>"""

    return html


# ── Helper functions ───────────────────────────────────────────────────────────

def _fmt_duration(secs: int) -> str:
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _build_guard_rows(guards, alert_counts, patrol_cov, compliance_scores,
                      dwell_times, guard_zones) -> str:
    if not guards:
        return '<p style="color:#94a3b8;font-size:12px;padding:8px 0;">No named guards detected this session.</p>'

    rows = []
    for g in sorted(guards, key=lambda x: alert_counts.get(x, 0), reverse=True):
        alerts  = alert_counts.get(g, 0)
        cov     = patrol_cov.get(g, 0)
        score   = compliance_scores.get(g)
        dwell   = dwell_times.get(g, 0)
        zones   = guard_zones.get(g, set())

        cls = "ok" if cov >= 75 else "warn" if cov >= 50 else "danger"

        dwell_str = ""
        if dwell > 0:
            dwell_str = f"{dwell//60}m {dwell%60:02d}s" if dwell >= 60 else f"{dwell}s"
        else:
            dwell_str = "—"

        score_str = f"{score:.0f}%" if score is not None else "—"
        score_cls = ("ok" if score and score >= 75
                     else "warn" if score and score >= 50
                     else "danger") if score else ""

        zone_tags = "".join(f'<span class="zone-tag">{z}</span>' for z in sorted(zones)) or "—"

        rows.append(f"""
        <tr>
          <td style="font-weight:600">{_esc(g)}</td>
          <td>{alerts}</td>
          <td style="font-size:11px">{dwell_str}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-bg"><div class="bar-fill {cls}" style="width:{cov}%"></div></div>
              <span class="bar-pct">{cov}%</span>
            </div>
          </td>
          <td style="font-weight:600;color:{'#16a34a' if score_cls=='ok' else '#d97706' if score_cls=='warn' else '#dc2626'}">{score_str}</td>
          <td>{zone_tags}</td>
        </tr>""")

    return f"""<table class="data-table">
      <thead><tr>
        <th>Guard</th><th>Alerts</th><th>Dwell</th>
        <th>Patrol Coverage</th><th>Compliance</th><th>Zones Visited</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _build_alert_rows(alerts: list) -> str:
    rows = []
    for a in reversed(alerts):  # newest first
        sev   = a.get("severity", "low")
        acked = "✓ Reviewed" if a.get("acknowledged") else "Pending"
        acked_style = "color:#16a34a" if a.get("acknowledged") else "color:#94a3b8"
        rows.append(f"""
        <tr>
          <td style="font-family:monospace;font-size:11px">{_esc(a.get('timestamp',''))}</td>
          <td style="font-weight:500">{_esc(a.get('type',''))}</td>
          <td>{_esc(a.get('guard_id',''))}</td>
          <td>{_esc(a.get('zone',''))}</td>
          <td><span class="sev-badge {sev}">{sev}</span></td>
          <td style="font-size:11px;{acked_style}">{acked}</td>
        </tr>""")
    return "".join(rows) or '<tr><td colspan="6" style="color:#94a3b8;padding:16px 0;">No alerts recorded.</td></tr>'


def _build_incidents(incidents: list) -> str:
    if not incidents:
        return '<p style="color:#94a3b8;font-size:12px;">No high-severity incidents.</p>'
    items = []
    for a in reversed(incidents[-6:]):
        items.append(f"""
        <div class="incident-item">
          <div class="incident-icon"></div>
          <div>
            <div class="incident-type">{_esc(a.get('type',''))}</div>
            <div class="incident-meta">
              {_esc(a.get('guard_id',''))} &middot; Zone {_esc(a.get('zone',''))} &middot; {_esc(a.get('timestamp',''))}
            </div>
          </div>
        </div>""")
    return "".join(items)


def _build_zone_cells(zone_counts: Counter) -> str:
    max_val = max(zone_counts.values()) if zone_counts else 1
    cells = []
    for zone in ["A", "B", "C", "D"]:
        count = zone_counts.get(zone, 0)
        intensity = count / max(max_val, 1)
        r = int(219 + (29 - 219) * intensity)
        g_val = int(234 + (130 - 234) * intensity)
        b = int(254 + (246 - 254) * intensity)
        bg = f"rgb({r},{g_val},{b})"
        cells.append(f"""
        <div class="zh" style="background:{bg}">
          <span class="zh-lbl">{zone}</span>
          <span class="zh-cnt">{count}</span>
          <span class="zh-sub">alerts</span>
        </div>""")
    return "".join(cells)


def _build_bar_svg(entries: list, width=420, height=200) -> str:
    """Build an SVG horizontal bar chart for alert types."""
    if not entries:
        return '<p style="color:#94a3b8;font-size:12px;">No alert data yet.</p>'

    max_val  = max(v for _, v in entries) or 1
    bar_h    = 18
    gap      = 8
    label_w  = 160
    bar_area = width - label_w - 50
    total_h  = len(entries) * (bar_h + gap)
    actual_h = max(total_h + 20, height)

    svg_parts = [f'<svg width="{width}" height="{actual_h}" xmlns="http://www.w3.org/2000/svg">']

    colors = {
        "high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"
    }

    for i, (label, val) in enumerate(entries):
        y = i * (bar_h + gap) + 10
        bar_w = int((val / max_val) * bar_area)

        # Color by type
        lbl_lower = label.lower()
        if any(k in lbl_lower for k in ["missing", "weapon", "sleeping", "fire", "fight", "tamper"]):
            color = "#ef4444"
        elif any(k in lbl_lower for k in ["idle", "smoking", "unknown", "crowd"]):
            color = "#f59e0b"
        else:
            color = "#3b82f6"

        # Label text — truncate if too long
        display_label = label if len(label) <= 22 else label[:20] + "…"

        svg_parts.append(
            f'<text x="{label_w - 6}" y="{y + bar_h - 4}" '
            f'text-anchor="end" font-size="10" font-family="sans-serif" fill="#374151">'
            f'{_esc_svg(display_label)}</text>'
        )
        svg_parts.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'rx="3" fill="{color}" opacity="0.85"/>'
        )
        svg_parts.append(
            f'<text x="{label_w + bar_w + 5}" y="{y + bar_h - 4}" '
            f'font-size="10" font-family="monospace" fill="#64748b">{val}</text>'
        )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _build_hourly_svg(hourly_counts: dict, width=640, height=110) -> str:
    """Build an SVG bar chart for hourly distribution."""
    hours  = [f"{str(i).zfill(2)}:00" for i in range(24)]
    counts = [hourly_counts.get(h, 0) for h in hours]
    max_c  = max(counts) if any(counts) else 1

    bar_w     = (width - 40) / 24
    chart_h   = height - 25
    svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    for i, (h, c) in enumerate(zip(hours, counts)):
        x      = 20 + i * bar_w
        bar_h  = int((c / max_c) * chart_h) if c > 0 else 2
        y      = chart_h - bar_h + 5
        color  = "#3b82f6" if c > 0 else "#e2e8f0"
        opacity = "0.85" if c > 0 else "0.5"

        svg_parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_w - 1:.1f}" height="{bar_h}" '
            f'rx="2" fill="{color}" opacity="{opacity}"/>'
        )
        if i % 4 == 0:
            svg_parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{height - 4}" '
                f'text-anchor="middle" font-size="8" font-family="sans-serif" fill="#94a3b8">'
                f'{h[:2]}h</text>'
            )
        if c > 0:
            svg_parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{y - 2}" '
                f'text-anchor="middle" font-size="8" font-family="monospace" fill="#64748b">'
                f'{c}</text>'
            )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _build_severity_svg(severity_counts: Counter) -> str:
    """Build an SVG donut chart for severity distribution."""
    high   = severity_counts.get("high", 0)
    medium = severity_counts.get("medium", 0)
    low    = severity_counts.get("low", 0)
    total  = high + medium + low

    if total == 0:
        return '<p style="color:#94a3b8;font-size:12px;">No alerts recorded.</p>'

    import math

    cx, cy, r_outer, r_inner = 100, 100, 80, 50
    colors  = ["#ef4444", "#f59e0b", "#22c55e"]
    labels  = ["High", "Medium", "Low"]
    values  = [high, medium, low]
    angles  = [v / total * 2 * math.pi for v in values]

    svg_parts = ['<svg width="240" height="220" xmlns="http://www.w3.org/2000/svg">']

    start = -math.pi / 2
    for i, (angle, color) in enumerate(zip(angles, colors)):
        if angle == 0:
            start += angle
            continue
        end  = start + angle
        x1   = cx + r_outer * math.cos(start)
        y1   = cy + r_outer * math.sin(start)
        x2   = cx + r_outer * math.cos(end)
        y2   = cy + r_outer * math.sin(end)
        xi1  = cx + r_inner * math.cos(end)
        yi1  = cy + r_inner * math.sin(end)
        xi2  = cx + r_inner * math.cos(start)
        yi2  = cy + r_inner * math.sin(start)
        large = 1 if angle > math.pi else 0

        path = (f"M {x1:.2f} {y1:.2f} "
                f"A {r_outer} {r_outer} 0 {large} 1 {x2:.2f} {y2:.2f} "
                f"L {xi1:.2f} {yi1:.2f} "
                f"A {r_inner} {r_inner} 0 {large} 0 {xi2:.2f} {yi2:.2f} Z")

        svg_parts.append(f'<path d="{path}" fill="{color}" opacity="0.85"/>')
        start = end

    # Center text
    svg_parts.append(
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'font-size="22" font-weight="bold" font-family="sans-serif" fill="#0f172a">'
        f'{total}</text>'
    )
    svg_parts.append(
        f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" '
        f'font-size="9" font-family="sans-serif" fill="#94a3b8">TOTAL</text>'
    )

    # Legend
    ly = 195
    lx = 10
    for label, value, color in zip(labels, values, colors):
        pct = round(value / total * 100)
        svg_parts.append(
            f'<rect x="{lx}" y="{ly}" width="8" height="8" rx="2" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{lx + 11}" y="{ly + 8}" '
            f'font-size="9" font-family="sans-serif" fill="#374151">'
            f'{label} ({value}, {pct}%)</text>'
        )
        lx += 78

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_svg(s) -> str:
    return _esc(s).replace('"', "&quot;")