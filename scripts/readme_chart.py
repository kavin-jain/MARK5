"""Render the README equity-curve chart (light + dark SVG) from docs/data/mark6.json.

Stdlib only. Re-run after export_dashboard.py so the chart never drifts from the feed:
    python scripts/readme_chart.py
"""
import json
import math
import os
from datetime import date

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEED = os.path.join(_ROOT, "docs", "data", "mark6.json")
OUT = os.path.join(_ROOT, "docs", "img")

THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series="#2a78d6", bench="#898781", ring="#ffffff"),
    "dark": dict(ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series="#3987e5", bench="#898781", ring="#0d1117"),
}
W, H = 880, 400
L, R, T, B = 56, 128, 84, 68  # plot margins
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def render(sys_curve, bench_curve, t):
    d0 = date.fromisoformat(sys_curve[0][0])
    d1 = date.fromisoformat(sys_curve[-1][0])
    span = (d1 - d0).days
    vals = [v for _, v in sys_curve + bench_curve]
    lo, hi = math.log(min(vals) * 0.9), math.log(max(vals) * 1.15)
    pw, ph = W - L - R, H - T - B

    def x(day):
        return L + pw * (date.fromisoformat(day) - d0).days / span

    def y(v):
        return T + ph * (hi - math.log(v)) / (hi - lo)

    def path(curve):
        return " ".join(f"{'M' if i == 0 else 'L'}{x(dd):.1f},{y(v):.1f}" for i, (dd, v) in enumerate(curve))

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" style="font-variant-numeric:tabular-nums">',
         f'<title>Growth of 1 rupee: MARK6 simulated vs Nifty 50 TRI, {d0.year} to {d1.year}, log scale</title>',
         f'<text x="{L}" y="26" font-size="16" font-weight="600" fill="{t["ink"]}">Growth of ₹1 — MARK6 (simulated) vs Nifty 50 TRI</text>',
         f'<text x="{L}" y="46" font-size="12" fill="{t["ink2"]}">Weekly, {d0:%b %Y} – {d1:%b %Y} · log scale · after costs and tax paid along the way</text>']
    # legend
    lx = L
    for label, col in (("MARK6 (simulated)", t["series"]), ("Nifty 50 TRI", t["bench"])):
        o.append(f'<line x1="{lx}" y1="66" x2="{lx + 18}" y2="66" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<text x="{lx + 24}" y="70" font-size="12" fill="{t["ink2"]}">{label}</text>')
        lx += 24 + 7 * len(label) + 20
    # y grid: powers of two
    v = 1.0
    while math.log(v) > lo:
        v /= 2
    while math.log(v) < hi:
        if math.log(v) > lo:
            o.append(f'<line x1="{L}" y1="{y(v):.1f}" x2="{L + pw}" y2="{y(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
            o.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">₹{v:g}</text>')
        v *= 2
    # x axis: every other year
    o.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="{t["axis"]}" stroke-width="1"/>')
    for yr in range(d0.year, d1.year + 1, 2):
        xx = x(f"{yr}-01-01")
        o.append(f'<text x="{xx:.1f}" y="{T + ph + 18}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{yr}</text>')
    # lines: benchmark under, focus series on top
    for curve, col in ((bench_curve, t["bench"]), (sys_curve, t["series"])):
        o.append(f'<path d="{path(curve)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    # end markers + direct labels (text in ink, mark carries identity)
    for curve, col in ((sys_curve, t["series"]), (bench_curve, t["bench"])):
        dd, val = curve[-1]
        o.append(f'<circle cx="{x(dd):.1f}" cy="{y(val):.1f}" r="4" fill="{col}" stroke="{t["ring"]}" stroke-width="2"/>')
        o.append(f'<text x="{x(dd) + 10:.1f}" y="{y(val) + 4:.1f}" font-size="13" font-weight="600" fill="{t["ink"]}">₹{val:.2f}</text>')
    o.append(f'<text x="{L}" y="{H - 14}" font-size="11" fill="{t["muted"]}">Curves exclude the one-off exit tax on the last day; the CAGRs in the table include it. Data: docs/data/mark6.json</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    research = json.load(open(FEED))["research"]
    assert research["terminal_tax"]["curves_are_gross"], "footnote assumes pre-exit-tax curves"
    os.makedirs(OUT, exist_ok=True)
    for name, theme in THEMES.items():
        with open(os.path.join(OUT, f"equity-curve-{name}.svg"), "w") as f:
            f.write(render(research["equity_curve"], research["benchmark_curve"], theme))


if __name__ == "__main__":
    main()
