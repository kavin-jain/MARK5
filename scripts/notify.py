"""
The daily message — what happened to the money, in plain English.
=================================================================
The owner is not a trader and should not have to open a dashboard to learn that
something broke. On 4-5 August the daily refresh silently stopped and nobody
noticed for two days, because nothing in this system has ever spoken first.
This is the part that speaks first.

Design rules, all of them consequences of Mandate §8:

  * No jargon. "worst dip" not "max drawdown", "if you had just bought the index"
    not "benchmark relative return". Every number carries its unit.
  * Paper gains and realised gains are never blended into one figure.
  * The weakness goes IN the message, not in a footnote — the worst dip and the
    PAPER disclaimer appear every single day, including good days.
  * A broken day still sends. A notifier that goes quiet exactly when something
    is wrong is worse than none, because silence starts to mean "fine".

Channels — whichever is configured wins, Telegram first:

  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID     private, renders monospace properly
  NTFY_TOPIC [+ NTFY_SERVER, NTFY_TOKEN]    no account needed; a public ntfy.sh
                                            topic is readable by anyone who
                                            guesses the name, so use a long
                                            random one or self-host

With neither set it prints and exits 0, so CI never fails for want of a token.

  python3 scripts/notify.py              # print it, send nothing
  python3 scripts/notify.py --send       # actually send
  python3 scripts/notify.py --no-health  # skip the health subprocess
"""
import argparse
import html
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(_ROOT, "data", "paper", "paper_export.json")
PAGE = "https://kavinjain.in/mark6"

BLOCKS = "▁▂▃▄▅▆▇█"


# ── formatting ───────────────────────────────────────────────────────────
def _grp(digits: str) -> str:
    """Indian digit grouping: 519910 -> 5,19,910. The reader is Indian; western
    grouping (519,910) makes them do a conversion in their head every time."""
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    out = []
    while len(head) > 2:
        out.insert(0, head[-2:])
        head = head[:-2]
    if head:
        out.insert(0, head)
    return ",".join(out + [tail])


def rs(x, signed=False):
    """₹ amount, Indian grouping, ASCII sign so it survives any transport."""
    s = ("+" if x >= 0 else "-") if signed else ("-" if x < 0 else "")
    return f"{s}Rs {_grp(f'{abs(float(x)):.0f}')}"


def pct(x, d=2, signed=True):
    return f"{'+' if (signed and x >= 0) else ''}{float(x):.{d}f}%"


def spark(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return BLOCKS[0] * len(vals)
    return "".join(BLOCKS[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


# ── health ───────────────────────────────────────────────────────────────
def health():
    """Run the existing check rather than re-implementing its judgement here.
    Never raises: a notifier that dies because the thing it reports on is
    unhealthy defeats its own purpose."""
    try:
        p = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "healthcheck.py"),
                            "--json"], capture_output=True, text=True, cwd=_ROOT, timeout=240)
        blob = p.stdout[p.stdout.index("{"):]
        d = json.loads(blob)
        return {"n": len(d["results"]), "fails": d["fails"], "warns": d["warns"],
                "failing": [r["check"] for r in d["results"] if r["status"] == "FAIL"]}
    except Exception as e:                                   # noqa: BLE001
        return {"n": 0, "fails": -1, "warns": 0, "failing": [f"health check did not run: {e}"]}


# ── the message ──────────────────────────────────────────────────────────
W = 36


def _row(label, value):
    return f"  {label:<20}{value:>14}"


def build(L, hp):
    nav, cap = float(L["nav"]), float(L["capital"])
    pnl = nav - cap
    hist = L.get("nav_history") or []
    navs = [float(h["nav_inr"]) for h in hist]
    asof = hist[-1]["date"] if hist else L.get("generated", "")[:10]

    out = [f"MARK6  ·  day {L['days_live']}  ·  as of {asof}",
           "─" * W,
           "YOUR MONEY",
           _row("put in", rs(cap)),
           _row("worth now", rs(nav)),
           _row(f"{'profit' if pnl >= 0 else 'loss'} ({pct(L['return_pct'])})", rs(pnl, True))]

    # Previous mark -> this one. Absent on day one and whenever the prior mark is
    # missing; saying nothing beats inventing a move that did not happen. It is
    # labelled by DATE, not "yesterday" — over a weekend or an outage those differ,
    # and that difference is exactly the thing worth noticing.
    if len(navs) >= 2:
        d_abs, d_pct = navs[-1] - navs[-2], (navs[-1] / navs[-2] - 1) * 100
        out += ["", f"SINCE {hist[-2]['date']}", _row(f"change ({pct(d_pct)})", rs(d_abs, True))]
    if len(navs) >= 3:
        out += [f"  {spark(navs[-20:])}  last {len(navs[-20:])} marks"]

    bench = L.get("benchmark_nav")
    if bench:
        # relative_pct is a difference of two percentages, so its unit is
        # percentage POINTS. Printing it as "%" is the classic quiet lie.
        out += ["", "IF YOU'D JUST BOUGHT THE INDEX",
                _row("you'd have", rs(bench)),
                _row(f"{'ahead' if nav >= bench else 'behind'} by "
                     f"({float(L['relative_pct']):+.2f}pp)", rs(abs(nav - float(bench))))]

    rows = (L.get("sleeves") or {}).get("rows") or []
    if rows:
        out += ["", "THE PARTS"]
        for r in rows:
            out.append(f"  {r['label'][:17]:<17}{rs(r['pnl_inr'], True):>10}"
                       f"{pct(r['return_pct'], 1):>7}")

    real = float(L.get("realised_pnl", 0.0))
    out += ["", "WORTH KNOWING",
            f"  · worst dip so far: {pct(L['max_drawdown_pct'], 1, False)} below its own"
            f"\n    peak. It will be worse than that one day.",
            f"  · only {rs(real, True)} is locked in by actually"
            f"\n    selling. The other {rs(pnl - real, True)} is on paper"
            f"\n    and can still go away."]
    if float(L.get("tax_liability", 0)) > 0:
        out.append(f"  · tax owed if it all ended today: {rs(L['tax_liability'])}")
    out.append("  · PAPER money. Nothing real is invested.")

    if hp["fails"] > 0:
        out += ["", f"⚠ SYSTEM  {hp['fails']} of {hp['n']} checks FAILED"]
        out += [f"    ✗ {c}" for c in hp["failing"][:6]]
    elif hp["fails"] < 0:
        out += ["", "⚠ SYSTEM  the health check itself did not run", f"    {hp['failing'][0][:70]}"]
    else:
        out += ["", f"SYSTEM  all {hp['n']} checks passed"
                    + (f", {hp['warns']} warning(s)" if hp["warns"] else "")]

    out.append(f"\n{PAGE}")
    return "\n".join(out)


# ── transport ────────────────────────────────────────────────────────────
def scrub(text):
    """Never let a bot token reach stdout. The token lives in the URL path for
    every Telegram call, so any code path that prints a URL — an exception, a
    redirect, a traceback — leaks it into the CI log, where it is public. GitHub
    masks registered secrets, but only the exact string, and only if it was
    registered. This is the belt to that braces."""
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    s = str(text)
    if tok:
        s = s.replace(tok, "<TELEGRAM_BOT_TOKEN>")
        if ":" in tok:                       # the numeric id half is also secret-ish
            s = s.replace(tok.split(":", 1)[1], "<REDACTED>")
    for k in ("NTFY_TOKEN",):
        if os.getenv(k):
            s = s.replace(os.environ[k], f"<{k}>")
    return s


def _post(url, data, headers):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def whoami():
    """Print the chat_id for TELEGRAM_BOT_TOKEN, so setup is one command instead
    of hand-parsing getUpdates JSON. Reads only; sends nothing."""
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tok:
        sys.exit("set TELEGRAM_BOT_TOKEN first (do not paste it into a file)")
    try:
        with urllib.request.urlopen(
                f"https://api.telegram.org/bot{tok}/getUpdates", timeout=30) as r:
            d = json.loads(r.read().decode())
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"could not reach Telegram: {scrub(e)}")
    if not d.get("ok"):
        sys.exit(f"Telegram rejected the token: {scrub(d.get('description', d))}")
    seen = {}
    for u in d.get("result", []):
        c = (u.get("message") or u.get("channel_post") or {}).get("chat") or {}
        if c.get("id"):
            seen[c["id"]] = c.get("username") or c.get("title") or c.get("first_name") or "?"
    if not seen:
        sys.exit("No messages yet. Send your bot any message, then run this again.")
    for cid, who in seen.items():
        print(f"  TELEGRAM_CHAT_ID={cid}   ({who})")


def send(title, body):
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        # <pre> so the column alignment survives Telegram's proportional font.
        payload = json.dumps({"chat_id": chat, "parse_mode": "HTML",
                              "disable_web_page_preview": True,
                              "text": f"<pre>{html.escape(body)}</pre>"}).encode()
        _post(f"https://api.telegram.org/bot{tok}/sendMessage", payload,
              {"Content-Type": "application/json"})
        return "telegram"

    topic = os.getenv("NTFY_TOPIC")
    if topic:
        server = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        # ntfy puts the title in an HTTP header, which must stay ASCII.
        hdr = {"Title": title.encode("ascii", "replace").decode(),
               "Tags": "chart_with_upwards_trend", "Click": PAGE}
        if os.getenv("NTFY_TOKEN"):
            hdr["Authorization"] = "Bearer " + os.environ["NTFY_TOKEN"]
        _post(f"{server}/{topic}", body.encode(), hdr)
        return "ntfy"

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually deliver it")
    ap.add_argument("--no-health", action="store_true")
    ap.add_argument("--whoami", action="store_true",
                    help="print the chat_id for the configured bot, then exit")
    a = ap.parse_args()

    if a.whoami:
        return whoami()

    L = json.load(open(EXPORT))
    hp = {"n": 0, "fails": 0, "warns": 0, "failing": []} if a.no_health else health()
    body = build(L, hp)
    title = f"MARK6 day {L['days_live']}: {pct(L['return_pct'])} ({rs(float(L['nav']) - float(L['capital']), True)})"

    print(body)
    if not a.send:
        return
    try:
        via = send(title, body)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        # A failed send must never fail the refresh job — the money record is the
        # deliverable, the notification is a convenience on top of it.
        print(f"\n  notify: delivery FAILED ({scrub(e)}) — the day's mark is still recorded",
              file=sys.stderr)
        return
    print(f"\n  notify: sent via {via}" if via else
          "\n  notify: no channel configured (set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID or NTFY_TOPIC)")


if __name__ == "__main__":
    main()
