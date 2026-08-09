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
  * Unrealised and realised gains are never blended into one figure.
  * The weakness goes IN the message, not in a footnote — the worst dip and the
    simulation disclosure appear every single day, including good days. The
    wording is a tear-sheet footer rather than a warning label because this gets
    shown to other people, but it is never absent: an unlabelled simulated track
    record is not a professional artifact, it is a misrepresentation, and every
    institution that publishes one labels it for exactly that reason.
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
import csv
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

# The worst peak-to-trough dip the deployed 50/25/25 book took across 2007-2026,
# measured 2026-08-09 and disclosed on the page. Quoted whenever a live drawdown
# is reported, so the live number always arrives next to the worst one known —
# a dip is frightening in isolation and ordinary against its own history.
# Shorter windows flatter it: the same book shows -23.78% over 2016-2026 only
# because that window has no 2008 in it.
BACKTEST_WORST_DD = -41.80

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
LEDGER = os.path.join(_ROOT, "data", "paper", "paper_ledger.csv")


def todays_trades(asof):
    """Fills booked on `asof`, straight from the append-only ledger.

    Read from the ledger rather than from any summary, because the ledger is the
    record — a summary can disagree with it, and if the two ever diverge the one
    the owner is shown must be the one that is authoritative.
    """
    try:
        with open(LEDGER) as fh:
            return [r for r in csv.DictReader(fh) if r.get("date") == asof]
    except (OSError, csv.Error):
        return []


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

    # On a rebalance day this is the most important thing in the message: the
    # system re-picked the book by itself and this is the record of what it did.
    # Reported as completed fills, not as a to-do list — there is no human step.
    tr = todays_trades(asof)
    if tr:
        out += ["", f"THE SYSTEM REBALANCED ON {asof}"]
        for r in tr:
            # NOT `pnl` — that name holds the book's profit and is used below.
            # Shadowing it here would have crashed the message on precisely the
            # one day it exists to report.
            note, leg = r.get("note", ""), ""
            if "P&L" in note:
                leg = note.split("P&L", 1)[1].split("·")[0].strip()
                leg = f"  {leg.replace('+', '+Rs ').replace('-', '-Rs ')}"
            out.append(f"  {r['action'].lower():<5} {r['ticker'][:12]:<12}"
                       f"{int(float(r['qty'])):>5} @ {float(r['price']):>9,.2f}{leg}")
        out.append(f"  {len(tr)} orders · costs and tax already deducted")

    real = float(L.get("realised_pnl", 0.0))
    out += ["", "WORTH KNOWING",
            f"  · worst dip so far: {pct(L['max_drawdown_pct'], 1, False)} below its own"
            f"\n    peak. It will be worse than that one day.",
            f"  · only {rs(real, True)} is locked in by actually"
            f"\n    selling. The other {rs(pnl - real, True)} is on paper"
            f"\n    and can still go away."]
    if float(L.get("tax_liability", 0)) > 0:
        out.append(f"  · tax owed if it all ended today: {rs(L['tax_liability'])}")
    out.append("  · model portfolio — simulated execution")

    if hp["fails"] > 0:
        out += ["", f"⚠ SYSTEM  {hp['fails']} of {hp['n']} checks FAILED"]
        out += [f"    ✗ {c}" for c in hp["failing"][:6]]
    elif hp["fails"] < 0:
        out += ["", "⚠ SYSTEM  the health check itself did not run", f"    {hp['failing'][0][:70]}"]
    elif hp["n"] == 0:
        # Zero checks run is NOT zero checks failed. The old wording rendered this
        # as "all 0 checks passed", which reads as reassurance and is the exact
        # inversion this notifier exists to prevent: nothing ran, and the message
        # said everything was fine.
        out += ["", "SYSTEM  not checked on this run"]
    else:
        out += ["", f"SYSTEM  all {hp['n']} checks passed"
                    + (f", {hp['warns']} warning(s)" if hp["warns"] else "")]

    # Tear-sheet footer. Every institution that publishes a simulated track
    # record labels it on the artifact itself, because the label is what makes it
    # a track record rather than a claim. Wording is professional; the disclosure
    # is not optional and test_the_simulation_is_always_disclosed enforces it.
    out += ["", "─" * W,
            "Model portfolio. Simulated execution at live",
            "market prices, net of costs and Indian tax.",
            "Not investment advice.", PAGE]
    return "\n".join(out)


# ── the exceptional day ──────────────────────────────────────────────────
# A message that looks identical every day stops being read. Three months in it
# is wallpaper, and the one day it says something urgent it gets skimmed past
# with the rest. So the routine message keeps its shape and an EXTRA one is sent
# only when the day was genuinely unusual — that second message is what carries
# the alarm, and its rarity is what makes it credible.

BIG_MOVE_SIGMA = 3.0     # a move this many times a normal day's size
MIN_OBS = 30             # ...measured from the book's own history once there is enough
FALLBACK_DAILY_VOL = 0.86    # %, implied by the deployed book's ~13.6% annual vol
MIN_MOVE_PCT = 1.0       # floor, so a becalmed book cannot alert on a rounding error
DD_BAND = 5              # a new-low alert fires once per 5-point band...
DD_FLOOR = 10            # ...and not at all until the book is this far down


def alert(L):
    """The second message, or None on an ordinary day.

    The threshold is in units of the book's OWN daily volatility rather than a
    fixed percentage, for two reasons. A fixed 2.5% is ~3 sigma for today's
    50/25/25 book but nearly unreachable for January's four-sleeve version, which
    is deliberately calmer — so a fixed number would quietly stop firing at the
    exact moment the allocation changed. And "today was 3.6x a normal day" is a
    sentence the owner can act on; "today was -2.7%" is not, without knowing what
    normal looks like.
    """
    hist = L.get("nav_history") or []
    navs = [float(h["nav_inr"]) for h in hist
            if str(h.get("nav_inr", "")).strip() not in ("", "None")]
    if len(navs) < 3:
        return None
    rets = [(navs[i] / navs[i - 1] - 1) * 100 for i in range(1, len(navs))]
    today = rets[-1]

    if len(rets) >= MIN_OBS:
        mu = sum(rets) / len(rets)
        sigma = (sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    else:
        sigma = FALLBACK_DAILY_VOL
    sigma = max(sigma, 1e-9)

    # Current drawdown, and yesterday's, each against the peak known AT THE TIME.
    # Comparing today's dip to a peak that had not happened yet would invent
    # crossings that never occurred.
    dd_now = (navs[-1] / max(navs) - 1) * 100
    dd_prev = (navs[-2] / max(navs[:-1]) - 1) * 100
    band = lambda d: int(-d // DD_BAND) * DD_BAND            # noqa: E731
    new_low = band(dd_now) > band(dd_prev) and -dd_now >= DD_FLOOR

    big = abs(today) >= max(BIG_MOVE_SIGMA * sigma, MIN_MOVE_PCT)
    if not (big or new_low):
        return None

    nav, cap = float(L["nav"]), float(L["capital"])
    move_rs = nav - nav / (1 + today / 100)
    out = []
    if big:
        out += [f"{'BIG UP DAY' if today > 0 else 'BIG DOWN DAY'}  {pct(today)}",
                "─" * W,
                _row("moved today", rs(move_rs, True)),
                _row("worth now", rs(nav)),
                f"  about {abs(today) / sigma:.1f}x a normal day"]
    if new_low:
        peak = max(navs)
        out += ([""] if out else []) + [
            f"NEW LOW  {pct(dd_now, 1)} below its peak", "─" * W,
            _row("below the peak", rs(nav - peak, True)),
            _row("still ahead of cost", rs(nav - cap, True))]

    # The paragraph that matters. This message arrives on precisely the day the
    # owner is most likely to want to do something, and the research log says
    # every version of "do something" tested worse: six separate approaches died
    # for cutting exposure after a loss, because Indian equity's drift is positive
    # and its recoveries are V-shaped, so forward expected return is HIGHEST
    # exactly when those rules sell. Saying so here is not reassurance, it is the
    # finding.
    out += ["", "WHAT HAPPENS NOW", "─" * W,
            "  Nothing. No trade was made and none is",
            "  scheduled — the book is only re-picked on",
            "  its cadence, whatever the day did.",
            "",
            f"  The 19-year test dipped {pct(BACKTEST_WORST_DD, 1, False)} at its",
            "  worst and recovered. Every version of this",
            "  system that cut exposure after a loss scored",
            "  worse, because the rebound is what pays.",
            "", "─" * W, PAGE]
    return "\n".join(out)


# ── the other messages that are not daily ────────────────────────────────
BOOK = os.path.join(_ROOT, "data", "paper", "paper_book.json")
REBAL_DAYS = 182
NOTICE_DAYS = 7          # how far ahead the rebalance is announced
LTCG_WATCH_DAYS = 30


def _book():
    try:
        return json.load(open(BOOK))
    except (OSError, ValueError):
        return {}


def _days(a, b):
    return (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime(b, "%Y-%m-%d")).days


def monthly(L):
    """A differently-shaped message once a month, on the first mark of a new one.

    The daily message is identical every day by design, and over six months of
    absence that is exactly what turns it into wallpaper. This is the same money
    described a different way, so it has to be read rather than recognised.
    """
    hist = [h for h in (L.get("nav_history") or [])
            if str(h.get("nav_inr", "")).strip()]
    if len(hist) < 3 or hist[-1]["date"][:7] == hist[-2]["date"][:7]:
        return None                       # still inside the same month
    month = hist[-2]["date"][:7]
    inside = [h for h in hist if h["date"][:7] == month]
    if len(inside) < 3:
        return None
    start = hist[hist.index(inside[0]) - 1] if hist.index(inside[0]) else inside[0]
    n0, n1 = float(start["nav_inr"]), float(inside[-1]["nav_inr"])
    move = (n1 / n0 - 1) * 100

    out = [f"MONTH IN REVIEW  ·  {month}", "─" * W,
           _row(f"the month ({pct(move)})", rs(n1 - n0, True)),
           _row("ended at", rs(n1))]
    b0 = b1 = None
    for h in inside:
        if str(h.get("bench_inr", "")).strip():
            b1 = float(h["bench_inr"])
            b0 = b0 if b0 is not None else b1
    if b0 and b1 and b0 > 0:
        bm = (b1 / b0 - 1) * 100
        out += [_row("the index did", pct(bm)),
                _row("you were", f"{move - bm:+.2f}pp {'ahead' if move >= bm else 'behind'}")]

    # Stocks only. The sleeves are whole asset classes bought on purpose, so
    # crowning GOLDBEES "best holding" would credit the stock picker for a
    # decision it did not make.
    sleeves = set((L.get("config") or {}).get("sleeve_targets") or {})
    rows = sorted((h for h in (L.get("holdings") or []) if h["ticker"] not in sleeves),
                  key=lambda h: -float(h["pnl_pct"]))
    if rows:
        out += ["", "THE STOCKS, SINCE THE BOOK OPENED", "─" * W,
                _row(f"best   {rows[0]['ticker'][:12]}", pct(float(rows[0]["pnl_pct"]), 1)),
                _row(f"worst  {rows[-1]['ticker'][:12]}", pct(float(rows[-1]["pnl_pct"]), 1))]
    out += ["",
            "  One month is noise. It is here so the daily",
            "  message does not become the only thing you",
            "  see, not because a month means anything.",
            "", "─" * W, PAGE]
    return "\n".join(out)


def rebalance_notice(L):
    """Announce the rebalance BEFORE it happens, not after.

    It is the only scheduled event in a six-month absence, and until now it
    arrived with no warning at all: the owner would learn the book had been
    re-picked from the message reporting the fills. A week's notice also means a
    failure in the machinery surfaces while there is still time to fix it.
    """
    book = _book()
    last = book.get("last_rebalance") or book.get("start_date")
    asof = (L.get("nav_history") or [{}])[-1].get("date")
    if not last or not asof:
        return None
    due_in = REBAL_DAYS - _days(asof, last)
    if not 0 < due_in <= NOTICE_DAYS:
        return None

    out = [f"REBALANCE IN {due_in} DAY{'S' if due_in != 1 else ''}", "─" * W,
           "  The system will re-pick the stock sleeve by",
           "  itself and post every fill here. There is no",
           "  step for you, and nothing to approve."]
    try:
        sig = json.load(open(os.path.join(_ROOT, "data", "paper", "signals.json")))
        held = {h["ticker"] for h in (L.get("holdings") or [])}
        sleeves = set((L.get("config") or {}).get("sleeve_targets") or {})
        n_hold = (L.get("config") or {}).get("n_hold") or 20
        top = {t for t, _ in sorted(sig["scores"].items(),
                                    key=lambda kv: kv[1]["rank"])[:n_hold]}
        stocks = held - sleeves
        out += ["", "ON TODAY'S RANKING IT WOULD", "─" * W,
                _row("keep", f"{len(top & stocks)} of {len(stocks)}"),
                _row("sell", f"{len(stocks - top)}"),
                _row("buy", f"{len(top - stocks)}"),
                "", f"  out  {', '.join(sorted(stocks - top))[:64] or '—'}",
                f"  in   {', '.join(sorted(top - stocks))[:64] or '—'}",
                "",
                f"  Ranking as of {sig.get('asof', '?')}. The real",
                "  decision is made on the day, on that day's",
                "  prices, so this is an indication."]
    except (OSError, ValueError, KeyError):
        out += ["", "  (no ranking recorded to preview against)"]
    out += ["", "─" * W, PAGE]
    return "\n".join(out)


def tax_watch(L):
    """Holdings about to cross the 365-day line.

    Worth 7.5 percentage points of the gain — 20% short-term against 12.5%
    long-term — and nothing else in the system mentions it. Reported as
    information, NOT as a plan: holding a deranked name to reach the date was
    tested and lost 0.23pp net, so the engine deliberately does not wait.
    """
    book, asof = _book(), (L.get("nav_history") or [{}])[-1].get("date")
    if not asof:
        return None
    soon = []
    for h in (L.get("holdings") or []):
        e = ((book.get("positions") or {}).get(h["ticker"]) or {}).get("entry_date")
        if not e:
            continue
        left = 366 - _days(asof, e)
        if 0 < left <= LTCG_WATCH_DAYS:
            soon.append((left, h["ticker"], float(h["pnl"])))
    if not soon:
        return None
    out = [f"{len(soon)} HOLDING(S) TURN LONG-TERM SOON", "─" * W]
    for left, t, pnl in sorted(soon):
        out.append(f"  {t[:12]:<13}{left:>3} days   gain {_amt_local(pnl)}")
    out += ["",
            "  After 365 days the tax on a gain drops from",
            "  20% to 12.5%, and the first Rs 1,25,000 of",
            "  long-term gain each year is exempt.",
            "",
            "  The system does NOT wait for these dates.",
            "  Holding a deranked name to reach one was",
            "  tested and lost 0.23pp net. This is here to",
            "  be known, not acted on.",
            "", "─" * W, PAGE]
    return "\n".join(out)


def _amt_local(x):
    return f"{'+' if x >= 0 else '-'}Rs {_grp(f'{abs(float(x)):.0f}')}"


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
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        # Telegram puts the ONLY useful part of a failure in the response body —
        # "chat not found", "bot is not a member of the group chat", "not enough
        # rights to send text messages". Letting HTTPError propagate bare turns
        # all of those into "HTTP Error 400: Bad Request", which is the same
        # message for every possible cause and sends you guessing.
        try:
            body = json.loads(e.read().decode())
        except Exception:                                    # noqa: BLE001
            raise e from None
        desc = body.get("description", "")
        # A group silently becomes a supergroup and its id changes. Telegram hands
        # back the new one; without this you are told "chat not found" about a
        # chat you are looking at on screen.
        new_id = (body.get("parameters") or {}).get("migrate_to_chat_id")
        if new_id:
            desc += (f"  — this group became a SUPERGROUP and its id changed. "
                     f"Set TELEGRAM_CHAT_ID={new_id}")
        raise RuntimeError(f"Telegram refused it: {desc}") from None


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
            seen[c["id"]] = (c.get("title") or c.get("username")
                             or c.get("first_name") or "?", c.get("type", "?"))
    if not seen:
        sys.exit("No messages yet. Send the bot a message — in the GROUP if you want the "
                 "group's id — then run this again. Bots only ever see chats that have "
                 "spoken to them.")
    # Type matters: a private id is positive, a group/supergroup id is negative, and
    # they are easy to mix up when copying. Sending to the wrong one fails silently
    # from the sender's side — the message simply arrives somewhere else.
    for cid, (who, kind) in sorted(seen.items(), key=lambda kv: kv[0]):
        star = "  <- group" if kind in ("group", "supergroup") else ""
        print(f"  TELEGRAM_CHAT_ID={cid:<16} {kind:<11} {who}{star}")


def send(title, body, chat=None):
    # `chat` overrides the configured destination so bot.py can answer in the
    # chat that asked. Without it a question sent in a DM would be answered into
    # the group, which is both confusing and a small privacy leak.
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat or os.getenv("TELEGRAM_CHAT_ID")
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


def diagnose():
    """Walk the chain token -> chat -> permission -> send and say which link broke.

    "It doesn't work" has at least five distinct causes here — wrong token, wrong
    chat id, bot not in the chat, bot lacking send rights, or an id that changed
    under a supergroup upgrade — and every one of them looks identical from the
    outside: no message arrives. This prints the first one that actually fails.
    """
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    print(f"  token set : {'yes' if tok else 'NO'}")
    print(f"  chat id   : {chat or 'NOT SET'}")
    if not tok:
        sys.exit("  -> set TELEGRAM_BOT_TOKEN first")

    def api(m, payload=None):
        url = f"https://api.telegram.org/bot{tok}/{m}"
        try:
            if payload is None:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return json.loads(r.read().decode())
            return json.loads(_post(url, json.dumps(payload).encode(),
                                    {"Content-Type": "application/json"}))
        except Exception as e:                               # noqa: BLE001
            return {"ok": False, "description": scrub(e)}

    me = api("getMe")
    if not me.get("ok"):
        sys.exit(f"  -> TOKEN REJECTED: {me.get('description')}")
    print(f"  bot       : @{me['result'].get('username')}  (token is valid)")

    ups = api("getUpdates")
    chats = {}
    for u in ups.get("result", []):
        c = (u.get("message") or u.get("channel_post") or {}).get("chat") or {}
        if c.get("id"):
            chats[c["id"]] = (c.get("title") or c.get("username")
                              or c.get("first_name") or "?", c.get("type", "?"))
    print(f"  chats seen: {len(chats)}")
    for cid, (who, kind) in sorted(chats.items()):
        mark = "  <- currently configured" if str(cid) == str(chat) else ""
        print(f"      {cid:<16} {kind:<11} {who}{mark}")
    if not chats:
        print("      (none — send /start@<bot> IN the group, then re-run)")

    if not chat:
        sys.exit("  -> set TELEGRAM_CHAT_ID to one of the ids above")

    info = api("getChat", {"chat_id": chat})
    if not info.get("ok"):
        sys.exit(f"  -> CANNOT REACH THAT CHAT: {info.get('description')}\n"
                 f"     Usual cause: the bot was never added to it, or the id is a DM "
                 f"id when you meant the group (group ids are NEGATIVE).")
    r = info["result"]
    print(f"  target    : {r.get('title') or r.get('username')} ({r.get('type')})")

    sent = api("sendMessage", {"chat_id": chat, "text": "MARK6 connectivity test."})
    if sent.get("ok"):
        print("\n  RESULT: sent. If it is not visible, you are looking at a different chat.")
    else:
        print(f"\n  RESULT: SEND FAILED — {sent.get('description')}")
        print("     'not enough rights'  -> the bot is in the group but cannot post; "
              "give it Send Messages, or make it admin.")
        print("     'chat not found'     -> wrong id, or the bot was removed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually deliver it")
    ap.add_argument("--no-health", action="store_true")
    ap.add_argument("--whoami", action="store_true",
                    help="print the chat_id for the configured bot, then exit")
    ap.add_argument("--diagnose", action="store_true",
                    help="walk token -> chat -> permission -> send and report the break")
    a = ap.parse_args()

    if a.diagnose:
        return diagnose()
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
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError) as e:
        # A failed send must never fail the refresh job — the money record is the
        # deliverable, the notification is a convenience on top of it.
        print(f"\n  notify: delivery FAILED ({scrub(e)}) — the day's mark is still recorded",
              file=sys.stderr)
        return
    print(f"\n  notify: sent via {via}" if via else
          "\n  notify: no channel configured (set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID or NTFY_TOPIC)")

    # Sent AFTER the daily message and only on an unusual day, so it lands last
    # and reads as the exception it is. Failure here must not disturb anything
    # above it — the mark is recorded, the routine message is already delivered.
    # Each is silent on an ordinary day and speaks only when its own condition is
    # met, so the count of messages is itself information: one means nothing
    # happened, two means something did. Any of them failing to build must not
    # take the others down — the daily message is already delivered by here.
    for label, fn in (("alert", alert), ("rebalance notice", rebalance_notice),
                      ("tax watch", tax_watch), ("month in review", monthly)):
        try:
            extra = fn(L)
        except Exception as e:                               # noqa: BLE001
            print(f"  notify: {label} could not be built ({scrub(e)})", file=sys.stderr)
            continue
        if not extra:
            continue
        print("\n" + extra)
        try:
            send(f"MARK6 {label}", extra)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError) as e:
            print(f"  notify: {label} delivery FAILED ({scrub(e)})", file=sys.stderr)


if __name__ == "__main__":
    main()
