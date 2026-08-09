"""Two-way Telegram — ask the system a question, get an answer back.
====================================================================
notify.py speaks once a day and then goes quiet. This is the other direction:
the owner types /update at 11pm on a Tuesday and the system answers.

WHY IT POLLS INSTEAD OF USING A WEBHOOK
---------------------------------------
A webhook needs a public HTTPS endpoint that is up 24/7. This system has no
server — it is a git repository plus a scheduled runner, deliberately, because
that is what survives six months of nobody looking at it. So the bot runs the
only place it can: inside GitHub Actions, on a cron, long-polling Telegram for
whatever was said while it was asleep.

The consequence, stated plainly because the owner will notice it: the FIRST
command after a quiet period can take up to ~10 minutes to answer. Telegram
holds the message until the next run collects it, so nothing is ever lost —
it is late, not dropped. Once a run has answered one command it stays awake
(ACTIVE_EXIT below), so a real back-and-forth replies in about a second.

NOTHING HERE CAN TOUCH THE MONEY RECORD, AND THAT IS A DESIGN DECISION
----------------------------------------------------------------------
Mandate §6: the book is an append-only integrity record, never rebalanced
off-cadence, never stamped with a mid-session price. A chat message is the
single worst authorisation mechanism for a write to that record — it is one
fat-finger from a fill nobody scheduled, in a file whose entire value is that
nothing unscheduled is in it. So the workflow that runs this grants
`contents: read` and there is nothing here that could write even if it tried.

Reading is a different question, and reading is genuinely useful. That is
almost the whole of what this does — the single exception is /clear, which
deletes messages in the Telegram chat and cannot reach anything else.

  python3 scripts/bot.py            # one drain-and-reply pass, then exit
  python3 scripts/bot.py --serve    # the polling window the workflow runs
  python3 scripts/bot.py --dry      # print what it would answer, send nothing
  python3 scripts/bot.py --say /why BHEL   # render one command locally
"""
import argparse
import html
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from notify import (EXPORT, PAGE, _grp, build, health, pct,     # noqa: E402
                    rs, scrub, send)

BOOK = os.path.join(_ROOT, "data", "paper", "paper_book.json")

# Cheap when nobody is talking, responsive when they are. An idle run costs ~30s
# of runner time; only a run that actually hears something stays up.
IDLE_EXIT = 30       # silence before a quiet run gives up
ACTIVE_EXIT = 150    # silence to wait for a follow-up after answering
HARD_CAP = 540       # ceiling, must stay under the cron interval
POLL = 25            # long-poll held open per request

# Telegram's hard limit is 4096 characters and it rejects the whole message on
# overflow — a silent total failure, not a truncation. Split below the line.
CHUNK = 3400


# ── transport ────────────────────────────────────────────────────────────
def _api(method, **params):
    """One Telegram call. Raises RuntimeError carrying Telegram's own reason."""
    tok = os.environ["TELEGRAM_BOT_TOKEN"]
    url = f"https://api.telegram.org/bot{tok}/{method}"
    data = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}).encode()
    # Socket timeout must outlive the long-poll or urllib kills a healthy wait.
    wait = int(params.get("timeout") or 0) + 20
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, data=data), timeout=wait) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:                                    # noqa: BLE001
            raise RuntimeError(scrub(e)) from None
        raise RuntimeError(f"{e.code} {body.get('description', '')}") from None
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(scrub(e)) from None


def allowed_chats():
    """Who may command this bot.

    A Telegram bot is reachable by anyone who learns its @username — there is no
    such thing as a private bot. Without this every stranger who finds it can
    read the book's positions and P&L. TELEGRAM_CHAT_ID is the owner's channel;
    TELEGRAM_ADMIN_CHATS adds any extra ids (comma separated) such as a DM
    alongside the group.
    """
    ids = {os.getenv("TELEGRAM_CHAT_ID", "").strip()}
    ids |= {c.strip() for c in os.getenv("TELEGRAM_ADMIN_CHATS", "").split(",")}
    return {i for i in ids if i}


# ── the answers ──────────────────────────────────────────────────────────
W = 40


def _export():
    return json.load(open(EXPORT))


def _asof(L):
    hist = L.get("nav_history") or []
    return hist[-1]["date"] if hist else L.get("generated", "")[:10]


def h_update(arg=""):
    """The daily message, on demand. Same builder, so the answer to 'how is it
    going' is byte-identical whether it was pushed or pulled — two formatters
    would eventually disagree and the owner would have no way to know which
    one lied."""
    return build(_export(), health())


def _amt(x, signed=False):
    """Rupees, no 'Rs ' prefix. A 22-row table has to fit a phone screen without
    wrapping — a wrapped monospace table is unreadable — and the unit is stated
    once in the header instead of 44 times inside it."""
    s = ("+" if x >= 0 else "-") if signed else ("-" if x < 0 else "")
    return f"{s}{_grp(f'{abs(float(x)):.0f}')}"


def _line(h):
    return (f"  {h['ticker'][:11]:<11}{_amt(h['value']):>10}"
            f"{_amt(h['pnl'], True):>9}{pct(h['pnl_pct'], 1):>7}")


def h_holdings(arg=""):
    """Split the passive sleeves out from the stocks, because a flat list of 22
    lines reads as "we hold 22 stocks" and the system holds 20. The other two are
    a whole sleeve each — one ETF standing in for gold, one for the Nasdaq-100 —
    and they are the two largest lines on the page. `n_hold` governs the equity
    sleeve alone; when the gilt sleeve lands this becomes 23 lines and still 20
    stocks. Presenting them as peers of BHEL invites exactly the wrong count.

    Which lines are sleeves is read off the export's own sleeve table rather than
    hardcoded here, so adding the gilt sleeve needs no change in this file.
    """
    L = _export()
    rows = sorted(L.get("holdings") or [], key=lambda h: -float(h["pnl"]))
    if not rows:
        return "No positions recorded yet."

    sleeve_of = {round(float(r["value_inr"]), 2): r["label"]
                 for r in (L.get("sleeves") or {}).get("rows") or []
                 if r.get("passive") and r.get("n_holdings") == 1}
    etfs = [h for h in rows if round(float(h["value"]), 2) in sleeve_of]
    stocks = [h for h in rows if h not in etfs]

    out = [f"HOLDINGS  as of {_asof(L)}",
           f"  {len(stocks)} stocks + {len(etfs)} whole-sleeve ETFs",
           "  all figures in rupees"]
    if etfs:
        out += ["─" * W, "SLEEVES IT JUST BUYS AND HOLDS",
                f"  {'':<11}{'worth':>10}{'P&L':>9}{'':>7}"]
        for h in etfs:
            out.append(_line(h) + f"\n    = {sleeve_of[round(float(h['value']), 2)]}")
    out += ["─" * W, f"THE {len(stocks)} STOCKS THE SYSTEM PICKED",
            f"  {'':<11}{'worth':>10}{'P&L':>9}{'':>7}"]
    for h in stocks:
        out.append(_line(h))
    # The column above is UNREALISED — it is what the positions are worth today.
    # /update reports a different, smaller number, because that one also carries
    # the loss already banked on names that were sold. Two screens showing two
    # profits with no bridge between them is how a reader stops trusting both, so
    # the bridge is printed here rather than left for them to work out.
    unreal = sum(float(h["pnl"]) for h in rows)
    real = float(L.get("realised_pnl", 0.0))
    out += ["─" * W,
            f"  {'still held':<11}{_amt(sum(float(h['value']) for h in rows)):>10}"
            f"{_amt(unreal, True):>9}",
            f"  {'already sold':<21}{_amt(real, True):>9}",
            f"  {'YOUR PROFIT':<21}{_amt(unreal + real, True):>9}",
            "",
            "Sorted by rupees made or lost, not by percent —",
            "a big gain on a tiny position is not a big gain.",
            "",
            "'Still held' can still go away; only 'already",
            "sold' is banked.", PAGE]
    return "\n".join(out)


def h_next(arg=""):
    """Ask paper_track the same question the scheduler asks it.

    Deliberately a subprocess to the exact command refresh.yml runs, rather than
    re-deriving the date here. A second copy of the cadence rule would drift from
    the first, and the owner would be told a date the system does not act on.
    """
    p = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "paper_track.py"),
                        "rebalance", "--check"],
                       capture_output=True, text=True, cwd=_ROOT, timeout=120)
    line = (p.stdout.strip().splitlines() or ["(no answer)"])[-1]

    out = ["NEXT REBALANCE", "─" * W]
    if line.startswith("DUE"):
        out += ["  A rebalance is DUE.", "",
                "  The next scheduled run will re-pick the",
                "  stocks by itself and post every fill here.",
                "  There is no step for you."]
    else:
        date = line.split("next ~")[-1].rstrip(")") if "next ~" in line else "?"
        days = line.split("(")[-1].split("d to go")[0] if "d to go" in line else "?"
        out += [f"  date        {date:>{W - 14}}",
                f"  days away   {days:>{W - 14}}", "",
                "  Until then the book is left alone. That is",
                "  the strategy, not neglect — trading it more",
                "  often tested worse, after tax."]

    try:
        b = json.load(open(BOOK))
        out += ["", "HOW IT WILL DECIDE", "─" * W,
                f"  stocks it will hold {int(b.get('n_hold', 0)):>{W - 22}}",
                f"  chosen from         {int(b.get('top_n_liquid', 0)):>{W - 22}}",
                f"  rebalances so far   {len(b.get('rebalances') or []):>{W - 22}}",
                "",
                "  It ranks every liquid stock on momentum and",
                "  delivery-volume factors, keeps the top names",
                "  and sells the rest. No human picks anything."]
    except (OSError, ValueError, KeyError):
        pass
    out += ["", PAGE]
    return "\n".join(out)


def h_health(arg=""):
    hp = health()
    out = ["SYSTEM CHECK", "─" * W]
    if hp["fails"] < 0:
        out += ["  ⚠ the check itself could not run", f"  {hp['failing'][0][:W * 2]}"]
    elif hp["fails"]:
        out += [f"  ⚠ {hp['fails']} of {hp['n']} checks FAILED"]
        out += [f"    ✗ {c}" for c in hp["failing"][:10]]
    else:
        out += [f"  all {hp['n']} checks passed"
                + (f", {hp['warns']} warning(s)" if hp["warns"] else "")]
    out += ["", "Covers the ledger's integrity hash, the price",
            "freshness guard, and whether the published page",
            "agrees with the numbers behind it."]
    return "\n".join(out)


def _series(hist, key):
    """(dates, values) for one series, blanks DROPPED and dates kept as datetimes.

    Two bugs live here, both of which produce a chart that looks fine.

    Validate before coercing: `bench_inr` is "" on days the benchmark had no
    print, and float("" or 0) is 0.0 — which plots as a vertical crash to the
    bottom of the axis on a chart whose entire job is the comparison.

    Return datetimes, never the date STRINGS. Strings make matplotlib build a
    CATEGORICAL axis ordered by first appearance across all plotted series. The
    benchmark has no print on day one, so that date was missing from the
    categories it established, got appended at the far RIGHT, and the NAV line
    drew a phantom segment looping back to it.
    """
    pts = [(datetime.strptime(h["date"], "%Y-%m-%d"), h[key]) for h in hist
           if str(h.get(key, "")).strip() not in ("", "None")]
    return [p[0] for p in pts], [float(p[1]) for p in pts]


class Photo:
    """A reply that is an image rather than text."""

    def __init__(self, png, caption):
        self.png, self.caption = png, caption


def h_chart(arg=""):
    """The book against the index, as a picture.

    The daily message carries a 20-character sparkline, which answers "roughly
    which way" and nothing else. The question this answers is the one actually
    being asked — am I ahead of just buying the index, and by how much — and a
    line pair answers it in less time than reading two numbers and subtracting.

    matplotlib is imported here, not at module scope, so a broken or missing
    plotting stack costs this one command instead of the whole bot.
    """
    import matplotlib
    matplotlib.use("Agg")                        # no display on a CI runner
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    L = _export()
    hist = L.get("nav_history") or []

    # Validate BEFORE coercing. `bench_inr` is "" on days the benchmark had no
    # print, and float("" or 0) silently becomes zero — which plots as a crash to
    # the bottom of the axis on a chart whose whole job is to show a comparison.
    d_nav, nav = _series(hist, "nav_inr")
    d_ben, ben = _series(hist, "bench_inr")
    if len(nav) < 2:
        return "Not enough history to draw yet — it needs at least two marks."

    cap = float(L["capital"])
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(cap, color="#b0b0b0", lw=1, ls=(0, (4, 3)), zorder=1)
    if len(ben) > 1:
        ax.plot(d_ben, ben, color="#9aa0a6", lw=1.6, zorder=2,
                label="if you'd just bought the index")
    up = nav[-1] >= cap
    ax.plot(d_nav, nav, color="#1a7f37" if up else "#b3261e", lw=2.4, zorder=3,
            label="your money")
    ax.scatter([d_nav[-1]], [nav[-1]], s=34, zorder=4,
               color="#1a7f37" if up else "#b3261e")
    ax.annotate(rs(nav[-1]), (d_nav[-1], nav[-1]), textcoords="offset points",
                xytext=(-6, 10), ha="right", fontsize=10, fontweight="bold",
                color="#1a7f37" if up else "#b3261e")

    ax.set_title(f"MARK6 · day {L['days_live']} · {pct(L['return_pct'])}",
                 fontsize=12, loc="left", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rs(v)))
    # One label every ~6 marks: a tick per trading day is unreadable at phone size.
    step = max(1, len(d_nav) // 6)
    ax.set_xticks(d_nav[::step])
    ax.set_xticklabels([d.strftime("%d %b") for d in d_nav[::step]], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)

    pnl = float(L["nav"]) - cap
    cap_txt = (f"{'Profit' if pnl >= 0 else 'Loss'} {rs(pnl, True)} on {rs(cap)} "
               f"since {L.get('start_date', '?')}.")
    if L.get("benchmark_nav"):
        cap_txt += (f" {'Ahead of' if float(L['nav']) >= float(L['benchmark_nav']) else 'Behind'}"
                    f" the index by {float(L['relative_pct']):+.2f}pp.")
    cap_txt += " Dotted line is what you put in. Model portfolio."
    return Photo(buf.getvalue(), cap_txt)


SIGNALS = os.path.join(_ROOT, "data", "paper", "signals.json")

FACTOR_PLAIN = {
    "momentum":  "went up more than most",
    "trend":     "went up steadily, not in one jump",
    "low_vol":   "did not swing around wildly",
    "stability": "behaved consistently",
    "deliv_chg": "buyers took delivery, not day trades",
}


def _bar(p, n=8):
    """A percentile as a bar. 0.87 needs converting in the reader's head; a bar
    does not, and this is read on a phone by someone who is not a trader."""
    return "█" * int(round((p or 0) * n)) + "░" * (n - int(round((p or 0) * n)))


def _fundamentals(ticker):
    """Basic company figures from a free third-party feed.

    Fails OPEN and quietly. This is the only part of /why depending on a server
    nobody here controls, and the reasoning above it must still render when that
    server is down, rate-limited, or has renamed a field. Losing the whole
    explanation because a supplementary block failed is the worse trade.
    """
    try:
        import yfinance as yf
        i = yf.Ticker(f"{ticker}.NS").info or {}
    except Exception:                                        # noqa: BLE001
        return []

    def cr(v):
        return f"Rs {_grp(f'{float(v) / 1e7:.0f}')} cr" if isinstance(v, (int, float)) else None

    def num(v, nd=1):
        return f"{float(v):.{nd}f}" if isinstance(v, (int, float)) else None

    roe = i.get("returnOnEquity")
    rows = [("sector", i.get("sector")),
            ("market cap", cr(i.get("marketCap"))),
            ("revenue 12m", cr(i.get("totalRevenue"))),
            ("profit 12m", cr(i.get("netIncomeToCommon"))),
            ("P/E", num(i.get("trailingPE"))),
            ("price / book", num(i.get("priceToBook"))),
            ("debt / equity", num(i.get("debtToEquity"))),
            ("return on equity", num(roe * 100) + "%" if isinstance(roe, (int, float)) else None)]
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return []
    out = ["", "─" * W, "THE COMPANY ITSELF",
           "  The system did NOT use any of this to",
           "  choose the stock. It is here so a person",
           "  can judge for themselves. Third-party",
           "  figures, unverified by this system."]
    return out + [f"  {k:<17}{str(v)[:20]:>{W - 19}}" for k, v in rows]


LEDGER = os.path.join(_ROOT, "data", "paper", "paper_ledger.csv")


def _sectors():
    try:
        with open(os.path.join(_ROOT, "config", "sector_map.json")) as f:
            # The file wraps the map in a "sectors" key alongside its
            # provenance note; reading the top level silently yields nothing.
            d = json.load(f)
            d = d.get("sectors", d)
            return {k.upper(): v for k, v in d.items()}
    except (OSError, ValueError):
        return {}


SECTORS = _sectors()


def px(v):
    """A price with paise. rs() rounds to whole rupees, which is right for a
    portfolio total and wrong for an execution price — a position note that says
    "Rs 1,587" cannot be checked against a contract note."""
    body = f"{abs(float(v)):.2f}"
    whole, dec = body.split(".")
    return f"{'-' if float(v) < 0 else ''}Rs {_grp(whole)}.{dec}"


def _kv(label, value):
    return f"  {label:<19}{str(value):>{W - 21}}"


def _days_since(datestr):
    try:
        return (datetime.now() - datetime.strptime(datestr, "%Y-%m-%d")).days
    except (TypeError, ValueError):
        return None


def _plus_year(datestr):
    """The date a holding turns long-term. Sec 112A: >365 days, so the rate falls
    on day 366 — not on the anniversary, which is the usual off-by-one here."""
    try:
        d = datetime.strptime(datestr, "%Y-%m-%d")
        return (d.replace(year=d.year + 1)).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _vs_bench(L, entry_date, stock_pct):
    """This name against the index over the SAME window it has been held.

    "+16%" alone is not information — the market may have done +15%. The relevant
    figure is what the position added over simply owning the index for the same
    days, which is what anyone comparing this to a mutual fund actually wants.

    Read from nav_history, which is committed, so this works on a runner with no
    price cache. Returns nothing rather than guessing when the benchmark has no
    print on or before the entry date.
    """
    hist = [h for h in (L.get("nav_history") or [])
            if str(h.get("bench_inr", "")).strip() not in ("", "None")]
    if not hist or not entry_date:
        return []
    at_entry = [h for h in hist if h["date"] <= entry_date]
    base = at_entry[-1] if at_entry else hist[0]
    b0, b1 = float(base["bench_inr"]), float(hist[-1]["bench_inr"])
    if b0 <= 0:
        return []
    bench_pct = (b1 / b0 - 1) * 100
    lead = stock_pct - bench_pct
    out = [_kv("this stock", pct(stock_pct, 1)),
           _kv("Nifty 50", pct(bench_pct, 1)),
           _kv("ahead / behind", f"{lead:+.1f}pp"),
           "  Measured over the same days, so this is",
           "  what holding it added over simply owning",
           "  the index."]
    if not at_entry:
        out.append("  Index measured from the book's first mark.")
    return out


def bought_on(ticker):
    """First BUY date for a name, from the ledger.

    The book stores `entry_date` on a position, but 13 of the 22 open positions
    do not have one — it was added after those were opened, so /why printed a
    bare "?" for most of the book. The ledger has always had it: every fill is a
    row with a date, and the ledger is the record the book is a summary OF. So
    ask the record rather than backfilling the summary.
    """
    import csv
    try:
        with open(LEDGER) as fh:
            days = [r["date"] for r in csv.DictReader(fh)
                    if r.get("ticker") == ticker and r.get("action", "").upper() == "BUY"]
        return min(days) if days else None
    except (OSError, csv.Error):
        return None


SCORE_MEANING = os.path.join(_ROOT, "reports", "score_meaning.json")


def _what_the_score_has_meant(score):
    """What stocks at this score ACTUALLY did over the next six months.

    This is the honest form of "how much will it grow". The engine ranks; it does
    not forecast, and its information coefficient explains under 1% of any single
    name's return, so a per-stock growth number would be fabricated. A base rate
    with its spread is the same question answered with evidence.

    The spread is the point, and it is why this block leads with the median and
    then immediately undercuts it. Stocks in the top band returned +12.8% at the
    median — and 34% of them still lost money. A reader shown only the median
    would take away the exact opposite of what the data says.
    """
    try:
        d = json.load(open(SCORE_MEANING))
    except (OSError, ValueError):
        return []
    band = next((k for k in d.get("bands", {})
                 if len(k.split("-")) == 2
                 and float(k.split("-")[0]) < score <= float(k.split("-")[1])), None)
    b = (d.get("bands") or {}).get(band)
    if not b:
        return []
    return ["", "─" * W, f"WHAT A SCORE OF {band} HAS MEANT",
            f"  Over the SIX MONTHS after scoring this,",
            f"  across {b['n']:,} historical cases:", "",
            _kv("typical (middle)", pct(b["median"], 1)),
            _kv("a quarter did worse", pct(b["p25"], 1)),
            _kv("a quarter did better", pct(b["p75"], 1)),
            _kv("LOST MONEY", f"{b['pct_negative']:.0f}% of them"),
            "",
            f"  {d.get('evaluation_dates', '?')} measurement dates, about "
            f"{d.get('independent_periods', '?')}",
            "  genuinely independent six-month periods.",
            "",
            "  This is what stocks at this score DID.",
            "  It is not a prediction about this one, and",
            f"  {b['pct_negative']:.0f}% of them still lost money.",
            "  Gross of costs and tax; the book pays both."]


def _sizing(L, h):
    """Why this position is this size.

    "We hold LAURUSLABS" is half an answer; the other half is how much, and why
    that much. Equal weighting would be the null choice, so the honest way to show
    the sizing rule is the gap between what equal weight WOULD be and what this
    name actually got — which states the rule and its effect in one line without
    exposing a formula.
    """
    cfg = L.get("config") or {}
    rows = [r for r in (L.get("sleeves") or {}).get("rows") or [] if not r.get("passive")]
    if not rows:
        return []
    eq = rows[0]
    n = eq.get("n_holdings") or 0
    if not n:
        return []
    equal = float(eq["weight_pct"]) / n
    actual = float(h["weight"])
    cap = cfg.get("max_weight_per_name")
    out = ["", "HOW THE SIZE WAS SET",
           _kv("this position", f"{actual:.2f}%"),
           _kv("equal weight", f"{equal:.2f}%")]
    if cap:
        out.append(_kv("hard cap", f"{float(cap) * 100:.1f}%"))
    lean = "above" if actual > equal * 1.05 else ("below" if actual < equal * 0.95 else "at")
    out += ["",
            f"  Sized {lean} equal weight. Money is split by",
            "  inverse volatility: the steadier a stock has",
            "  been, the more it gets, because a calm name",
            "  and a wild one at the same weight are not",
            "  the same amount of risk.",
            "  The cap exists so no single name can decide",
            "  the year, however good its score."]
    return out


def _next_rebalance(L):
    try:
        p = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "paper_track.py"),
                            "rebalance", "--check"], capture_output=True, text=True,
                           cwd=_ROOT, timeout=120)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        return line.split("next ~")[-1].rstrip(")") if "next ~" in line else None
    except Exception:                                        # noqa: BLE001
        return None


def _exit_rule(L):
    """When it goes. Stated as a rule, because it IS one.

    This is the section a reader will look for when the position is down, and the
    honest answer is the least intuitive one in the whole system: nothing happens.
    Six separate approaches that cut exposure after a loss were tested and every
    one scored worse, so "we sell if it falls" is not a missing feature — it is a
    falsified one, and the note says which.
    """
    cfg = L.get("config") or {}
    nxt = _next_rebalance(L)
    out = ["", "WHEN IT WOULD BE SOLD"]
    if nxt:
        out.append(_kv("next review", nxt))
    if cfg.get("n_hold"):
        out.append(_kv("kept if in", f"top {cfg['n_hold']} by score"))
    out += ["",
            "  It is sold at the next scheduled review if",
            "  it is no longer in the top by score. That is",
            "  the only thing that removes it.",
            "",
            "  Not on a price fall, not on news, not on a",
            "  bad month. Six versions of this system that",
            "  cut exposure after a loss were tested and",
            "  all scored worse: Indian equity recovers in",
            "  a V, so selling after a fall sells the",
            "  rebound. The calendar is the rule."]
    return out


def h_why(ticker=""):
    """Why this stock is in the book — and what was never looked at.

    The second half is not a disclaimer bolted onto the first. The ranking uses
    five price and volume statistics and has never read a balance sheet, so a
    reader who assumes the picks were vetted on the business is holding a view of
    this book that is false. Saying so is the finding, not a caveat.
    """
    t = (ticker or "").strip().upper().replace(".NS", "")
    L = _export()
    held = {h["ticker"]: h for h in (L.get("holdings") or [])}
    if not t:
        sleeves = set((L.get("config") or {}).get("sleeve_targets") or {})
        names = sorted(set(held) - sleeves)
        if not names:
            return "No positions yet."
        return Menu(f"WHICH ONE?\n\nTap a stock for its position note.\n"
                    f"{len(names)} held. Or type  /why BHEL",
                    [(t2, f"why:{t2}") for t2 in names])

    try:
        book = json.load(open(BOOK))
    except (OSError, ValueError):
        book = {}
    sector = SECTORS.get(t, "")
    head = f"{t}" + (f"  ·  {sector}" if sector else "")
    out = [head, f"Position note  ·  {_asof(L)}" if t in held
           else "NOT HELD  ·  this stock is not in the book", "─" * W]

    if t in held:
        h = held[t]
        pos = (book.get("positions") or {}).get(t, {})
        when = pos.get("entry_date") or bought_on(t) or "?"
        nav, cap = float(L["nav"]), float(L["capital"])

        out += ["THE POSITION",
                _kv("entered", when),
                _kv("entry price", px(pos.get("entry_price", 0))),
                _kv("last price", px(h["price"])),
                _kv("shares", f"{int(h['qty']):,}"),
                _kv("market value", rs(h["value"])),
                _kv("unrealised P&L", f"{_amt(h['pnl'], True)}  {pct(h['pnl_pct'], 1)}"),
                _kv("weight in book", f"{float(h['weight']):.2f}%")]

        # Days held drives the tax rate, and it is the single most consequential
        # date on the position — 20% versus 12.5% is a 7.5pp swing on the gain.
        # Nobody should have to work it out from an entry date.
        days = _days_since(when)
        if days is not None:
            lt = _plus_year(when)
            out += [_kv("held", f"{days} days"),
                    _kv("tax if sold today", "20% short-term" if days <= 365
                        else "12.5% long-term")]
            if days <= 365 and lt:
                out += [_kv("  drops to 12.5% on", lt)]

        # Contribution in basis points: the standard way a desk reports what one
        # name did FOR THE BOOK, as opposed to what the stock did. A +16% stock at
        # a 3.5% weight and a +16% stock at a 0.5% weight are not the same event.
        bps = float(h["pnl"]) / cap * 10000
        out += ["", "CONTRIBUTION TO THE BOOK",
                _kv("this position", f"{bps:+.0f} bps"),
                _kv("whole book", f"{(nav - cap) / cap * 10000:+.0f} bps"),
                "  A basis point is one hundredth of a",
                "  percent. This is what the position did to",
                "  YOUR return, not what the stock did."]

        # Relative performance over the holding period. The benchmark series is in
        # the export, so this needs no price cache — which matters, because the
        # bot runs on a runner that has none.
        rel = _vs_bench(L, when, float(h["pnl_pct"]))
        if rel:
            out += ["", "SINCE IT WAS BOUGHT"] + rel

    try:
        sig = json.load(open(SIGNALS))
    except (OSError, ValueError):
        sig = None
    sc = ((sig or {}).get("scores") or {}).get(t)

    out += ["", "WHAT RANKED IT", "─" * W]
    if not sc:
        # Nothing invented. A rebuild of an old ranking does not return the old
        # ranking — corporate actions adjust price history retroactively and the
        # cross-sectional ranks move as names enter and leave the universe.
        out += ["  No scores recorded for this name.",
                "  The ranking is saved from each rebalance",
                "  onward. Re-deriving an old one gives",
                "  different numbers, so nothing is shown",
                "  rather than something made up."]
    else:
        fw = sig.get("factor_weights") or {}
        n = sig.get("n_eligible", "?")
        # A 0-100 score, not a raw z-score. The composite is a blended z-score:
        # "1.21" is meaningless without knowing the spread it came from, whereas
        # a position in the field is self-explaining and is what was asked for.
        # It is a RANK, and only ever a rank — see the caveat printed below it.
        top = (1 - (sc["rank"] - 1) / max(n - 1, 1)) * 100 if isinstance(n, int) else None
        if top is not None:
            out += [f"  {'SCORE':<17}{f'{top:.0f} / 100':>{W - 19}}"]
        out += [f"  {'ranked':<17}{f'{sc['rank']} of {n}':>{W - 19}}", ""]
        out += ["  what went into that score"]
        for f, p in sorted((sc.get("factors") or {}).items(),
                           key=lambda kv: -(fw.get(kv[0], 0))):
            if p is None:
                continue
            out.append(f"  {f:<10}{fw.get(f, 0) * 100:>3.0f}%  {_bar(p)}{p * 100:>4.0f}")
            out.append(f"    {FACTOR_PLAIN.get(f, '')}")
        out += ["", f"  Each is out of 100 against the other {n}.",
                "  The percentage is how much it counted.",
                "", f"  {sig.get('basis', 'basis not recorded')}"]
        if top is not None:
            out += _what_the_score_has_meant(top)

    if t in held:
        out += _sizing(L, held[t])
        out += _exit_rule(L)

    out += ["", "─" * W, "WHAT IT DID NOT LOOK AT",
            "  profits · debt · revenue · valuation",
            "  management · the business itself",
            "  It has never read a balance sheet.",
            "",
            "  Company financials were tested on real",
            "  12-year data across 98 companies and",
            "  made results WORSE (research log K15).",
            "  They help in flight-to-quality years and",
            "  hurt in junk rallies — a bet on the",
            "  regime, not on the company.",
            "",
            "HOW MUCH THIS ONE PICK MATTERS",
            "  Little. The ranking is right slightly",
            "  more often than a coin flip. The edge",
            "  comes from holding 20 of them for",
            "  months, not from this one being good."]
    out += _fundamentals(t)
    out += ["", "─" * W,
            "Model portfolio. Simulated execution at live",
            "market prices, net of costs and Indian tax.",
            "Not investment advice.", PAGE]
    return "\n".join(out)


def _does_it_work():
    """The ranking's own track record, printed next to the ranking.

    A list of "top 20" with no hit rate beside it reads as a promise. This is the
    measured answer: how often the picked 20 actually beat the field they were
    picked from, over six months, and by how much — including the periods they
    lost. Both halves go together or neither should be shown.
    """
    try:
        hh = (json.load(open(SCORE_MEANING)) or {}).get("head_to_head") or {}
    except (OSError, ValueError):
        return []
    if not hh.get("periods"):
        return []
    return ["", "HAS THIS RANKING WORKED?",
            _kv("beat the field", f"{hh['top20_beat_the_field']} of {hh['periods']} times"),
            _kv("typical edge", f"{hh['median_edge_pp']:+.1f}pp per 6 months"),
            _kv("worst period", f"{hh['worst_period_pp']:+.1f}pp"),
            "",
            f"  So it loses about 1 period in 3. The edge is",
            f"  real but small, and it is measured over only",
            f"  ~{hh['independent_periods']:.0f} independent six-month periods —",
            f"  t-stat {hh['t_stat_on_independent_periods']}, under this project's own",
            "  bar of 3.0. Treat it as evidence, not proof."]


def h_ranking(arg=""):
    """What the ranking says TODAY — which is not the same as what is held.

    Keeping those two apart is the whole point of the section headers here. The
    book changes only at the scheduled review; a name topping the list today has
    no claim on the portfolio until then, and showing this list without saying so
    would invite exactly the off-cadence trading Mandate §6 forbids.

    It is worth showing anyway, because the gap between the two IS the
    information: it is how much the book has drifted from what the rules would
    pick now, and therefore roughly how much January will change.
    """
    try:
        sig = json.load(open(SIGNALS))
    except (OSError, ValueError):
        return ("No ranking recorded yet. It is saved at each rebalance, and can "
                "be refreshed in between.")
    n = sig.get("n_eligible", 0)
    L = _export()
    sleeves = set((L.get("config") or {}).get("sleeve_targets") or {})
    held = {h["ticker"] for h in (L.get("holdings") or [])} - sleeves

    try:
        want = int(arg.strip())
    except (TypeError, ValueError):
        want = (L.get("config") or {}).get("n_hold") or 20
    want = max(5, min(want, 50))

    rows = sorted(sig["scores"].items(), key=lambda kv: kv[1]["rank"])[:want]
    out = [f"TOP {want} BY SCORE TODAY",
           f"  out of {n} the rules may choose from", "─" * W,
           f"  {'#':>3} {'stock':<12}{'score':>6}  held"]
    for t, v in rows:
        score = (1 - (v["rank"] - 1) / max(n - 1, 1)) * 100
        out.append(f"  {v['rank']:>3} {t[:12]:<12}{score:>6.0f}  "
                   + ("yes" if t in held else "—"))

    top = {t for t, _ in rows}
    keep, new, gone = len(top & held), len(top - held), len(held - top)
    out += ["─" * W,
            _kv("already held", f"{keep} of {want}"),
            _kv("would be new", str(new)),
            _kv("held, now outside", str(gone))]
    out += _does_it_work()
    out += ["",
            "THIS IS NOT THE BOOK.",
            "  It is what the rules say today. Nothing is",
            "  bought or sold until the scheduled review —",
            "  trading a list like this as it moves is the",
            "  thing that tested worst of everything tried.",
            "",
            f"  {sig.get('basis', '')}",
            "", PAGE]
    return "\n".join(out)


def h_sector(arg=""):
    """Where the money actually sits, by industry — as a chart.

    Concentration is the risk a holdings list hides. Twenty names reads as
    diversified right up until you notice eleven of them are capital goods, and
    nobody spots that by scanning tickers. This is the one view that answers "how
    exposed am I to one thing", so it is drawn rather than tabulated.

    The passive sleeves are shown as their own slices, not folded into equity:
    gold and the Nasdaq are the diversification, so hiding them inside a single
    "portfolio" bar would misstate exactly what the chart exists to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = _export()
    holdings = L.get("holdings") or []
    if not holdings:
        return "No positions yet."
    nav = float(L["nav"])
    sleeve_label = {}
    for r in (L.get("sleeves") or {}).get("rows") or []:
        if r.get("passive") and r.get("n_holdings") == 1:
            sleeve_label[round(float(r["value_inr"]), 2)] = r["label"]

    buckets, unlabelled = {}, []
    for h in holdings:
        lbl = sleeve_label.get(round(float(h["value"]), 2))
        if lbl is None:
            lbl = SECTORS.get(h["ticker"])
            if lbl is None:
                # Named, never silently bundled: an unlabelled name also escapes
                # the sector cap, so the chart must not imply it is accounted for.
                unlabelled.append(h["ticker"])
                lbl = "unclassified"
        buckets[lbl] = buckets.get(lbl, 0.0) + float(h["value"])
    rows = sorted(buckets.items(), key=lambda kv: -kv[1])

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(rows) + 1.4)), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = [r[0][:28] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colours = ["#1a7f37" if r[0] in sleeve_label.values() else
               ("#c2410c" if r[0] == "unclassified" else "#2563eb") for r in rows][::-1]
    ax.barh(labels, vals, color=colours, height=0.68)
    for i, v in enumerate(vals):
        ax.text(v + nav * 0.008, i, f"{v / nav * 100:.1f}%  {rs(v)}",
                va="center", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.34)
    ax.set_title(f"Where the money sits  ·  {rs(nav)}  ·  {_asof(L)}",
                 fontsize=12, loc="left", pad=12)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=9, length=0)
    fig.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)

    top = rows[0]
    cap = "  ".join(f"{k} {v / nav * 100:.0f}%" for k, v in rows[:4])
    txt = (f"Biggest single exposure: {top[0]} at {top[1] / nav * 100:.1f}% "
           f"({rs(top[1])}).\n{cap}")
    if unlabelled:
        txt += (f"\n{len(unlabelled)} holding(s) have no industry label "
                f"({', '.join(unlabelled[:4])}) — those also escape the sector cap.")
    txt += "\nGreen = whole-sleeve funds. Model portfolio."
    return Photo(buf.getvalue(), txt[:1024])


class Menu:
    """A message with tappable buttons under it.

    Typing a ticker is the only step in this bot where the owner can get it
    wrong — a typo, a wrong suffix, a name that was sold last rebalance — and the
    reply to a wrong ticker is useless rather than obviously wrong. Buttons remove
    the failure mode entirely: the only tickers offered are the ones actually
    held, so the question cannot be malformed.
    """

    def __init__(self, text, buttons, per_row=3):
        self.text, self.buttons, self.per_row = text, buttons, per_row

    def markup(self):
        rows, cur = [], []
        for label, data in self.buttons:
            cur.append({"text": label, "callback_data": data[:64]})
            if len(cur) == self.per_row:
                rows.append(cur); cur = []
        if cur:
            rows.append(cur)
        return json.dumps({"inline_keyboard": rows})


class Sweep:
    """A request to delete chat history. Carries no message ids of its own —
    `handle` supplies them, because only it knows which message asked."""

    def __init__(self, n):
        self.n = n


def h_clear(arg=""):
    """Delete recent messages in this chat.

    The Bot API has no "fetch history" call, so there is no list of what to
    delete. Message ids within a chat are sequential integers, so the sweep walks
    BACKWARDS from the id of the /clear message itself and asks Telegram to
    delete each one. Ids that are not deletable simply fail and are counted.

    That failure behaviour is the safety mechanism, not a workaround. Telegram
    refuses to let a non-admin bot delete a message it did not send — so in a
    group with other people in it, a plain bot removes only its own clutter and
    cannot touch anyone else's messages, whatever number is passed. If the bot IS
    made an admin with delete rights, the same sweep will remove EVERYTHING in
    range, including other members' messages. The reply says which of those two
    just happened rather than leaving it to be discovered.

    Telegram also refuses to delete anything older than 48 hours.
    """
    try:
        n = int((arg or "").strip())
    except (TypeError, ValueError):
        n = 100
    return Sweep(max(1, min(n, 400)))


def h_help(arg=""):
    out = ["WHAT YOU CAN ASK", "─" * W]
    out += [f"  /{n:<10} {d}" for n, d, _ in COMMANDS]
    out += ["",
            "Nothing you type can buy, sell, or change the",
            "money record — that runs on its own schedule",
            "and reports back. /clear is the one command",
            "that changes anything, and only this chat.",
            "",
            "If a reply takes a few minutes, the bot was",
            "asleep. Nothing is lost; it answers on waking.",
            "", PAGE]
    return "\n".join(out)


COMMANDS = [
    ("update",   "Where your money is right now",        h_update),
    ("holdings", "Every position, best to worst",        h_holdings),
    ("chart",    "A picture: you vs the index",          h_chart),
    ("why",      "Why a stock is held: /why BHEL",       h_why),
    ("ranking",  "What the rules rank highest today",    h_ranking),
    ("sector",   "Chart: which industries hold my money", h_sector),
    ("clear",    "Delete recent messages: /clear 100",   h_clear),
    ("next",     "When it next re-picks the stocks",     h_next),
    ("health",   "Run the integrity checks now",         h_health),
    ("help",     "This list",                            h_help),
]
HANDLERS = {n: f for n, _, f in COMMANDS}
ALIASES = {"status": "update", "pnl": "update", "money": "update",
           "start": "help", "positions": "holdings", "stocks": "holdings", "graph": "chart",
           "rebalance": "next", "explain": "why", "top": "ranking", "scores": "ranking", "sectors": "sector", "division": "sector", "industry": "sector", "allocation": "sector", "clean": "clear"}


# ── dispatch ─────────────────────────────────────────────────────────────
def answer(text):
    """Command text -> reply body, or None to stay silent.

    Silence is the correct response to anything that is not a command. This bot
    sits in a group with a human in it; a bot that replies to conversation is a
    bot that gets muted, and a muted bot is the same as no bot on the day it has
    something urgent to say.
    """
    word = (text or "").strip().split(maxsplit=1)[0] if (text or "").strip() else ""
    if not word.startswith("/"):
        return None
    cmd = word[1:].split("@", 1)[0].lower()      # /update@MARK5K_BOT in groups
    cmd = ALIASES.get(cmd, cmd)
    rest = (text or "").strip().split(maxsplit=1)
    arg = rest[1] if len(rest) > 1 else ""
    fn = HANDLERS.get(cmd)
    if fn is None:
        return f"No such command: /{cmd}\n\n" + h_help()
    try:
        return fn(arg)
    except Exception as e:                                   # noqa: BLE001
        # An exception here must reach the owner as words, not vanish into a CI
        # log they are not reading. Saying "it broke and here is why" is the
        # entire point of this system talking at all.
        return f"That command failed:\n  {scrub(e)}\n\nThe money record is untouched."


def send_photo(chat, png, caption):
    """sendPhoto needs multipart/form-data, which urllib will not build for us.
    Hand-rolled rather than pulling in `requests` for one call."""
    tok = os.environ["TELEGRAM_BOT_TOKEN"]
    bnd = "----mark6" + os.urandom(12).hex()
    body = b""
    for k, v in (("chat_id", str(chat)), ("caption", caption[:1024])):
        body += (f"--{bnd}\r\nContent-Disposition: form-data; "
                 f'name="{k}"\r\n\r\n{v}\r\n').encode()
    body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"photo\"; "
             f'filename="nav.png"\r\nContent-Type: image/png\r\n\r\n').encode()
    body += png + b"\r\n" + f"--{bnd}--\r\n".encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{tok}/sendPhoto", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(scrub(e)) from None


def reply(chat, body, dry=False):
    if isinstance(body, Menu):
        if dry:
            print(f"--- would offer {len(body.buttons)} buttons to {chat} ---\n{body.text}")
            return
        _api("sendMessage", chat_id=chat, text=body.text,
             reply_markup=body.markup())
        return
    if isinstance(body, Photo):
        if dry:
            print(f"--- would send a {len(body.png)}-byte chart to {chat} ---")
            print(f"    caption: {body.caption}")
            return
        send_photo(chat, body.png, body.caption)
        return
    if dry:
        print(f"--- would reply to {chat} ---\n{body}\n")
        return
    for i in range(0, len(body), CHUNK):
        send("MARK6", body[i:i + CHUNK], chat=chat)


def _sweep(chat, from_id, n, dry=False):
    """Delete `n` message ids backwards from `from_id`, then report.

    Deliberately reports rather than staying silent: a command that removes
    things must say how many, or the owner cannot tell "nothing to delete" from
    "the bot has no permission". The confirmation is itself sent AFTER the sweep
    so it survives it.
    """
    if not from_id:
        return False
    if dry:
        print(f"--- would sweep {n} ids back from {from_id} in {chat} ---")
        return True

    # Whether other people's messages are at risk depends entirely on this.
    admin = False
    try:
        me = _api("getMe").get("result") or {}
        st = (_api("getChatMember", chat_id=chat, user_id=me.get("id"))
              .get("result") or {})
        admin = bool(st.get("can_delete_messages"))
    except RuntimeError:
        pass

    gone = 0
    for mid in range(int(from_id), max(0, int(from_id) - n), -1):
        try:
            if (_api("deleteMessage", chat_id=chat, message_id=mid) or {}).get("ok"):
                gone += 1
        except RuntimeError:
            pass                 # not ours, too old, or already gone — all fine
        time.sleep(0.04)         # Telegram throttles bulk deletes

    note = ("This bot is an ADMIN with delete rights, so that included messages "
            "from everyone in this chat." if admin else
            "Only this bot's own messages could be removed — Telegram protects "
            "everyone else's, and anything older than 48 hours.")
    try:
        send("MARK6", f"Cleared {gone} message(s).\n{note}", chat=chat)
    except Exception:                                        # noqa: BLE001
        pass
    print(f"  swept {gone} of {n} ids in {chat} (admin={admin})")
    return True


def handle_tap(cq, dry=False):
    """A button press. Same authorisation as a typed command — a callback carries
    a chat id like anything else, and a bot that trusts taps but not text has
    simply moved the hole."""
    chat = str(((cq.get("message") or {}).get("chat") or {}).get("id", ""))
    data = cq.get("data") or ""
    if chat not in allowed_chats():
        print(f"  ignored a tap from unauthorised chat {chat}")
        return False
    if not dry:
        # Always answer, even on failure: an unanswered callback leaves a spinner
        # turning on the button forever, which reads as a hung system.
        try:
            _api("answerCallbackQuery", callback_query_id=cq.get("id"))
        except RuntimeError:
            pass
    if not data.startswith("why:"):
        return False
    ticker = data.split(":", 1)[1]
    print(f"  {chat} tapped {ticker}")
    try:
        body = h_why(ticker)
    except Exception as e:                                   # noqa: BLE001
        body = f"That failed:\n  {scrub(e)}\n\nThe money record is untouched."
    reply(chat, body, dry)
    return True


def handle(msg, dry=False):
    chat = str((msg.get("chat") or {}).get("id", ""))
    text = msg.get("text") or ""
    if chat not in allowed_chats():
        # Logged, never answered: the id is what the owner needs to add a new
        # chat, and a reply would confirm to a stranger that the bot is live.
        print(f"  ignored a message from unauthorised chat {chat}")
        return False
    body = answer(text)
    if body is None:
        return False
    if isinstance(body, Sweep):
        return _sweep(chat, msg.get("message_id"), body.n, dry)
    size = f"a {len(body.png)}-byte chart" if isinstance(body, Photo) else f"{len(body)} chars"
    print(f"  {chat} said {text.split()[0]!r} -> replying {size}")
    try:
        _api("sendChatAction", chat_id=chat, action="typing")
    except RuntimeError:
        pass                                     # cosmetic only, never fatal
    reply(chat, body, dry)
    return True


# ── the polling window ───────────────────────────────────────────────────
def serve(once=False, dry=False):
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("  no TELEGRAM_BOT_TOKEN — nothing to poll")
        return
    if not allowed_chats():
        # Refusing here rather than defaulting to "answer everyone" — the failure
        # mode of an open bot is that the book's positions leak silently.
        print("  no TELEGRAM_CHAT_ID / TELEGRAM_ADMIN_CHATS — refusing to answer anyone")
        return

    hard = time.monotonic() + HARD_CAP
    quiet = time.monotonic() + IDLE_EXIT
    offset, served = None, 0

    while True:
        left = min(quiet, hard) - time.monotonic()
        if left <= 0:
            break
        try:
            d = _api("getUpdates", offset=offset,
                     timeout=0 if once else int(min(POLL, max(1, left))),
                     allowed_updates='["message","callback_query"]')
        except RuntimeError as e:
            # 409 means another run of this workflow is already polling. Not an
            # error worth failing the job over — the other one is doing the work.
            print(f"  poll stopped: {e}")
            return
        ups = d.get("result") or []
        if ups:
            offset = max(u["update_id"] for u in ups) + 1
            # Confirm BEFORE acting. If this process dies mid-answer the command
            # is not replayed on the next run; at-most-once is the right default
            # for anything triggered by a message, and a lost /update costs one
            # retype. (Updates newer than `offset` stay unconfirmed and return.)
            try:
                _api("getUpdates", offset=offset, timeout=0, limit=1)
            except RuntimeError:
                pass
            for u in ups:
                if u.get("callback_query"):
                    ok = handle_tap(u["callback_query"], dry)
                else:
                    ok = handle(u.get("message") or {}, dry)
                if ok:
                    served += 1
            quiet = time.monotonic() + ACTIVE_EXIT
        if once:
            break

    print(f"  window closed — answered {served} command(s)")


def register():
    """Publish the command list so Telegram offers autocomplete.

    Done on every run rather than as a one-off setup step: the menu is then
    derived from COMMANDS and cannot drift from what the code actually answers.
    """
    try:
        _api("setMyCommands",
             commands=json.dumps([{"command": n, "description": d}
                                  for n, d, _ in COMMANDS]))
    except RuntimeError as e:
        print(f"  could not publish the command menu: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="hold a polling window open (what the workflow runs)")
    ap.add_argument("--dry", action="store_true", help="print replies, send nothing")
    ap.add_argument("--say", metavar="CMD",
                    help="render one command locally, e.g. --say /holdings")
    a = ap.parse_args()

    if a.say:
        body = answer(a.say if a.say.startswith("/") else "/" + a.say)
        if isinstance(body, Menu):
            print(body.text + "\n")
            for r in json.loads(body.markup())["inline_keyboard"]:
                print("  [ " + " ] [ ".join(b["text"] for b in r) + " ]")
            return
        if isinstance(body, Photo):
            import tempfile
            # Not the repo root — a preview is scratch, not an artifact.
            out = os.path.join(tempfile.gettempdir(), "mark6_chart.png")
            open(out, "wb").write(body.png)
            print(f"wrote {out} ({len(body.png)} bytes)\n{body.caption}")
        else:
            print(body if body is not None else "(not a command — it would stay silent)")
        return
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        register()
    serve(once=not a.serve, dry=a.dry)


if __name__ == "__main__":
    main()
