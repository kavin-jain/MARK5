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

EVERY COMMAND HERE IS READ-ONLY, AND THAT IS A DESIGN DECISION
---------------------------------------------------------------
Mandate §6: the book is an append-only integrity record, never rebalanced
off-cadence, never stamped with a mid-session price. A chat message is the
single worst authorisation mechanism for a write to that record — it is one
fat-finger from a fill nobody scheduled, in a file whose entire value is that
nothing unscheduled is in it. So the workflow that runs this grants
`contents: read` and there is nothing here that could write even if it tried.

Reading is a different question, and reading is genuinely useful. That is the
whole of what this does.

  python3 scripts/bot.py            # one drain-and-reply pass, then exit
  python3 scripts/bot.py --serve    # the polling window the workflow runs
  python3 scripts/bot.py --dry      # print what it would answer, send nothing
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from notify import (EXPORT, PAGE, _grp, build, health, pct,     # noqa: E402
                    scrub, send)

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


def h_update():
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


def h_holdings():
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


def h_next():
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


def h_health():
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


def h_help():
    out = ["WHAT YOU CAN ASK", "─" * W]
    out += [f"  /{n:<10} {d}" for n, d, _ in COMMANDS]
    out += ["",
            "Everything here only reads. Nothing you type",
            "can buy, sell, or change the record — that runs",
            "on its own schedule and reports back.",
            "",
            "If a reply takes a few minutes, the bot was",
            "asleep. Nothing is lost; it answers on waking.",
            "", PAGE]
    return "\n".join(out)


COMMANDS = [
    ("update",   "Where your money is right now",        h_update),
    ("holdings", "Every position, best to worst",        h_holdings),
    ("next",     "When it next re-picks the stocks",     h_next),
    ("health",   "Run the integrity checks now",         h_health),
    ("help",     "This list",                            h_help),
]
HANDLERS = {n: f for n, _, f in COMMANDS}
ALIASES = {"status": "update", "pnl": "update", "money": "update",
           "start": "help", "positions": "holdings", "stocks": "holdings",
           "rebalance": "next"}


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
    fn = HANDLERS.get(cmd)
    if fn is None:
        return f"No such command: /{cmd}\n\n" + h_help()
    try:
        return fn()
    except Exception as e:                                   # noqa: BLE001
        # An exception here must reach the owner as words, not vanish into a CI
        # log they are not reading. Saying "it broke and here is why" is the
        # entire point of this system talking at all.
        return f"That command failed:\n  {scrub(e)}\n\nThe money record is untouched."


def reply(chat, body, dry=False):
    if dry:
        print(f"--- would reply to {chat} ---\n{body}\n")
        return
    for i in range(0, len(body), CHUNK):
        send("MARK6", body[i:i + CHUNK], chat=chat)


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
    print(f"  {chat} said {text.split()[0]!r} -> replying {len(body)} chars")
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
                     allowed_updates='["message"]')
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
                if handle(u.get("message") or {}, dry):
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
        print(body if body is not None else "(not a command — the bot would stay silent)")
        return
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        register()
    serve(once=not a.serve, dry=a.dry)


if __name__ == "__main__":
    main()
