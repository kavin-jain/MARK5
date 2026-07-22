"""Inject the real mark6.json into the dashboard template -> a standalone HTML file.

  python3 docs/build_dashboard.py          # -> docs/index.html
The Next.js version reads the same JSON at runtime; this build exists so the page
also works as a single file on any host, with no build step and no dependencies.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data", "mark6.json")))

# The trade ledger is the bulk of the payload. Keep every row (it is the evidence)
# but drop the redundant/empty columns the table never renders.
for t in data["research"]["trades"]:
    t.pop("term", None)

tpl = open(os.path.join(HERE, "template.html")).read()
out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
path = os.path.join(HERE, "index.html")
open(path, "w").write(out)
print(f"  wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")
