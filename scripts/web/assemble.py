"""Assemble the explorer: template.html + data (+ analysis_results.json) -> NYC_Environment_Cancer_Explorer.html

    python3 assemble.py [path/to/analysis_results.json]

Data source: ./data.json (written by build_data.py) if it exists and the page is written next to it; otherwise the data already
embedded in outputs/NYC_Environment_Cancer_Explorer.html is reused and that file is rewritten in place (so the page can be rebuilt
from this repository alone, for example after re-running the analysis).
The analysis results default to ../../outputs/analysis/analysis_results.json (written by scripts/analysis/run_analysis.py); if that
file is missing, section 5 of the page shows a "results not found" message.
"""
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
page = os.path.join(here, "..", "..", "outputs", "NYC_Environment_Cancer_Explorer.html")
ap = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "outputs", "analysis", "analysis_results.json")
if os.path.exists("data.json"):
    data, out = json.load(open("data.json")), "NYC_Environment_Cancer_Explorer.html"
else:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', open(page, encoding="utf-8").read(), re.S)
    data, out = json.loads(m.group(1)), page
data.pop("analysis", None)
if os.path.exists(ap):
    data["analysis"] = json.load(open(ap))
    print("embedding", os.path.abspath(ap))
else:
    print("no analysis results at", os.path.abspath(ap), "- section 5 will show a notice")
d = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
t = open(os.path.join(here, "template.html"), encoding="utf-8").read()
assert t.count("/*DATA*/") == 1
open(out, "w", encoding="utf-8").write(t.replace("/*DATA*/", d))
print("wrote", os.path.abspath(out), len(d) // 1024, "KB of embedded data")
