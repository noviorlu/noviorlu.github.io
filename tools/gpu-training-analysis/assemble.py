"""Fill {{fig-x-y}} placeholders from figs.json and {{code:file:name}} from source files."""
import json, os, re, sys
tpl, figs, out = sys.argv[1:4]
md = open(tpl).read()
by_id = {re.match(r'<figure id="([^"]+)"', h).group(1): h for h in json.load(open(figs)).values()}
used = set(re.findall(r"\{\{(fig-\d-\d)\}\}", md))
assert used == set(by_id), (sorted(used ^ set(by_id)))
md = re.sub(r"\{\{(fig-\d-\d)\}\}", lambda m: by_id[m.group(1)], md)
# {{code:file.py:name}} -> the lines between "# post:name" and "# post:end" in file.py (next to this script)
def code(m):
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), m.group(1))).read()
    return re.search(rf"^# post:{m.group(2)}\n(.*?)^# post:end$", src, re.S | re.M).group(1).rstrip()
md = re.sub(r"\{\{code:([\w.]+):(\w+)\}\}", code, md)
open(out, "w").write(md)
print("assembled", len(used), "figures")
