"""Fill {{fig-x-y}} placeholders in the post template from figs.json."""
import json, re, sys
tpl, figs, out = sys.argv[1:4]
md = open(tpl).read()
by_id = {re.match(r'<figure id="([^"]+)"', h).group(1): h for h in json.load(open(figs)).values()}
used = set(re.findall(r"\{\{(fig-\d-\d)\}\}", md))
assert used == set(by_id), (sorted(used ^ set(by_id)))
md = re.sub(r"\{\{(fig-\d-\d)\}\}", lambda m: by_id[m.group(1)], md)
open(out, "w").write(md)
print("assembled", len(used), "figures")
