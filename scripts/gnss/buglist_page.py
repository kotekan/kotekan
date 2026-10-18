#!/usr/bin/env python3
"""Render docs/CHORD_BUGLIST.md (+ CHORD_BUGLIST_CLOSED.md) as one HTML page for the artifact.

    python3 scripts/gnss/buglist_page.py --out /tmp/buglist.html

The page is a view of the two markdown files and nothing else: regenerate it after every
reconcile and republish to the same artifact URL (the page says which commit it was built
from). The converter covers what the two files use -- headings, paragraphs, bold/italic,
inline code, links, bullet and numbered lists, pipe tables, rules, fenced code -- and escapes
everything else, so a construct it does not know renders as text rather than as markup.
"""
import argparse
import datetime
import html
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_REF = re.compile(r"(?<![\w#/])#(\d{1,4})\b")


def inline(text, ids):
    """Escape, then mark up. Code spans are lifted out first so nothing inside them is touched.
    `ids` maps a bug number (str) to the anchor its #NNN references land on."""
    spans = []

    def lift(m):
        spans.append(m.group(1))
        return "\x00%d\x00" % (len(spans) - 1)

    t = _CODE.sub(lift, text)
    t = html.escape(t, quote=False)
    t = _LINK.sub(lambda m: '<a href="%s">%s</a>' % (html.escape(m.group(2), True), m.group(1)), t)
    t = _BOLD.sub(r"<strong>\1</strong>", t)
    t = _ITAL.sub(r"<em>\1</em>", t)
    t = _REF.sub(lambda m: ('<a class="ref" href="#%s">#%s</a>' % (ids[m.group(1)], m.group(1)))
                 if m.group(1) in ids else m.group(0), t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: "<code>%s</code>" % html.escape(spans[int(m.group(1))], False), t)
    return t


# "### #201 — title", "### #142 residual — title", "### #93 / GAP 3 — title", or a plain "### title".
_ENTRY = re.compile("^### (?:#(?P<id>[0-9]+)(?P<suf>[^\u2014]*))?\\s*(?:\u2014\\s*)?(?P<title>.*)$")
_H2 = re.compile(r"^## (.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}")
_BULLET = re.compile(r"^(\s*)[-*] (.*)$")
_NUMBER = re.compile(r"^(\s*)(\d+)\. (.*)$")


def blocks(lines, ids):
    """Markdown body lines -> HTML. Paragraph lines are joined (the files hard-wrap at ~118)."""
    out, i, n = [], 0, len(lines)
    para = []

    def flush():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(s.strip() for s in para), ids))
            para.clear()

    while i < n:
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("```"):
            flush()
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre>%s</pre>" % html.escape("\n".join(lines[i + 1:j]), False))
            i = j + 1; continue
        if re.match(r"^-{3,}$", s):
            flush(); out.append("<hr>"); i += 1; continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            out.append(table(rows, ids)); continue
        m = _BULLET.match(ln) or _NUMBER.match(ln)
        if m:
            flush()
            tag = "ol" if _NUMBER.match(ln) else "ul"
            items = []
            while i < n:
                m = _BULLET.match(lines[i]) if tag == "ul" else _NUMBER.match(lines[i])
                if not m:
                    break
                item = [m.group(m.lastindex)]
                i += 1
                # continuation lines: indented, not a new item, not blank
                while i < n and lines[i].strip() and not (_BULLET.match(lines[i]) or _NUMBER.match(lines[i])) \
                        and (lines[i].startswith(" ") or lines[i].startswith("\t")):
                    item.append(lines[i].strip()); i += 1
                items.append("<li>%s</li>" % inline(" ".join(item), ids))
            out.append("<%s>%s</%s>" % (tag, "".join(items), tag)); continue
        para.append(ln); i += 1
    flush()
    return "\n".join(out)


def table(rows, ids):
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    head, body = cells[0], []
    for r in cells[1:]:
        if all(_TABLE_SEP.match(c + "-") or re.match(r"^:?-+:?$", c) for c in r):
            continue
        body.append(r)
    h = "".join("<th>%s</th>" % inline(c, ids) for c in head)
    b = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(c, ids) for c in r) for r in body)
    return '<div class="tbl"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (h, b)


def parse(path):
    """-> (title, preamble_lines, [section]) ; section = {title, intro_lines, entries:[{id,title,lines}]}"""
    lines = open(path, encoding="utf-8").read().split("\n")
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("# ") else os.path.basename(path)
    pre, sections, cur, ent = [], [], None, None
    for ln in lines[1:]:
        m2 = _H2.match(ln)
        if m2:
            cur = {"title": m2.group(1).strip(), "intro": [], "entries": []}
            sections.append(cur); ent = None; continue
        me = _ENTRY.match(ln) if ln.startswith("### ") else None
        if me and cur is not None:
            num, suf, title = me.group("id"), (me.group("suf") or "").strip(), me.group("title").strip()
            if num:
                label = "#%s%s" % (num, (" " + suf) if suf else "")
                anchor = "b%s%s" % (num, ("-" + slug(suf)) if suf else "")
            else:
                label, anchor, title = "", "e-" + slug(title), title or "untitled"
            ent = {"id": num, "suffix": suf, "label": label, "anchor": anchor, "title": title, "lines": []}
            cur["entries"].append(ent); continue
        if ent is not None:
            ent["lines"].append(ln)
        elif cur is not None:
            cur["intro"].append(ln)
        else:
            pre.append(ln)
    return title, pre, sections


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


_TAIL = re.compile(r"\s*\(([^()]*)\)\s*$")
_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
_FIXW = re.compile(r"\*\*Fix \[([^\]]+)\]")
_LABEL = re.compile(r"^\*\*[^*]+\*\*\s*")


def split_title(title):
    """'...title (2026-10-18, CONFIRMED ...)' -> (title, date, status). The trailing
    parenthetical carries the date and, after it, whatever status the author put there."""
    m = _TAIL.search(title)
    if not m:
        return title, "", ""
    inner = m.group(1)
    d = _DATE.search(inner)
    date = d.group(1) if d else ""
    status = inner
    if d:
        status = (inner[:d.start()] + inner[d.end():]).strip(" ,;")
    return title[:m.start()].rstrip(), date, status


def gist(lines, limit=210):
    """The first sentence of the body with the lead label (**What [live]:**) dropped, plain text."""
    text = " ".join(l.strip() for l in lines if l.strip() and not l.strip().startswith("|")
                    and not l.strip().startswith("-") and not l.strip().startswith("```"))
    text = _LABEL.sub("", text)
    text = re.sub(r"\*\*|`", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"[.;]\s+(?=[A-Z(*])", text[:limit + 80])
    cut = text[:m.start() + 1] if m and m.start() < limit + 60 else text[:limit]
    if len(cut) < len(text) and not cut.endswith("."):
        cut = cut.rstrip(" ,;:") + "\u2026"
    return cut


def render_sections(sections, ids, kind):
    out = []
    for sec in sections:
        sid = "%s-%s" % (kind, slug(sec["title"]))
        cnt = len(sec["entries"])
        out.append('<section class="sec" id="%s"><h2>%s%s</h2>' % (
            sid, inline(sec["title"], ids),
            (' <span class="count" data-sec="%s">%d</span>' % (sid, cnt)) if cnt else ""))
        if any(st.strip() for st in sec["intro"]):
            intro = blocks(sec["intro"], ids)
            if cnt:
                out.append('<details class="intro"><summary>About this lane</summary>%s</details>' % intro)
            else:
                out.append('<div class="intro">%s</div>' % intro)
        for e in sec["entries"]:
            body = blocks(e["lines"], ids)
            text = html.escape(re.sub(r"\s+", " ", "%s %s %s" % (e["label"], e["title"], " ".join(e["lines"]))).lower(), True)
            chip = ('<a class="chip" href="#%s">%s</a>' % (e["anchor"], html.escape(e["label"]))) if e["label"] else ""
            title, date, status = split_title(e["title"])
            fw = _FIXW.search(" ".join(e["lines"]))
            tags = []
            if date:
                tags.append('<span class="tag date">%s</span>' % date)
            if status:
                tags.append('<span class="tag st">%s</span>' % html.escape(status))
            if fw:
                tags.append('<span class="tag fix">fix: %s</span>' % html.escape(fw.group(1)))
            g = gist(e["lines"])
            out.append('<details class="entry %s" id="%s" data-sec="%s" data-text="%s">'
                       '<summary><span class="row">%s<span class="ttl">%s</span><span class="tags">%s</span></span>'
                       '<span class="gist">%s</span></summary><div class="body">%s</div></details>'
                       % (kind, e["anchor"], sid, text, chip, inline(title, ids), "".join(tags),
                          html.escape(g, False), body))
        out.append("</section>")
    return "\n".join(out)


def git_stamp():
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "--", "docs/CHORD_BUGLIST.md",
                                "docs/CHORD_BUGLIST_CLOSED.md"]).returncode != 0
        return sha + ("+edits" if dirty else "")
    except Exception:
        return "unknown"


CSS = """
:root{--bg:#f6f5f1;--ink:#1f2328;--ink2:#5b6168;--line:#dcd8ce;--chip:#ebe8e0;--acc:#b8741a;--acc-ink:#fff;
 --closed:#3a7d44;--code:#efece5;--link:#1f5fa8;--mark:#fff3d6;--nav:#fbfaf7}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#16181c;--ink:#e6e3dc;
 --ink2:#a4a8ae;--line:#2d3139;--chip:#262a31;--acc:#e0a040;--acc-ink:#1a1206;--closed:#6fbf7a;--code:#1f2329;
 --link:#7fb3ec;--mark:#3a2e12;--nav:#1b1e23}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#16181c;--ink:#e6e3dc;--ink2:#a4a8ae;--line:#2d3139;--chip:#262a31;
 --acc:#e0a040;--acc-ink:#1a1206;--closed:#6fbf7a;--code:#1f2329;--link:#7fb3ec;--mark:#3a2e12;--nav:#1b1e23}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
a{color:var(--link)} a.ref{text-decoration:none;border-bottom:1px dotted var(--link)}
code,pre,.chip,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code);padding:0 .3em;border-radius:3px;font-size:.92em}
pre{background:var(--code);padding:.7em .9em;border-radius:4px;overflow-x:auto;font-size:.86em;line-height:1.4}
.top{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--nav);border-bottom:1px solid var(--line);
 padding:.6rem 16px;display:flex;flex-wrap:wrap;gap:.5rem 1.2rem;align-items:baseline}
.top h1{font-size:1.15rem;margin:0;font-weight:600;letter-spacing:.01em}
.top .stamp{color:var(--ink2);font-size:.82rem}
.top input{font:inherit;font-size:.9rem;padding:.3em .6em;border:1px solid var(--line);border-radius:4px;background:var(--bg);
 color:var(--ink);min-width:14rem;flex:1 1 14rem;max-width:28rem}
.top input:focus{outline:2px solid var(--acc);outline-offset:1px}
.top .hits{font-size:.82rem;color:var(--ink2);font-variant-numeric:tabular-nums}
.wrap{display:grid;grid-template-columns:220px minmax(0,1fr);gap:0 2rem;padding-block:1rem 3rem;padding-inline:16px;max-width:1180px;margin:0 auto}
nav.side{position:sticky;top:calc(3.4rem + env(safe-area-inset-top,0px));align-self:start;font-size:.85rem;max-height:calc(100vh - 4.5rem);overflow:auto}
nav.side a{display:block;color:var(--ink);text-decoration:none;padding:.22em 0;border-left:2px solid transparent;padding-left:.6em}
nav.side a:hover{border-left-color:var(--acc)}
nav.side .n{color:var(--ink2);font-variant-numeric:tabular-nums}
nav.side .grp{margin:.8em 0 .2em;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}
main{max-width:78ch;min-width:0}
h2{font-size:1.25rem;margin:2.2rem 0 .6rem;font-weight:600;text-wrap:balance;border-bottom:1px solid var(--line);padding-bottom:.3rem}
h2 .count{font-size:.8rem;color:var(--ink2);font-weight:500;margin-left:.4em;font-variant-numeric:tabular-nums}
.intro{color:var(--ink);font-size:.95rem}
.entry{border-top:1px solid var(--line)}
.entry>summary{list-style:none;cursor:pointer;padding:.45rem 0 .45rem 1.1rem;position:relative}
.entry>summary::-webkit-details-marker{display:none}
.entry>summary::before{content:"";position:absolute;left:.1rem;top:.95rem;border:4px solid transparent;border-left:5px solid var(--ink2);transition:transform .12s}
.entry[open]>summary::before{transform:rotate(90deg)}
.entry>summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:3px}
.row{display:flex;gap:.5rem .6rem;align-items:baseline;flex-wrap:wrap}
.ttl{font-weight:600;font-size:.97rem}
.tags{display:inline-flex;gap:.35rem;flex-wrap:wrap}
.tag{font-size:.72rem;color:var(--ink2);background:var(--chip);padding:.02em .45em;border-radius:3px;white-space:nowrap}
.tag.date{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.tag.fix{color:var(--acc)}
.gist{display:block;color:var(--ink2);font-size:.86rem;line-height:1.4;margin-top:.15rem}
.entry[open] .gist{display:none}
.entry .body{padding:0 0 .8rem 1.1rem}
details.intro{font-size:.9rem;margin:.3rem 0 .6rem} details.intro>summary{cursor:pointer;color:var(--ink2)}
details.about{margin:.6rem 0 1rem;font-size:.95rem} details.about>summary{cursor:pointer;font-weight:600}
.top button{font:inherit;font-size:.8rem;padding:.25em .6em;border:1px solid var(--line);border-radius:4px;background:var(--chip);color:var(--ink);cursor:pointer}
.top button:hover{border-color:var(--acc)}
.chip{display:inline-block;background:var(--acc);color:var(--acc-ink);text-decoration:none;font-size:.78rem;padding:.05em .5em;
 border-radius:3px;font-weight:600;letter-spacing:.02em}
.entry.closed .chip{background:var(--closed);color:#fff}
.entry .body p{margin:.5rem 0}
.entry.hide,.sec.hide{display:none}
.tbl{overflow-x:auto;margin:.6rem 0}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:.25em .55em;text-align:left;vertical-align:top}
th{background:var(--chip);font-weight:600}
ul,ol{padding-left:1.4em} li{margin:.2em 0}
hr{border:0;border-top:1px solid var(--line);margin:1.4rem 0}
details.closedbox{margin-top:3rem;border-top:2px solid var(--closed);padding-top:.5rem}
details.closedbox summary{cursor:pointer;font-weight:600;font-size:1.1rem;padding:.4rem 0}
details.closedbox summary .n{color:var(--ink2);font-weight:500;font-size:.85rem}
.foot{color:var(--ink2);font-size:.8rem;margin-top:3rem;border-top:1px solid var(--line);padding-top:.6rem}
:target{scroll-margin-top:4.6rem} .entry:target{background:var(--mark);border-radius:4px}
@media (max-width:760px){.wrap{grid-template-columns:1fr}nav.side{position:static;max-height:none;display:flex;flex-wrap:wrap;gap:.2em .8em}
 nav.side .grp{width:100%} nav.side a{border-left:0;padding-left:0}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
"""

JS = """
(function(){
  var q=document.getElementById('flt'), hits=document.getElementById('hits');
  var entries=Array.prototype.slice.call(document.querySelectorAll('article.entry'));
  var secs=Array.prototype.slice.call(document.querySelectorAll('section.sec'));
  function apply(){
    var s=(q.value||'').trim().toLowerCase(), terms=s?s.split(/\\s+/):[];
    var shown=0, per={};
    entries.forEach(function(e){
      var t=e.getAttribute('data-text'), ok=terms.every(function(w){return t.indexOf(w)>=0;});
      e.classList.toggle('hide',!ok); if(ok){shown++; per[e.getAttribute('data-sec')]=(per[e.getAttribute('data-sec')]||0)+1;}
    });
    secs.forEach(function(sec){
      var n=sec.querySelectorAll('article.entry').length;
      if(n){ sec.classList.toggle('hide', terms.length>0 && !(per[sec.id]||0)); }
      var c=sec.querySelector('.count'); if(c) c.textContent=(terms.length?(per[sec.id]||0)+'/':'')+n;
    });
    document.querySelectorAll('nav.side .n[data-sec]').forEach(function(el){
      var id=el.getAttribute('data-sec'), n=document.getElementById(id)?document.getElementById(id).querySelectorAll('article.entry').length:0;
      el.textContent=terms.length?((per[id]||0)+'/'+n):n;
    });
    hits.textContent=terms.length?(shown+' of '+entries.length+' entries match'):(entries.length+' entries');
    if(terms.length){ var d=document.querySelector('details.closedbox'); if(d && (per[Object.keys(per).find(function(k){return k.indexOf('closed-')===0;})]||0)) d.open=true; }
  }
  function openHash(){ var id=(location.hash||'').slice(1); if(!id) return; var el=document.getElementById(id);
    if(el && el.tagName==='DETAILS'){ el.open=true; var box=el.closest('details.closedbox'); if(box) box.open=true; } }
  window.addEventListener('hashchange',openHash); openHash();
  document.getElementById('xall').addEventListener('click',function(){ entries.forEach(function(e){ if(!e.classList.contains('hide')) e.open=true; }); });
  document.getElementById('call').addEventListener('click',function(){ entries.forEach(function(e){ e.open=false; }); });
  q.addEventListener('input',function(){ apply(); var vis=entries.filter(function(e){return !e.classList.contains('hide');});
    if(q.value.trim() && vis.length<=4) vis.forEach(function(e){ e.open=true; }); });
  apply();
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", default=os.path.join(ROOT, "docs", "CHORD_BUGLIST.md"))
    ap.add_argument("--closed", default=os.path.join(ROOT, "docs", "CHORD_BUGLIST_CLOSED.md"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    o_title, o_pre, o_secs = parse(a.open)
    c_title, c_pre, c_secs = parse(a.closed)
    # A #NNN reference lands on the open entry, a plain one before a "residual", before the closed one.
    ids = {}
    for secs in (o_secs, c_secs):
        for plain in (True, False):
            for sec in secs:
                for e in sec["entries"]:
                    if e["id"] and (not e["suffix"]) == plain:
                        ids.setdefault(e["id"], e["anchor"])
    n_open = sum(len(s["entries"]) for s in o_secs)
    n_closed = sum(len(s["entries"]) for s in c_secs)
    stamp = git_stamp()
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    nav = ['<div class="grp">Open</div>']
    for s in o_secs:
        sid = "open-%s" % slug(s["title"])
        short = re.sub(r"\s+—.*$", "", s["title"])
        nav.append('<a href="#%s">%s%s</a>' % (sid, html.escape(short),
                   (' <span class="n" data-sec="%s">%d</span>' % (sid, len(s["entries"]))) if s["entries"] else ""))
    nav.append('<div class="grp">Closed</div>')
    for s in c_secs:
        sid = "closed-%s" % slug(s["title"])
        short = re.sub(r"\s+—.*$", "", s["title"])
        nav.append('<a href="#%s">%s%s</a>' % (sid, html.escape(short),
                   (' <span class="n" data-sec="%s">%d</span>' % (sid, len(s["entries"]))) if s["entries"] else ""))

    page = []
    page.append("<title>CHORD GNSS Open Items</title>")
    page.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
    page.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">')
    page.append("<style>%s</style>" % CSS)
    page.append('<div class="top"><h1>CHORD GNSS open items</h1>'
                '<span class="stamp">docs/CHORD_BUGLIST.md at <span class="mono">%s</span>, rendered %s &middot; %d open, %d closed</span>'
                '<input id="flt" type="search" placeholder="filter: id, word, chain&hellip;" aria-label="filter entries">'
                '<span class="hits" id="hits"></span>'
                '<button id="xall" type="button">expand all</button><button id="call" type="button">collapse all</button></div>'
                % (stamp, when, n_open, n_closed))
    page.append('<div class="wrap"><nav class="side">%s</nav><main>' % "".join(nav))
    page.append('<details class="about"><summary>About this list: how to read the marks, the axis it is sorted on, '
                'the instrument as configured</summary>%s</details>' % blocks(o_pre, ids))
    page.append(render_sections(o_secs, ids, "open"))
    page.append('<details class="closedbox"><summary>%s <span class="n">%d entries, from CHORD_BUGLIST_CLOSED.md</span></summary>'
                % (html.escape(c_title), n_closed))
    page.append('<details class="about"><summary>About the closed ledger</summary>%s</details>' % blocks(c_pre, ids))
    page.append(render_sections(c_secs, ids, "closed"))
    page.append("</details>")
    page.append('<div class="foot">Generated by scripts/gnss/buglist_page.py from the two markdown files; the files are the '
                'record, this page is a view. Entry anchors are <span class="mono">#b&lt;id&gt;</span>.</div>')
    page.append("</main></div>")
    page.append("<script>%s</script>" % JS)
    open(a.out, "w", encoding="utf-8").write("\n".join(page))
    print("wrote %s: %d open, %d closed entries, %d bytes, source %s" % (a.out, n_open, n_closed, os.path.getsize(a.out), stamp))


if __name__ == "__main__":
    main()
