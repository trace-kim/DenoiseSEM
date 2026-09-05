"""Render a docs markdown report as a self-contained HTML page (images embedded, math via KaTeX).

- images embedded as data URIs, shown at full column width, click to zoom to
  native pixels (scrollable);
- math kept for KaTeX auto-render (loaded from cdnjs);
- same palette / fonts as the earlier pages in edge_denoise/docs/.

    python tools/report_to_html.py edge_denoise/docs/fine_feature_report.md edge_denoise/docs/fine_feature_report.html
"""
from __future__ import annotations

import base64
import html
import re
import sys
from pathlib import Path

import markdown

src = Path(sys.argv[1])
out = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")

# ---- protect math from the markdown parser --------------------------------
math_blocks: list[str] = []


def _stash(match: re.Match) -> str:
    math_blocks.append(match.group(0))
    return f"MATHBLOCK{len(math_blocks) - 1}END"


text = re.sub(r"\$\$.+?\$\$", _stash, text, flags=re.DOTALL)
text = re.sub(r"(?<!\\)\$(?!\s)([^$\n]+?)(?<!\s)\$", _stash, text)

# ---- title -----------------------------------------------------------------
title_match = re.match(r"# (.+)", text)
title = title_match.group(1).strip() if title_match else src.stem
body_md = text[title_match.end():] if title_match else text

# ---- markdown -> html ------------------------------------------------------
md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2-3"}})
body = md.convert(body_md)
toc = md.toc

# restore math (escape-safe: the placeholders survived untouched)
for index, block in enumerate(math_blocks):
    body = body.replace(f"MATHBLOCK{index}END", html.escape(block, quote=False))

# ---- embed images ---------------------------------------------------------
def _embed(match: re.Match) -> str:
    attrs = match.group(1)
    src_match = re.search(r'src="([^"]+)"', attrs)
    alt_match = re.search(r'alt="([^"]*)"', attrs)
    path = (src.parent / src_match.group(1)).resolve()
    alt = alt_match.group(1) if alt_match else ""
    if not path.is_file():
        return f'<p class="missing">[missing image {html.escape(str(path))}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    css = "plate diagram" if path.suffix.lower() == ".svg" else "plate"
    return (
        f'<figure class="{css}"><img src="data:{mime};base64,' + data + '" alt="' + alt + '" '
        'title="click to toggle native size" loading="lazy">'
        f"<figcaption>{alt}</figcaption></figure>"
    )


# One pass: an image paragraph becomes a figure (never re-embed a data: URI).
body = re.sub(r"(?:<p>)?<img ((?![^>]*src=\"data:)[^>]*)>(?:</p>)?", _embed, body)

# The contents box goes right before the first section, so the conclusion,
# its summary figure and the TL;DR stay above it at the top of the page.
first_section = body.find("<h2")
toc_html = f'<nav class="tocbox">{toc}</nav>'
body_with_toc = body[:first_section] + toc_html + body[first_section:] if first_section >= 0 else toc_html + body

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
<style>
  :root {{
    --bg: #F2F4F6; --surface: #FFFFFF; --well: #E9EDF1; --ink: #1C2530; --ink2: #475664;
    --ink3: #82909D; --line: #D4DBE2; --beam: #14688F; --beam-soft: rgba(20,104,143,0.12);
    --grain: #A2711C; --code-bg: #E4E9EE;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0F141B; --surface: #171E28; --well: #1E2732; --ink: #E7EDF3; --ink2: #AAB8C5;
      --ink3: #6E7D8C; --line: #2B3644; --beam: #57ACD2; --beam-soft: rgba(87,172,210,0.16);
      --grain: #D8A64C; --code-bg: #10161E;
    }}
  }}
  html {{ background: var(--bg); color: var(--ink); overflow-x: hidden; }}
  body {{ margin: 0; overflow-x: hidden; font: 16px/1.55 "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif; }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 32px 40px 80px; }}
  h1, h2, h3 {{ font-family: "Bricolage Grotesque", Georgia, serif; line-height: 1.15; letter-spacing: -0.01em; }}
  h1 {{ font-size: 2.4rem; margin: 0 0 8px; }}
  h2 {{ font-size: 1.7rem; margin: 56px 0 12px; padding-top: 16px; border-top: 1px solid var(--line); }}
  h3 {{ font-size: 1.25rem; margin: 32px 0 8px; color: var(--ink2); }}
  p, li {{ max-width: 92ch; }}
  a {{ color: var(--beam); }}
  em {{ color: var(--ink2); }}
  code {{ font-family: "IBM Plex Mono", Consolas, monospace; font-size: 0.88em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }}
  pre {{ background: var(--code-bg); padding: 14px 16px; border-radius: 8px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 6px solid var(--beam); margin: 24px 0; padding: 14px 26px; color: var(--ink); background: var(--beam-soft); border-radius: 0 10px 10px 0; font-size: 1.08rem; line-height: 1.6; }}
  blockquote p {{ max-width: 110ch; }}
  nav.tocbox {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 22px; margin: 24px 0 8px; display: inline-block; }}
  nav.tocbox ul {{ margin: 4px 0; padding-left: 18px; }}
  nav.tocbox a {{ text-decoration: none; }}
  nav.tocbox > div > ul > li {{ margin: 3px 0; }}
  .tablewrap {{ overflow-x: auto; margin: 14px 0 22px; }}
  table {{ border-collapse: collapse; font-size: 0.9rem; background: var(--surface); }}
  th, td {{ border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }}
  td {{ min-width: 5ch; }}
  th {{ background: var(--well); font-weight: 700; }}
  tr:nth-child(even) td {{ background: color-mix(in srgb, var(--surface) 88%, var(--well)); }}
  /* Figures break out of the text column to the full browser width (up to 2100 px). */
  figure.plate {{ margin: 28px 0 36px; width: min(calc(100vw - 48px), 2100px); position: relative; left: 50%; transform: translateX(-50%); }}
  figure.plate img {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #181818; cursor: zoom-in; image-rendering: auto; }}
  figure.plate.zoomed {{ overflow-x: auto; }}
  figure.plate.zoomed img {{ width: auto; max-width: none; cursor: zoom-out; }}
  figure.diagram {{ width: min(calc(100vw - 48px), 1600px); }}
  figure.diagram img {{ background: #FFFFFF; }}
  figcaption {{ color: var(--ink2); font-size: 1rem; margin-top: 10px; max-width: 140ch; }}
  figcaption::after {{ content: "  (click the image to view at native pixel size)"; color: var(--ink3); }}
  .hint {{ color: var(--ink3); font-size: 0.86rem; }}
  .katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; }}
</style>
</head>
<body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="hint">HTML rendering of <code>{html.escape(src.name)}</code> · images are embedded at column width — click any figure to toggle native pixel size (scrolls horizontally).</p>
{body_with_toc}
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {{
  if (window.renderMathInElement) {{
    renderMathInElement(document.body, {{
      delimiters: [{{left: "$$", right: "$$", display: true}}, {{left: "$", right: "$", display: false}}],
      throwOnError: false
    }});
  }}
  document.querySelectorAll("figure.plate").forEach(function (fig) {{
    fig.querySelector("img").addEventListener("click", function () {{ fig.classList.toggle("zoomed"); }});
  }});
  document.querySelectorAll("table").forEach(function (t) {{
    var w = document.createElement("div"); w.className = "tablewrap";
    t.parentNode.insertBefore(w, t); w.appendChild(t);
  }});
}});
</script>
</body>
</html>
"""
out.write_text(page, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB", f"{len(math_blocks)} math spans")
