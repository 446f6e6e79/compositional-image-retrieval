#!/bin/sh
# Render the paper-style schematics from TikZ to committed SVGs.
#   sh figures/build.sh
# Sources: figstyle.tex (shared) + architecture.tex + training.tex.
set -e
cd "$(dirname "$0")"

for f in architecture architecture_detail training; do
  echo "==> $f"
  pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" >/dev/null
  # PDF -> SVG (glyph outlines embedded, so it renders identically everywhere)
  pdftocairo -svg "$f.pdf" "$f.svg"
done

# tidy LaTeX intermediates (kept out of git anyway)
rm -f ./*.aux ./*.log ./*.pdf
echo "done: architecture.svg, architecture_detail.svg, training.svg"
