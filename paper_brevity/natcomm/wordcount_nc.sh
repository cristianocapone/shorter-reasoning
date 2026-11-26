#!/bin/bash
# Word counts against the Nature Communications Article limits (read-only on the source file).
# Adapted from ../nmi/wordcount.sh. Works on both layouts:
#   - natcomm layout: \section{Introduction}/\section{Results}/\section{Discussion}/\section{Methods}, floats inline;
#   - nmi layout: %TC:ignore blocks, %ABSTRACT-BEGIN/END markers, \sect{...} headings, floats at the end.
# Limits (https://www.nature.com/ncomms/submit/article, read 2026-11-02): main text "ideally" <= 5,000 words, not
# including Abstract, Methods, References and Figure legends; abstract <= 200 words, no references (the 2019 brief
# guide and manuscript checklist say ~150); title <= 15 words; legends <= 350 words each; Methods "typically less than
# 3000 words"; up to 10 display items; references "should not exceed 70" as a guide.
# Three counts are printed: texcount (text + headers), the same with every inline formula as one word, and
# detex | wc -w (closest to a word-processor count).
# Usage: ./wordcount_nc.sh [path/to/main.tex]   (default: main.tex next to this script)
set -euo pipefail
src="${1:-$(dirname "$0")/main.tex}"
src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# region START END: lines from the first line containing START up to (not including) the first later line containing
# END. Markers are passed through the environment so that awk does not turn "\b" or "\s" into escape sequences.
region()   { A="$1" B="$2" awk 'index($0,ENVIRON["A"])&&!s{f=1;s=1;print;next} f&&index($0,ENVIRON["B"]){f=0} f' "$src" | grep -v '^%TC:' || true; }
nofloats() { awk '/^\\begin\{(figure|table)\}/{skip=1} !skip{print} /^\\end\{(figure|table)\}/{skip=0}'; }
words()    { texcount -1 -nobib -sum=1,1,1,0,0,0,0 "$1" 2>/dev/null | grep -v SUMWEIGHTS | tail -1 | awk '{print $1+0}'; }
plusmath() { texcount -1 -nobib -sum=1,1,1,0,0,1,1 "$1" 2>/dev/null | grep -v SUMWEIGHTS | tail -1 | awk '{print $1+0}'; }
wp()       { detex -n "$1" 2>/dev/null | wc -w | tr -d ' '; }
row()      { printf '%-44s %6s %8s %9s   %s\n' "$1" "$(words "$2")" "$(plusmath "$2")" "$(wp "$2")" "$3"; }

if grep -q '\\section{Introduction}' "$src"; then
  layout=natcomm
  region '\begin{abstract}' '\end{abstract}' | sed '1d' > "$tmp/abstract.tex"
  region '\section{Introduction}' '\section{Methods}' | nofloats > "$tmp/main.tex"
  region '\section{Introduction}' '\section{Results}' | nofloats > "$tmp/intro.tex"
  region '\section{Results}' '\section{Discussion}' | nofloats > "$tmp/results.tex"
  region '\section{Discussion}' '\section{Methods}' | nofloats > "$tmp/discussion.tex"
  region '\section{Methods}' '\section*{Data availability}' > "$tmp/methods.tex"
  region '\section*{Data availability}' '\bibliography' > "$tmp/statements.tex"
else
  layout=nmi
  region '%ABSTRACT-BEGIN' '%ABSTRACT-END' > "$tmp/abstract.tex"
  # main text: from the line after the abstract block to the %TC:ignore that opens the Methods
  awk '/^%TC:endignore/&&!s{s=1;f=1;next} f&&/^%TC:ignore/{exit} f' "$src" | nofloats > "$tmp/main.tex"
  awk '/^%TC:endignore/&&!s{s=1;f=1;next} f&&/\\sect\{Results\}/{exit} f' "$src" > "$tmp/intro.tex"
  region '\sect{Results}' '\sect{Discussion}' | nofloats > "$tmp/results.tex"
  A='\sect{Discussion}' awk 'index($0,ENVIRON["A"]){f=1} f&&/^%TC:ignore/{exit} f' "$src" > "$tmp/discussion.tex"
  region '\sect{Methods}' '\sect{Data availability}' > "$tmp/methods.tex"
  region '\sect{Data availability}' '% ------------------------------------------------------------------ display items' > "$tmp/statements.tex"
fi

echo "== $src ($layout layout)"
printf '%-44s %6s %8s %9s   %s\n' "part" "texcnt" "+formula" "detex/wc" "Nature Communications guide"
row "Abstract" "$tmp/abstract.tex" "<= 200 (current page); ~150 (2019 guide/checklist)"
row "Main text: Introduction + Results + Discussion" "$tmp/main.tex" "ideally <= 5,000"
row "  Introduction" "$tmp/intro.tex" "< 1,000 (2019 checklist)"
row "  Results" "$tmp/results.tex" ""
row "  Discussion" "$tmp/discussion.tex" "succinct, no subheadings"
row "Methods" "$tmp/methods.tex" "typically < 3,000"
row "Statements (data, code, acknowledgements ...)" "$tmp/statements.tex" "not counted"

echo "== Legends (caption text only; table cells not counted): limit 350 words each"
# floats after the Supplementary Information heading (natcomm) or the Extended Data heading (nmi) are not main items
boundary=$(grep -n -m1 -E '\\section\*\{Supplementary Information\}|\\sect\{Extended Data\}' "$src" | cut -d: -f1 || true)
boundary=${boundary:-999999}
awk -v d="$tmp" '/^\\begin\{(figure|table)\}/{n++; f=sprintf("%s/float%02d.tex", d, n); print NR > (f ".line")} f{print > f} /^\\end\{(figure|table)\}/{f=""}' "$src"
nmain=0; nsupp=0
for f in "$tmp"/float*.tex; do
  [ -e "$f" ] || continue
  awk '/\\caption\{/{c=1} c{print} /\\label\{/{c=0}' "$f" > "$f.cap"
  lab=$(grep -o 'label{[^}]*}' "$f" | head -1 | sed 's/label{//; s/}//'); typ=$(head -1 "$f" | grep -o 'figure\|table')
  if [ "$(cat "$f.line")" -gt "$boundary" ]; then
    kind=$([ "$layout" = nmi ] && echo "Extended Data" || echo "Supplementary"); nsupp=$((nsupp+1))
  else kind=main; nmain=$((nmain+1)); fi
  printf '   %-6s %-14s %-13s %4s words (%s with formulas)\n' "$typ" "$lab" "$kind" "$(words "$f.cap")" "$(plusmath "$f.cap")"
done
echo "== Display items: $nmain in the main paper (limit 10, 'depending on the word count'); $nsupp after the"
echo "   Supplementary/Extended Data heading. Nature Communications has no Extended Data: such items must become main"
echo "   display items or Supplementary items in a separate Supplementary Information file."
bbl="$(dirname "$src")/main.bbl"
[ -f "$bbl" ] && echo "== References: $(grep -c '^\\bibitem' "$bbl") (guide: should not exceed 70)"
