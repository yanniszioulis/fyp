-- Pandoc Lua filter: every table → `tabular` wrapped in
-- \resizebox{\textwidth}{!}{...} so wide tables auto-shrink to page width.
--
-- Pandoc's longtable output is unsuitable for \resizebox in two ways:
--   1. longtable spans pages and isn't a single TeX box.
--   2. cells are wrapped in `\begin{minipage}[b]{\linewidth}...\end{minipage}`,
--      and column specs are `>{...\arraybackslash}p{<frac of \linewidth>}`.
--      Once columns become simple `l`/`r`/`c`, both `\linewidth` references
--      refer to the *outer* line width, so each cell expands to the full
--      page width and the natural table becomes ncols × textwidth — which
--      \resizebox then shrinks back, producing tiny text.
--
-- Fix: convert longtable → tabular, rewrite column specs to single-letter
-- alignments, AND strip the minipage wrappers so cells take their natural
-- (math/text) width.

local function rewrite_spec(spec)
  spec = spec:gsub('>%s*{\\raggedright\\arraybackslash}%s*p%b{}',  'l')
  spec = spec:gsub('>%s*{\\raggedleft\\arraybackslash}%s*p%b{}',   'r')
  spec = spec:gsub('>%s*{\\centering\\arraybackslash}%s*p%b{}',    'c')
  spec = spec:gsub('@{}',                                          '')
  spec = spec:gsub('%s+',                                          '')
  return spec
end

local function rewrite_one(latex)
  -- longtable → tabular (with optional `[...]` options block).
  latex = latex:gsub(
    '\\begin{longtable}(%b[])(%b{})',
    function(_, spec)
      return '\\begin{tabular}{' .. rewrite_spec(spec:sub(2, -2)) .. '}'
    end)
  latex = latex:gsub(
    '\\begin{longtable}(%b{})',
    function(spec)
      return '\\begin{tabular}{' .. rewrite_spec(spec:sub(2, -2)) .. '}'
    end)
  latex = latex:gsub('\\end{longtable}', '\\end{tabular}')

  -- longtable-only row markers.
  latex = latex:gsub('\\endfirsthead', '')
  latex = latex:gsub('\\endhead',      '')
  latex = latex:gsub('\\endfoot',      '')
  latex = latex:gsub('\\endlastfoot',  '')

  -- Strip per-cell minipage wrappers so cells take natural width.
  latex = latex:gsub('\\begin{minipage}%[%w%]{\\linewidth}\\raggedright%s*', '')
  latex = latex:gsub('\\begin{minipage}%[%w%]{\\linewidth}\\raggedleft%s*',  '')
  latex = latex:gsub('\\begin{minipage}%[%w%]{\\linewidth}\\centering%s*',   '')
  latex = latex:gsub('%s*\\end{minipage}',                                   '')

  return latex
end

function Table(t)
  local latex = pandoc.write(pandoc.Pandoc({t}), 'latex')
  latex = rewrite_one(latex)
  return pandoc.RawBlock(
    'latex',
    '\\noindent\\resizebox{\\textwidth}{!}{%\n' .. latex .. '}\n'
  )
end
