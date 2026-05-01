Compare LLM models using Arena and Artificial Analysis data.

## Comparing Models

When the user asks to compare models:

1. Parse which models or families they want to compare from their message
2. Build the appropriate CLI command:
   - For specific models: `uv run compare-models compare -m "model1,model2"`
   - For model families: `uv run compare-models compare -m "family1,family2" --families`
   - For specific sources only: add `--sources arena` or `--sources artificial_analysis`
   - The CLI auto-generates a report name in `reports/` (e.g., `reports/claude_gpt_2025_05_01_00.md`). Use `-o path` only to override.
   - To also generate a PDF: add `--pdf` (requires pandoc and a LaTeX engine)
3. Run the command from the compare-models project directory (`/Users/anfredet/go/src/github.com/compare-models/`)
4. Read the generated report file (parse the path from the CLI's "Comparison written to ..." output)
5. **Enhance the Key Findings sections with interpretive prose:**
   - Read the Arena Key Findings and AA Key Findings sections in the generated file
   - Rewrite each finding in-place with narrative interpretation, adding:
     - Context about what the numbers mean practically (e.g., "this places it alongside models like X and Y")
     - Relative tier placement and what it implies
     - Implications for deployment decisions (e.g., "well-suited for latency-sensitive applications")
     - Caveats and limitations worth noting
   - Keep the same numbered-list format but with richer, more readable prose
   - Example: transform `**Speed:** X is 2.5x faster (132 vs 52 t/s).` into `**Speed advantage:** X is dramatically faster than comparable Y models: 132 t/s vs 52-55 t/s for the Y 235B variants. This is likely due to X's much smaller active parameter count (13B active vs 22B active) despite having more total parameters.`
6. **Write an Overall Conclusions section and insert it after the intro/section table (before Part 1).** The template reserves this position. Structure it as:
     1. **Overall positioning** — Where each model/family sits (frontier vs mid-tier vs budget) with specific rank and score evidence from both Arena and AA
     2. **Lineup depth** — How broad each family's lineup is (number of models on each platform)
     3. **Value proposition** — Each side's niche: speed/cost vs quality vs breadth, with specific numbers
     4. **Quality profile differences** — STEM-leaning vs humanities-leaning, citing specific category deltas from head-to-head data
     5. **Evaluation coverage** — Note any limitations (different variants evaluated on different platforms, missing data)
     6. **Summary table** — A markdown table comparing key factors side by side:

        | Factor | Model A | Model B |
        |--------|---------|---------|
        | Top-tier quality | ... | ... |
        | Same-tier quality | ... | ... |
        | Speed | ... | ... |
        | Latency | ... | ... |
        | Price | ... | ... |
        | Context window | ... | ... |
        | Strength categories | ... | ... |
        | Weakness categories | ... | ... |
        | Model variety | ... | ... |
        | Open weights | ... | ... |

     7. **Bottom line** — 2-3 sentence prose summary of when to pick each model family
   - Write the conclusions using the actual data from the report — cite specific numbers, ranks, scores, and category names
7. Summarize the key findings for the user
8. Offer to explain specific sections in more detail

## Syncing Artificial Analysis Data

AA data is fetched from the AA API and cached locally. When a comparison reports
"0 models found" from AA, the user likely needs to sync:

```bash
export AA_API_KEY=your_key
uv run compare-models sync-aa
```

This fetches all models (~500+) in a single API call and caches them at
`~/.cache/compare-models/aa_models.json`. The `--aa-data` flag on `compare`
can override this with a custom JSON file.

## Usage Examples

- "Compare trinity-large-preview with qwen3-235b-a22b"
- "How does Trinity stack up against Qwen models?" (use --families)
- "Compare just Arena data for trinity and qwen" (use --sources arena)
- "Sync the AA data" (run sync-aa)

## Notes

- Arena data requires network access (downloads from HuggingFace)
- AA data comes from the local cache — sync with `compare-models sync-aa`
- Model aliases are in `data/model_aliases.json`
