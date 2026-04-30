Compare LLM models using Arena and Artificial Analysis data.

## Comparing Models

When the user asks to compare models:

1. Parse which models or families they want to compare from their message
2. Build the appropriate CLI command:
   - For specific models: `uv run compare-models -m "model1,model2" -o comparison.md`
   - For model families: `uv run compare-models -m "family1,family2" --families -o comparison.md`
   - For specific sources only: add `--sources arena` or `--sources artificial_analysis`
3. Run the command
4. Read the generated comparison.md file
5. Summarize the key findings for the user
6. Offer to explain specific sections in more detail

## Updating Artificial Analysis Data

AA data is stored in `data/artificial_analysis.json`. When the user asks to add models,
or when a comparison reports "0 models found" from AA for models that likely exist:

1. Use WebSearch to find the model on artificialanalysis.ai (search for
   `artificialanalysis.ai <model name> intelligence speed price`)
2. Extract these fields from the search results or WebFetch:
   - `name`: Display name as shown on AA (e.g., "Claude Opus 4.7 (max)")
   - `slug`: URL slug from the model page URL
   - `organization`: Model developer
   - `intelligence_index`: AA Intelligence Index score (integer)
   - `speed_tps`: Output tokens/second (float or null)
   - `ttft_s`: Time to first token in seconds (float or null)
   - `input_price_per_1m`: Input price per 1M tokens (float or null)
   - `output_price_per_1m`: Output price per 1M tokens (float or null)
   - `context_window`: Context window in tokens (int or null)
   - `params_total_b`: Total parameters in billions (float or null)
   - `params_active_b`: Active parameters in billions (float or null)
   - `reasoning`: Whether this is a reasoning model (boolean)
   - `url`: Full URL to the AA model page
   - `accessed_date`: Today's date (YYYY-MM-DD)
3. Read the current `data/artificial_analysis.json` and append the new entry
4. Run `make test` to verify the JSON is valid

## Usage Examples

- "Compare trinity-large-preview with qwen3-235b-a22b"
- "How does Trinity stack up against Qwen models?" (use --families)
- "Compare just Arena data for trinity and qwen" (use --sources arena)
- "Add Claude Opus 4.7 to the AA data"
- "Update the AA data for the latest GPT models"

## Notes

- Arena data requires network access (downloads from HuggingFace)
- AA data comes from `data/artificial_analysis.json` — update it using the workflow above
- Model aliases are in `data/model_aliases.json`
- When a model has multiple effort levels (e.g., xhigh, high, medium, low),
  add each as a separate entry
