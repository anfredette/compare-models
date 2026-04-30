Compare LLM models using Arena and Artificial Analysis data.

## Instructions

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

## Usage Examples

- "Compare trinity-large-preview with qwen3-235b-a22b"
- "How does Trinity stack up against Qwen models?" (use --families)
- "Compare just Arena data for trinity and qwen" (use --sources arena)

## Notes

- Arena data requires network access (downloads from HuggingFace)
- AA data comes from data/artificial_analysis.json (may need manual updates)
- Model aliases are in data/model_aliases.json
