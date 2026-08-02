# OUTPUT_FORMAT

This file describes the cleaned expected output format for this case.

## Output Location
- Directory: `output/`

## Required Output Files
- `output.txt`

## File Formats

### `output.txt`
- Plain UTF-8 text output. Expected line structure:
- `predictions_top5:`
- `1. token=export | score=0.0010319497669115663 | sequence=Paris is the export of France.`
- `2. token=Ash | score=0.0010138326324522495 | sequence=Paris is the Ash of France.`
- `3. token=haz | score=0.0009779150132089853 | sequence=Paris is the haz of France.`
- `4. token=issue | score=0.000851769931614399 | sequence=Paris is theissue of France.`
- `5. token=1996 | score=0.0008150769281201065 | sequence=Paris is the 1996 of France.`

## Output Contract
- Only the files and fields listed above are part of the expected output.
- Keep the output format minimal and case-compatible.
