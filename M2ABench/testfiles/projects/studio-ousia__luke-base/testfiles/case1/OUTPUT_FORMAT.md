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
- `1. token=capital | score=0.9762480854988098 | sequence=Paris is the capital of France.`
- `2. token=Capital | score=0.017404813319444656 | sequence=Paris is the Capital of France.`
- `3. token=city | score=0.0011283897329121828 | sequence=Paris is the city of France.`
- `4. token=center | score=0.0011000799713656306 | sequence=Paris is the center of France.`
- `5. token=centre | score=0.0010632558260113 | sequence=Paris is the centre of France.`

## Output Contract
- Only the files and fields listed above are part of the expected output.
- Keep the output format minimal and case-compatible.
