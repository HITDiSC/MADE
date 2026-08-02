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
- `1. token=capital | score=0.5373229384422302 | sequence=Paris is the capital of Franc@@`
- `2. token=heart | score=0.05475250259041786 | sequence=Paris is the heart of Franc@@`
- `3. token=city | score=0.04573303088545799 | sequence=Paris is the city of Franc@@`
- `4. token=home | score=0.0439845509827137 | sequence=Paris is the home of Franc@@`
- `5. token=Capital | score=0.03538456931710243 | sequence=Paris is the Capital of Franc@@`

## Output Contract
- Only the files and fields listed above are part of the expected output.
- Keep the output format minimal and case-compatible.
