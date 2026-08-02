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
- `1. token=capital | score=0.8692432045936584 | sequence=paris is the capital of france.`
- `2. token=heart | score=0.016246648505330086 | sequence=paris is the heart of france.`
- `3. token=city | score=0.015057769604027271 | sequence=paris is the city of france.`
- `4. token=center | score=0.007126745767891407 | sequence=paris is the center of france.`
- `5. token=home | score=0.005901882890611887 | sequence=paris is the home of france.`

## Output Contract
- Only the files and fields listed above are part of the expected output.
- Keep the output format minimal and case-compatible.
