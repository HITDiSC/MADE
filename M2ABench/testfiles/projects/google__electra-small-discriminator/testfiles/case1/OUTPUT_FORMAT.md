# OUTPUT_FORMAT

This file describes the cleaned expected output format for this case.

## Output Location
- Directory: `output/`

## Required Output Files
- `output.txt`

## File Formats

### `output.txt`
- Plain UTF-8 text output. Expected line structure:
- `token_predictions:`
- `132. token=fake | predicted_fake=1 | prob=0.975309`
- `260. token=anime | predicted_fake=1 | prob=0.596862`
- `261. token=opening | predicted_fake=1 | prob=0.577226`

## Output Contract
- Only the files and fields listed above are part of the expected output.
- Keep the output format minimal and case-compatible.
