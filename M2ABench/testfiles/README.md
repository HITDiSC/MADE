# M2ABench: AAAI Review Supplement

## 1. Purpose and Scope

This archive is an **anonymous review subset** of M2ABench prepared for the
AAAI supplementary-material submission. It is not the complete benchmark
distribution.

The complete internal collection contains:

- 122 resource directories;
- five test cases per resource, for 610 cases in total; and
- one source-paper PDF per resource.

A lossless ZIP of the complete collection is approximately 678 MB. The source
PDFs and several binary scientific inputs account for most of that size and
are already internally compressed, so stronger ZIP settings provide little
additional reduction. To remain within the supplementary upload budget, this
review archive contains a deterministic subset:

- **80 resource directories**;
- **100 unmodified test cases**; and
- **15 source-paper PDFs**.

The omitted resources, cases, and PDFs are absent only from this size-limited
review archive. Their omission does not mean that they are absent from the
complete benchmark. Relative to the complete collection, this archive omits
42 resource directories, 510 test cases, and 107 source-paper PDFs.

## 2. Exact Selection Policy

The subset was selected using file type, directory name, and file size only.
Benchmark predictions, expected outputs, scores, and model performance were
not used for selection.

### 2.1 Resource directories

The archive contains:

- all 63 resources whose original case inputs use only the plain-data
  extensions `.txt`, `.md`, `.json`, `.csv`, and `.tsv`; and
- one representative resource for each of five broader special-input
  categories; and
- seven additional resources selected to expose special file formats not
  represented by those five core cases; and
- five further resources selected to expose structured and multimodal input
  combinations.

The five special-input representatives are:

| Category | Resource | Retained case | Retained input |
|---|---|---:|---|
| Image | `SatMAE-main` | `case1` | `.jpg` |
| Audio | `canary-1b-v2` | `case5` | `.wav` |
| Single-cell data | `transcriptformer-main` | `case3` | `.h5ad` |
| Scientific signal | `EEGPT-main` | `case3` | `.edf` |
| Biological sequence | `ESM-AA-main` | `case5` | `.a3m` |

These five categories are a compact packaging taxonomy used to preserve
representative non-text inputs. They are not intended to replace the more
detailed task or domain taxonomy of the complete benchmark. Within each
category, the retained case was chosen to provide a substantive, inspectable
input rather than the smallest possible placeholder. The five retained case
directories have original sizes of approximately 15.38 MB, 1.29 MB, 27.34 MB,
7.97 MB, and 0.44 MB, respectively.

The seven additional format representatives are:

| Added format coverage | Resource | Retained case | Retained input |
|---|---|---:|---|
| PNG image | `RADIO-main` | `case3` | `.png` |
| FLAC audio | `wav2vec2-base-960h` | `case5` | `.flac` |
| OPUS audio | `owsm_ctc_v4_1B` | `case4` | `.opus` |
| Geospatial raster | `hls-foundation-os-main` | `case1` | `.tif` |
| Serialized tensor | `galileo-main` | `case1` | `.pt` |
| Protein sequence and structure | `ProSST-main` | `case1` | `.fasta`, `.pdb` |
| Biomedical signal matrix | `ecg-fm-main` | `case1` | `.mat` |

The five further structured-input representatives are:

| Added coverage | Resource | Retained case | Retained input |
|---|---|---:|---|
| Arrow table | `LangCell-main` | `case4` | `.arrow` |
| JSON Lines record | `MMedLM-main` | `case1` | `.jsonl` |
| Source record | `BioMedLM-main` | `case3` | `.source` |
| Image and tensor bundle | `GeoLink_NeurIPS2025-main` | `case5` | `.png`, `.pt` |
| Vision-language image | `MiniCPM-o-2_6` | `case4` | `.png`, `.txt` |

### 2.2 Test cases

Every included resource has at least one complete case directory.

- For the 63 plain-input resources, `case1` is retained.
- The first 20 plain-input resources in case-insensitive alphabetical order
  additionally retain `case2`.
- Each of the 17 special-input representatives retains the single case listed
  in the tables above.

This produces exactly:

- 20 resources with two cases: 40 cases;
- 43 plain-input resources with one case: 43 cases; and
- 17 special-input resources with one case: 17 cases.

The total is therefore **100 cases across 80 resources**. Every retained case
is copied without modifying its input, expected output, or output-format
description.

### 2.3 Source-paper PDFs

Only the 15 smallest PDFs, measured by original byte size among the 80
included resources, are packaged:

1. `medAlpaca-main`
2. `allenai__scibert_scivocab_uncased`
3. `FacebookAI__roberta-base`
4. `vinai__bertweet-base`
5. `albert__albert-base-v2`
6. `BioClinical-ModernBERT-main`
7. `microsoft__DialoGPT-small`
8. `BioMistral-main`
9. `facebook__bart-base`
10. `medfound-main`
11. `owsm_ctc_v4_1B`
12. `google__byt5-small`
13. `google__long-t5-local-base`
14. `google__electra-small-discriminator`
15. `funnel-transformer__small`

The remaining 65 included resource directories intentionally have no local
PDF in this archive. This is not a missing-file or extraction error. Public
source locations for the resources are recorded in
`githublink_reference.xlsx`.

## 3. Archive Contents

```text
M2ABench_AAAI27_supplementary/
|-- README.md
|-- SELECTION.json
|-- githublink_reference.xlsx
`-- projects/
    `-- <resource-name>/
        |-- <source-paper>.pdf          # present for 15 resources only
        `-- testfiles/
            `-- <selected-case>/        # one or two cases per resource
```

Each retained case directory preserves its original internal structure, which
may include:

- `input/`: the test input files;
- `output/`: the expected output or reference artifact; and
- `OUTPUT_FORMAT.md`: the expected response format.

`SELECTION.json` is the machine-readable source of truth for the 80 included
resources, the exact case names retained for each resource, the 17
special-input representatives, and the 15 packaged PDFs.

`githublink_reference.xlsx` contains the verified public GitHub reference for
all 122 resources in the complete collection, including resources that are
not present in this review subset.

## 4. How to Inspect the Supplement

1. Open a resource directory under `projects/`.
2. Open one of its retained directories under `testfiles/`.
3. Inspect the files under `input/`.
4. Read `OUTPUT_FORMAT.md` when present.
5. Compare against the corresponding material under `output/`.
6. Consult `githublink_reference.xlsx` for the public source repository.
7. Consult `SELECTION.json` when an exact, programmatic inventory is needed.

Because this is a review subset, it must not be used to infer the total number
of tasks in the full benchmark or to reproduce aggregate benchmark results
that require all 122 resources and all 610 cases.

## 5. Integrity and Interpretation

- Resource and case names retain their original identifiers.
- Selected case files are included without content-level recompression or
  transformation; ZIP compression only changes the archive representation.
- Missing PDFs and missing case numbers are intentional consequences of the
  documented selection policy.
- The subset was constructed independently of benchmark outcomes.

## 6. Source and License Notice

Source papers, datasets, and public repositories remain subject to their
original copyright, license, and usage terms. Inclusion of a file or a public
reference in this anonymous review package does not relicense third-party
material.
