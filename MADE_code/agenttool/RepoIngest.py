"""RepoIngest phase: ingest a paper PDF, locate its GitHub link, clone the
repository, clean its READMEs, and pick the deployment task.

This phase replaces the previous PDFRead + GithubClone pair. They were a
strict 1->2 sequence (pdf_path -> github_link -> repo_root + cleaned_readmes
-> task) with no fan-in / fan-out, so keeping them as separate phases just
forced the phase manager to make a useless control-flow decision in between.
Merging removes that decision and gives the LLM a single linear playbook
from raw paper to a downstream-ready repo state.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pypdf
import shutil
from agenttool.base_phase import BasePhase
from agenttool.task_selection_tool import TaskSelectionTool
from agenttool.tool import linux_command
from backend.query import json_query


file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")


# Locate a github.com link inside PDF-extracted text. The scheme is OPTIONAL:
# PDFs frequently render the link as a bare "github.com/owner/repo" (the http
# prefix lives in a hyperlink annotation that text extraction drops). The path
# run is captured greedily up to the next whitespace / closing bracket / quote;
# all the messy trimming (glued words, concatenated second URLs, footnote
# markers, deeper paths) is handled afterwards by _clean_github_url.
GITHUB_URL_RE = re.compile(
    r"""(?xi)
    (?:https?://)?
    (?:www\.)?
    github\.com/
    (?P<path>[^\s)\]}>"'<]+)
    """
)

# Words/markers that PDF text extraction frequently glues onto the end of a
# repo name because there is no whitespace between the URL and the next token
# (e.g. "github.com/owner/RepoFigure 3 shows ..." -> "RepoFigure"). Stripped
# from the tail of the repo segment. Capitalized paper-structure words plus a
# few lowercase connectives actually observed glued on in the benchmark
# ("...BioMistralet al.", "...fairseqand", "...MMedLMThe").
_TRAILING_JUNK_WORDS = (
    "The", "This", "These", "We", "Our", "Its", "It", "And", "For",
    "Where", "With", "From", "Here", "See", "Note", "Using", "Based",
    "Figure", "Fig", "Table", "Tab", "Section", "Sec", "Appendix",
    "Abstract", "Introduction", "Published", "Preprint", "Contents",
    "Sequence", "Results", "Result", "Methods", "Method", "Model",
    "Github", "GitHub", "Code", "Available", "Project", "Page",
    "Nowadays", "Now", "Time",
    # Sentence-opener words seen glued on after a repo when the next sentence
    # starts immediately (no space). Multi-character only, so they cannot eat
    # into a real repo name the way short connectives could.
    "What", "Which", "While", "When", "However", "Moreover", "Furthermore",
    "Specifically", "Recently", "Recent", "Given", "Since", "Although",
    "Compared", "Unlike", "Following", "Inspired", "Importantly", "Notably",
    "Acknowledgments", "Acknowledgements",
)
# NOTE: deliberately NO lowercase connectives ("et", "and", "the") here -
# they eat into real repo names (MPNet -> MPN, BERTweet -> BERTwe, xlnet ->
# xln), which costs far more than the rare "...et al." glue they would fix.

# Path / name fragments that mark a link as a dataset, release asset, license,
# wiki, or single-file blob rather than the paper's own source repository.
_NON_REPO_PATH_HINTS = (
    "dataset", "/raw/", "/releases/", "/blob/", "license",
    ".zip", ".tar", ".gz", "/wiki/", "/issues/", "/pull/",
)

# Phrases that, appearing shortly BEFORE a github URL, strongly signal the link
# is the authors' own code/model release rather than a cited dependency.
_CODE_CUE_WORDS = (
    "code", "implementation", "available", "released", "release",
    "official", "project page", "source", "repository", "repo",
    "github", "open-source", "open source", "our model", "weights",
)

# Generic infrastructure / baseline repos that are almost never a paper's
# headline release. Mildly penalized so a better-cued candidate wins, and a
# paper whose ONLY github link is one of these gets routed to the LLM fallback.
_REPO_DENYLIST = {
    "pytorch/fairseq",
    "pytorch/audio",
    "nvidia/apex",
    "open-mmlab/mmsegmentation",
    "tatsu-lab/stanford_alpaca",
    "google/sentencepiece",
    "google/cld3",
}


class RepoIngest(BasePhase):
    name: str = "RepoIngest"
    description: str = (
        "Ingest the source paper PDF, locate the GitHub repository it references, "
        "clone the repo, clean its READMEs into a downstream-friendly form, and "
        "identify the deployment task to enable."
    )
    goal: str = (
        "have a cloned repo, cleaned READMEs, and an identified deployment task "
        "ready for downstream phases"
    )
    tools_schemas: List[Dict[str, Any]] = [
        {"name": "read_pdf", "description": "Read the source PDF and extract its full text. Only needed when github_link is not already available. requires: pdf_path. produces: pdf_content.", "args": {"pdf_path": Path}},
        {"name": "github_link_extract_with_text", "description": "(preferred when available) Try to find a github.com URL in the PDF text via regex. Cheap and deterministic. requires: pdf_content. produces: github_link on success. If no link is found, fall back to github_link_extract_with_llm.", "args": {"pdf_content": Dict[str, Any]}},
        {"name": "github_link_extract_with_llm", "description": "Fallback link extractor that asks an LLM to find the GitHub URL when the regex pass fails. The result is still validated against the github.com pattern, so hallucinated URLs are rejected. requires: pdf_content. produces: github_link.", "args": {"pdf_content": Dict[str, Any]}},
        {"name": "git_clone", "description": "Clone the GitHub repository. If github_link is already available, skip read_pdf and link extraction and call this directly. Also writes cleaned READMEs to <repo_root>/.autodeploy/readmes.json as a side effect so task_selection has them available. requires: github_link. produces: repo_root, cleaned_readmes_path.", "args": {"github_link": str}},
        {"name": "task_selection", "description": "Inspect the cloned repo + cleaned READMEs and decide which deployment task to enable. Uses test case expected output to help disambiguate tasks. This is the terminal step of the phase. requires: repo_root, cleaned_readmes_path, test_file_dir. produces: task.", "args": {"repo_root": Path, "cleaned_readmes_path": Path}},
    ]
    allowed_parallel_phases: List[str] = []
    suggested_next_phases: List[str] = ["weightresolve", "dockersetup"]

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self._exists_cache: Dict[str, Optional[bool]] = {}
        self.tools = {
            "read_pdf": self.read_pdf,
            "github_link_extract_with_text": self.github_link_extract_with_text,
            "github_link_extract_with_llm": self.github_link_extract_with_llm,
            "git_clone": self.git_clone,
            "task_selection": self.task_selection,
        }

        prompt_path = os.path.join(project_path, "prompts")
        with open(os.path.join(prompt_path, "pdfread_prompt.json"), "r", encoding="utf-8") as f:
            self.pdfread_prompt = json.load(f)
        with open(os.path.join(prompt_path, "readme_prompt.json"), "r", encoding="utf-8") as f:
            self.readme_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        # Only task_selection terminates the phase. Earlier link-extraction
        # tools were boundary tools when PDFRead was a separate phase, but in
        # the merged playbook they are stepping stones - they must still be
        # followed by git_clone + task_selection before the phase is done.
        return tool_name == "task_selection"

    def tool_arguments(self, tool_name: str) -> Dict[str, Any]:
        tool_arguments_dict = {
            "read_pdf": {"pdf_path": Path},
            "github_link_extract_with_text": {"pdf_content": Dict[str, Any]},
            "github_link_extract_with_llm": {"pdf_content": Dict[str, Any]},
            "git_clone": {"github_link": str},
            "task_selection": {"repo_root": Path, "cleaned_readmes_path": Path},
        }
        return tool_arguments_dict[tool_name]

    # ------------------------------------------------------------------
    # PDF read + GitHub link extraction (formerly PDFRead)
    # ------------------------------------------------------------------

    def read_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from the PDF and store it as a temporary variable."""
        pdf_content: Dict[str, Any] = {
            'text': '',
            'pages': [],
            'num_pages': 0,
        }
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                pdf_content['num_pages'] = len(pdf_reader.pages)
                for page_num in range(pdf_content['num_pages']):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.replace('\n', '')
                    pdf_content['pages'].append(page_text)
                    pdf_content['text'] += page_text
            return [{
                "value": pdf_content,
                "storage": "temporary",
                "variable_name": "pdf_content",
            }]
        except FileNotFoundError:
            return [{
                "value": f"PDF file not found: {pdf_path}",
                "storage": "error",
                "variable_name": "error_message",
            }]
        except Exception as e:
            return [{
                "value": f"Error reading PDF file: {str(e)}",
                "storage": "error",
                "variable_name": "error_message",
            }]

    @staticmethod
    def _clean_github_url(raw: str) -> Optional[str]:
        """Normalize a raw 'github.com/...' fragment into a canonical
        ``https://github.com/<owner>/<repo>`` URL, or None if it is not a usable
        repository link.

        Handles the messy reality of URLs pulled from PDF-extracted text:
        - optional / missing scheme;
        - a second URL glued straight on after the first
          ("github.com/google/cld34https://github.com/..." -> "google/cld3...");
        - sentence punctuation and paper-structure words glued onto the repo
          name (".../RepoFigure", ".../fairseqWe", ".../BioMistralet");
        - footnote ordinals glued on (".../ProSST38th" -> "ProSST");
        - deeper paths (/tree, /blob, /releases) reduced to the clonable
          ``owner/repo`` (also fixes git_clone, which otherwise derives the repo
          name from the LAST path segment).
        """
        if not raw:
            return None
        m = re.search(r"github\.com/(?P<path>[^\s)\]}>\"'<]+)", raw, re.IGNORECASE)
        if not m:
            return None
        path = m.group("path")

        # A second http(s):// inside the run means two URLs got concatenated
        # with no separator - keep only the first.
        path = re.split(r"https?://", path, maxsplit=1)[0]

        segments = [s for s in path.split("/") if s != ""]
        if len(segments) < 2:
            return None
        owner, repo = segments[0], segments[1]

        # Cut the repo at its first dot. Real repo names rarely contain a dot,
        # whereas PDF text constantly glues footnote / section / version markers
        # on via a dot ("scGPT.221", "satmae_pp.1", "hyena-dna.A.1"). Cutting
        # here also drops a legit trailing ".0" (e.g. Prithvi-EO-2.0 -> -2), an
        # acceptable trade for killing the far more common junk.
        repo = repo.split(".", 1)[0]

        punct_re = r"[.,;:!?)\]}>\"'`*©®™]+$"
        repo = re.sub(punct_re, "", repo)
        # Trim a glued footnote ordinal: ProSST38th -> ProSST, xlnet33rd -> xlnet.
        repo = re.sub(r"\d+(?:st|nd|rd|th)$", "", repo)
        # Repeatedly trim a glued trailing paper-structure word, re-stripping
        # punctuation each round (handles "...RepoFigure.").
        changed = True
        while changed and repo:
            changed = False
            for w in _TRAILING_JUNK_WORDS:
                if len(repo) > len(w) and repo.endswith(w):
                    repo = repo[: -len(w)]
                    changed = True
            new_repo = re.sub(punct_re, "", repo)
            if new_repo != repo:
                repo = new_repo
                changed = True
        repo = repo.rstrip("-_.")

        if not owner or not repo:
            return None
        repo_name_re = r"[A-Za-z0-9][A-Za-z0-9._-]*"
        if not re.fullmatch(repo_name_re, owner):
            return None
        if not re.fullmatch(repo_name_re, repo):
            return None
        # Reject gist-style hashes: github.com/<user>/<32-hex> is a gist, not a
        # clonable source repo, and tends to sit next to a code cue.
        if re.fullmatch(r"[0-9a-fA-F]{16,}", repo):
            return None
        # Reject github.com landing/feature pages that are not owner/repo.
        if owner.lower() in {
            "about", "search", "login", "join", "features", "marketplace",
            "sponsors", "topics", "collections", "settings", "notifications",
        }:
            return None
        return f"https://github.com/{owner}/{repo}"

    @staticmethod
    def _extract_first_github_link(text: str) -> Optional[str]:
        """Return the first cleaned github.com URL found in `text`, or None.

        Used by github_link_extract_with_llm to validate that the model returned
        a real github.com URL (and to normalize it through the same cleaner as
        the text path) instead of a hallucinated or trailing-junk string.
        """
        if not text:
            return None
        # PDF extraction often breaks URLs across lines; remove newlines so a
        # link split across two lines is still matched as one run.
        text = text.replace('\r\n', '').replace('\r', '').replace('\n', '')
        for m in GITHUB_URL_RE.finditer(text):
            url = RepoIngest._clean_github_url(m.group(0))
            if url is not None:
                return url
        return None

    @staticmethod
    def _score_candidate(text_nl: str, start: int, raw_path: str) -> float:
        """Score a single candidate link by how likely it is the paper's own repo.

        +2 if a code-availability cue ("code", "available at", "we release", ...)
        appears in the ~80 chars right before the URL; -5 if the raw path looks
        like a dataset / release asset / license / blob rather than a source repo.
        """
        score = 1.0
        low_path = raw_path.lower()
        if any(h in low_path for h in _NON_REPO_PATH_HINTS):
            score -= 5.0
        window = text_nl[max(0, start - 80):start].lower()
        if any(c in window for c in _CODE_CUE_WORDS):
            score += 2.0
        return score

    def _collect_candidates(self, text: str) -> List[Dict[str, Any]]:
        """Find every cleaned github candidate in `text` with a relevance score.

        Returns dicts {"url", "start", "score"}, deduped by canonical URL keeping
        the max score and earliest position.
        """
        if not text:
            return []
        text_nl = text.replace('\r\n', '').replace('\r', '').replace('\n', '')
        seen: Dict[str, int] = {}
        out: List[Dict[str, Any]] = []
        for m in GITHUB_URL_RE.finditer(text_nl):
            url = self._clean_github_url(m.group(0))
            if url is None:
                continue
            score = self._score_candidate(text_nl, m.start(), m.group("path"))
            owner_repo = url[len("https://github.com/"):].lower()
            if owner_repo in _REPO_DENYLIST:
                score -= 3.0
            if url not in seen:
                seen[url] = len(out)
                out.append({"url": url, "start": m.start(), "score": score})
            else:
                i = seen[url]
                out[i]["score"] = max(out[i]["score"], score)
                out[i]["start"] = min(out[i]["start"], m.start())
        return out

    # ------------------------------------------------------------------
    # GitHub existence verification (closes the trailing-footnote-glue gap)
    # ------------------------------------------------------------------

    @staticmethod
    def _git_ls_remote(url: str) -> Optional[bool]:
        """True if the repo resolves, False if it definitively does not, None if
        we could not determine it (git missing, no network, timeout)."""
        try:
            result = subprocess.run(
                ["git", "ls-remote", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            return result.returncode == 0
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def _repo_exists(self, url: str) -> Optional[bool]:
        """Cached _git_ls_remote. Set REPOINGEST_SKIP_VERIFY=1 to disable network
        verification entirely (returns None -> callers fall back to heuristics)."""
        if os.environ.get("REPOINGEST_SKIP_VERIFY") == "1":
            return None
        if url not in self._exists_cache:
            self._exists_cache[url] = self._git_ls_remote(url)
        return self._exists_cache[url]

    def _verify_and_trim(self, url: str) -> Tuple[str, str]:
        """Verify a candidate against GitHub, trimming trailing footnote/glue one
        char at a time. Returns (status, url):
        - ("verified", resolved_url): exists (possibly after trimming). The first
          hit when trimming fewest chars first is the LONGEST valid repo name, so
          a real name that happens to end in digits (DNABERT_2, ...2025) is kept
          intact while a glued footnote (CellPLM10 -> CellPLM) is shaved off.
        - ("none", url): definitively does not exist at any trim length.
        - ("unavailable", url): could not reach GitHub -> caller uses heuristics.
        """
        prefix = "https://github.com/"
        owner, _, repo = url[len(prefix):].partition("/")
        if not owner or not repo:
            return ("none", url)
        max_trim = min(len(repo) - 1, 12)
        for k in range(0, max_trim + 1):
            cand_repo = repo[: len(repo) - k].rstrip("-_.")
            if not cand_repo:
                break
            cand_url = f"{prefix}{owner}/{cand_repo}"
            exists = self._repo_exists(cand_url)
            if exists is None:
                return ("unavailable", url)
            if exists:
                return ("verified", cand_url)
        return ("none", url)

    def _verify_candidates(self, candidates: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        """Walk candidates best-score-first, verifying (and trimming) each.
        Returns the first that resolves on GitHub. ("unavailable", None) short-
        circuits to heuristics if the network/git is not usable."""
        for c in candidates[:6]:
            status, resolved = self._verify_and_trim(c["url"])
            if status == "unavailable":
                return ("unavailable", None)
            if status == "verified":
                return ("verified", resolved)
        return ("none", None)

    @staticmethod
    def _extract_abstract_text(pdf_content: Dict[str, Any]) -> str:
        """Best-effort extraction of the paper's abstract from the raw PDF text.

        Priority-1 search surface for github_link_extract_with_text: most
        authors advertise their own code repo in the abstract ("Code available
        at https://github.com/..."), so matching there is a much stronger
        signal than the first github.com URL in a paper that may just as well
        be a reference to someone else's code in related work or the
        bibliography. Returns "" when no abstract-looking region can be found
        - caller should then fall back to full-text search.

        Heuristics:
        1. Operate on page 1 (abstract is essentially always there). If page 1
           is empty/cover, try page 2.
        2. Anchor on a word-boundary "Abstract" header (case-insensitive),
           tolerating the "Abstract—", "Abstract:", "ABSTRACT." variants.
        3. Cut at the next section heading: Introduction / 1. Introduction /
           1 Introduction / I. Introduction / Keywords / Index Terms /
           CCS Concepts / Categories and Subject Descriptors.
        4. If no Abstract header is present, return the first 2000 chars of
           the effective first page as a weak "early part of paper" proxy.
        """
        pages = pdf_content.get("pages") or []
        if not pages:
            return ""
        first_page = (pages[0] or "").strip()
        if not first_page and len(pages) >= 2:
            first_page = (pages[1] or "").strip()
        if not first_page:
            return ""

        abstract_match = re.search(
            r"\bAbstract\b[\s:\.\-—]*",
            first_page,
            flags=re.IGNORECASE,
        )
        if abstract_match is None:
            # No explicit Abstract header - return the first ~2000 chars of
            # page 1 as a best-effort proxy. Authors frequently put the
            # "Code: https://github.com/..." badge near the title/authors
            # block on page 1 even without a labeled Abstract section.
            return first_page[:2000]

        start = abstract_match.end()
        tail = first_page[start:]

        end_patterns = (
            r"\n\s*1\s*[\.\)]?\s*Introduction\b",
            r"\n\s*I\s*[\.\)]\s*Introduction\b",
            r"\n\s*Introduction\b",
            r"\n\s*Keywords?\b[\s:\-]",
            r"\n\s*Index\s*Terms\b[\s:\-]",
            r"\n\s*CCS\s*Concepts\b",
            r"\n\s*Categories\s*and\s*Subject\s*Descriptors\b",
        )
        cut = len(tail)
        for pat in end_patterns:
            m = re.search(pat, tail, flags=re.IGNORECASE)
            if m and m.start() < cut:
                cut = m.start()
        return tail[:cut].strip()

    def github_link_extract_with_text(self, pdf_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Text-only (regex) github link extraction with abstract-first priority.

        Order:
        1. Try the paper's abstract (page 1 "Abstract" region). Authors almost
           always put their own "Code available at https://github.com/..."
           there, so a match here is high-confidence "the paper's own repo".
        2. Fall back to full-text scan. Anything we find outside the abstract
           might be a github.com URL cited in related work, a dataset source,
           or a dependency's homepage - weaker signal but better than nothing.
        """
        try:
            # Candidates from the abstract get a large bonus: a github URL there
            # is almost always the paper's own repo, not a cited dependency.
            abstract_text = self._extract_abstract_text(pdf_content)
            abstract_cands = self._collect_candidates(abstract_text)
            for c in abstract_cands:
                c["score"] += 3.0

            full_cands = self._collect_candidates(pdf_content.get("text", ""))

            merged: Dict[str, Dict[str, Any]] = {}
            for c in abstract_cands + full_cands:
                prev = merged.get(c["url"])
                if prev is None or c["score"] > prev["score"]:
                    merged[c["url"]] = c
            candidates = list(merged.values())

            if not candidates:
                return [{
                    "value": "Text search failed in the PDF file, please use the LLM to extract the Github Link",
                    "storage": "error",
                    "variable_name": "error_message",
                }]

            # Highest score wins; ties broken by earliest appearance.
            candidates.sort(key=lambda c: (-c["score"], c["start"]))
            best = candidates[0]

            # Every candidate looks like a dataset/release/baseline-dep link
            # (all scored <= 0). Don't commit to a wrong repo - defer to the LLM,
            # which can read the surrounding prose to disambiguate.
            if best["score"] <= 0:
                return [{
                    "value": (
                        "Text search only found dataset/dependency-like github "
                        "links (e.g. "
                        + ", ".join(c["url"] for c in candidates[:3])
                        + "), none clearly the paper's own repo. "
                        "Please use the LLM to extract the Github Link."
                    ),
                    "storage": "error",
                    "variable_name": "error_message",
                }]

            # Verify candidates against GitHub, trimming trailing footnote/glue.
            status, resolved = self._verify_candidates(candidates)
            if status == "verified":
                print(f"[RepoIngest] github_link (text, verified): {resolved}")
                return [{
                    "value": resolved,
                    "storage": "permanent",
                    "variable_name": "github_link",
                }]
            if status == "none":
                # Found github-looking links, but none resolve on GitHub even
                # after trimming - likely a mangled / wrong owner. Let the LLM,
                # which sees the surrounding prose, take a turn.
                return [{
                    "value": (
                        "Text search found github links that do not resolve on "
                        "GitHub (e.g. "
                        + ", ".join(c["url"] for c in candidates[:3])
                        + "). Please use the LLM to extract the Github Link."
                    ),
                    "storage": "error",
                    "variable_name": "error_message",
                }]

            # status == "unavailable": no network / git -> trust the heuristic best.
            print(f"[RepoIngest] github_link (text, unverified, score={best['score']:.0f}): {best['url']}")
            return [{
                "value": best["url"],
                "storage": "permanent",
                "variable_name": "github_link",
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "github_link_extract_with_text_error",
            }]

    def _assemble_pdfread_prompt(self, pdf_content: Dict[str, Any]) -> str:
        prompt_parts = []
        role_prompt = "\n".join(self.pdfread_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        # Surface the abstract as a dedicated section so the LLM can prioritize
        # any github.com URL that appears there (the paper's own code repo)
        # over URLs buried later in Related Work or References (which are
        # usually other papers' repos).
        abstract_text = self._extract_abstract_text(pdf_content)
        if abstract_text:
            prompt_parts.append("\n=== PAPER ABSTRACT (priority-1 search surface) ===")
            prompt_parts.append(abstract_text)

        # Rules section (only present if the prompt json defines one).
        rules = self.pdfread_prompt.get('rules')
        if rules:
            prompt_parts.append("\n=== RULES ===")
            prompt_parts.append("\n".join(rules))

        pdf_text_prompt = "\n".join(self.pdfread_prompt['pdf_file']).format(
            pdf_text=pdf_content['text']
        )
        prompt_parts.append("\n=== PDF TEXT (full paper, use only as fallback) ===")
        prompt_parts.append(pdf_text_prompt)

        output_format_prompt = "\n".join(self.pdfread_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def github_link_extract_with_llm(self, pdf_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            prompt = self._assemble_pdfread_prompt(pdf_content)
            response = json_query(prompt, "github_link_extract", self.backend)
            if isinstance(response, str):
                response = json.loads(response)
            raw_link = response.get('github_link') or ''

            # Run the LLM output through the same regex+dedup helper as the
            # text path so hallucinated URLs / multi-link / surrounding prose
            # all get caught locally.
            github_link = self._extract_first_github_link(raw_link)
            if github_link is None:
                return [{
                    "value": (
                        "LLM did not return a valid github.com URL "
                        f"(raw response: {raw_link!r}). Please stop the phase and report the error."
                    ),
                    "storage": "error",
                    "variable_name": "error_message",
                }]
            return [{
                "value": github_link,
                "storage": "permanent",
                "variable_name": "github_link",
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "github_link_extract_with_llm_error",
            }]

    # ------------------------------------------------------------------
    # git clone + README cleaning + task selection (formerly GithubClone)
    # ------------------------------------------------------------------

    def normalize_git_url(self, github_link: str) -> str:
        link = github_link.strip().rstrip("/")

        # case 1: already https
        if link.startswith("https://github.com/"):
            if not link.endswith(".git"):
                link += ".git"
            return link

        # case 2: ssh form -> convert to https
        if link.startswith("git@github.com:"):
            path_part = link[len("git@github.com:"):]
            if path_part.endswith(".git"):
                path_part = path_part[:-4]
            return f"https://github.com/{path_part}.git"

        # case 3: bare path (e.g. owner/repo)
        if "github.com" not in link:
            if link.endswith(".git"):
                link = link[:-4]
            return f"https://github.com/{link}.git"

        # fallback (other unusual cases)
        return link if link.endswith(".git") else link + ".git"


    def git_clone(self, github_link: str) -> List[Dict[str, Any]]:
        repo_root: Optional[Path] = None
        try:
            download_setting = global_config.get("download_setting") or {}
            repo_parent = download_setting.get("repo_root")

            repo_name = Path(github_link.rstrip("/").split("/")[-1].replace(".git", ""))
            if not repo_parent or str(repo_parent).strip().lower() == "none":
                parent_dir = Path("/tmp")
            else:
                parent_dir = Path(repo_parent)

            parent_dir.mkdir(parents=True, exist_ok=True)
            repo_root = parent_dir / repo_name

            if repo_root.exists():
                shutil.rmtree(repo_root)

            git_link = self.normalize_git_url(github_link)

            linux_command(f"git clone {git_link} {repo_root}")

            cleaned_readmes_path = self.save_clean_readmes(repo_root)

            return [
                {
                    "value": repo_root,
                    "storage": "permanent",
                    "variable_name": "repo_root",
                },
                {
                    "value": cleaned_readmes_path,
                    "storage": "permanent",
                    "variable_name": "cleaned_readmes_path",
                },
            ]
        except Exception as e:
            if repo_root is not None:
                try:
                    linux_command(f"rm -rf {repo_root}")
                except Exception:
                    pass
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "git_clone_error",
            }]

    def is_readme_file(self, path: Path) -> bool:
        name = path.name.lower()
        return name == "readme" or name.startswith("readme")

    def clean_readme_by_keywords(self, readme_path) -> str:
        if not os.path.exists(readme_path):
            raise FileNotFoundError(readme_path)

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Keep preprocessing lightweight so the LLM can still see inference
        # examples, demo commands, download snippets, and long shell blocks.
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)

        lines = text.splitlines()
        remove_keywords = {
            "reference",
            "references",
            "citation",
            "bibtex",
            "acknowledgement",
            "acknowledgements",
        }

        cleaned_lines = []
        skip = False
        current_level = None

        for line in lines:
            header_match = re.match(r"(#+)\s*(.+)", line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).lower()

                if any(k in title for k in remove_keywords):
                    skip = True
                    current_level = level
                    continue

                if skip and level <= current_level:
                    skip = False
                    current_level = None

            if not skip:
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line for line in text.splitlines() if line.strip())
        return text.strip()

    def number_readme_lines(self, clean_readme_text: str) -> str:
        lines = clean_readme_text.splitlines()
        width = max(4, len(str(len(lines))))

        numbered_lines = []
        for idx, line in enumerate(lines, start=1):
            numbered_lines.append(f"[{str(idx).zfill(width)}] {line}")

        return "\n".join(numbered_lines)

    def select_readme_lines(self, numbered_readme_text: str, keep_ranges: List[Tuple[int, int]]) -> str:
        keep_lines = set()
        for start, end in keep_ranges:
            keep_lines.update(range(start, end + 1))

        output_lines = []

        for line in numbered_readme_text.splitlines():
            match = re.match(r"\[(\d+)\]\s?(.*)", line)
            if not match:
                continue

            line_no = int(match.group(1))
            content = match.group(2)

            if line_no in keep_lines:
                output_lines.append(content)

        return "\n".join(output_lines).strip()

    def normalize_keep_ranges(self, keep_ranges) -> List[Tuple[int, int]]:
        if isinstance(keep_ranges, str):
            nums = [int(x) for x in re.findall(r"\d+", keep_ranges)]
            return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]

        normalized_ranges: List[Tuple[int, int]] = []
        for item in keep_ranges or []:
            if isinstance(item, dict):
                start = item.get("start_line_number")
                end = item.get("end_line_number")
            else:
                start = getattr(item, "start_line_number", None)
                end = getattr(item, "end_line_number", None)
                if start is None or end is None:
                    try:
                        start, end = item
                    except Exception:
                        continue

            if start is None or end is None:
                continue
            normalized_ranges.append((int(start), int(end)))

        return normalized_ranges

    def _assemble_readme_prompt(self, path_to_readme_file: str, readme_text: str) -> str:
        prompt_parts = []
        role_prompt = "\n".join(self.readme_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        path_to_readme_file_prompt = "\n".join(self.readme_prompt["path_to_readme_file"]).format(
            path_to_readme_file=path_to_readme_file
        )
        prompt_parts.append("\n=== PATH TO README FILE ===")
        prompt_parts.append(path_to_readme_file_prompt)

        readme_text_prompt = "\n".join(self.readme_prompt["readme_text"]).format(
            readme_text=readme_text
        )
        prompt_parts.append("\n=== README TEXT ===")
        prompt_parts.append(readme_text_prompt)

        rules_prompt = "\n".join(self.readme_prompt["rules"])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        output_format_prompt = json.dumps(self.readme_prompt["output_format"])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def inference_clean_readme(self, path_to_readme_file: str, basic_clean_readme_text: str):
        numbered_readme_text = self.number_readme_lines(basic_clean_readme_text)
        prompt = self._assemble_readme_prompt(path_to_readme_file, numbered_readme_text)
        response = json_query(prompt, "clean_readme", self.backend)
        if isinstance(response, str):
            response = json.loads(response)
        keep_ranges = self.normalize_keep_ranges(response["keep_ranges"])
        return self.select_readme_lines(numbered_readme_text, keep_ranges)

    def clean_readme(self, readme_path) -> str:
        basic_clean_readme_text = self.clean_readme_by_keywords(readme_path)
        clean_readme_text = self.inference_clean_readme(readme_path, basic_clean_readme_text)
        return clean_readme_text

    def get_all_clean_readmes(self, repo_root: Path) -> List[Dict[str, str]]:
        repo_root = Path(repo_root)
        readmes: List[Dict[str, str]] = []

        ignore_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            "site-packages",
        }
        not_inference_key_substrings = ("pretrain", "pre-training", "finetun", "fine-tun")

        for file in repo_root.rglob("*"):
            if not file.is_file():
                continue

            parts_lower = [p.lower() for p in file.parts]

            if any(p in ignore_dirs for p in parts_lower):
                continue

            if not self.is_readme_file(file):
                continue

            if any(any(k in p for k in not_inference_key_substrings) for p in parts_lower):
                continue

            readmes.append({
                "path": str(file.relative_to(repo_root)),
                "content": self.clean_readme(file),
            })

        return readmes

    def save_clean_readmes(self, repo_root: Path) -> Path:
        repo_root = Path(repo_root)
        readmes = self.get_all_clean_readmes(repo_root)

        workdir = repo_root / ".autodeploy"
        workdir.mkdir(parents=True, exist_ok=True)

        out_path = workdir / "readmes.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(readmes, f, ensure_ascii=False, indent=2)

        return out_path

    def task_selection(self, repo_root: Path, cleaned_readmes_path: Path) -> List[Dict[str, Any]]:
        try:
            task_selection_tool = TaskSelectionTool(repo_root, cleaned_readmes_path)
            task = task_selection_tool.task_detection()
            return [{
                "value": task,
                "storage": "permanent",
                "variable_name": "task",
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "task_selection_error",
            }]
