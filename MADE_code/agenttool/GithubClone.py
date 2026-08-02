import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from agenttool.FileTracker import FileTracker
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


class GithubClone(BasePhase):
    name: str = "GithubClone"
    description: str = "Clone the Github repository and find the task user want to enable."
    goal: str = "get the Github repository and find the task user want to enable"
    tools_schemas: List[Dict[str, any]] = [
        {"name": "git_clone", "description": "Clone the Github repository", "args": {"github_link": str}},
        {"name": "task_selection", "description": "Find the task user want to enable", "args": {"repo_root": Path, "cleaned_readmes_path": Path}},
    ]
    allowed_parallel_phases: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.tools = {
            "git_clone": self.git_clone,
            "task_selection": self.task_selection,
        }
        prompt_path = os.path.join(project_path, "prompts")
        readme_prompt_filepath = os.path.join(prompt_path, "readme_prompt.json")
        with open(readme_prompt_filepath, "r", encoding="utf-8") as f:
            self.readme_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        return tool_name in {"task_selection"}

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "git_clone": {"github_link": str},
            "task_selection": {"repo_root": Path, "cleaned_readmes_path": Path},
        }
        return tool_arguments_dict[tool_name]

    def git_clone(self, github_link: str) -> Dict[str, any]:
        try:
            with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
                global_config = yaml.safe_load(f)

            download_setting = global_config.get("download_setting")
            repo_parent = download_setting.get("repo_root")
            repo = Path(github_link.rstrip("/").split("/")[-1].replace(".git", ""))

            if not repo_parent or str(repo_parent).strip().lower() == "none":
                repo = Path(github_link.rstrip("/").split("/")[-1].replace(".git", ""))
                repo_root = Path("/tmp") / repo
            else:
                repo_root = Path(repo_parent) / repo

            if os.path.exists(repo_root):
                linux_command(f"rm -rf {repo_root}")

            git_link = github_link if github_link.rstrip("/").endswith(".git") else github_link.rstrip("/") + ".git"
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

    def clean_readme_by_keywords(self, readme_path: str) -> str:
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

    def assemble_prompt(self, path_to_readme_file: str, readme_text: str):
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
        prompt = self.assemble_prompt(path_to_readme_file, numbered_readme_text)
        response = json_query(prompt, "clean_readme", self.backend)
        if isinstance(response, str):
            response = json.loads(response)
        keep_ranges = self.normalize_keep_ranges(response["keep_ranges"])
        return self.select_readme_lines(numbered_readme_text, keep_ranges)

    def clean_readme(self, readme_path: str | Path) -> str:
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

    def task_selection(self, repo_root: Path, cleaned_readmes_path: Path):
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
