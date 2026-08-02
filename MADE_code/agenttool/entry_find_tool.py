import json
import os
import re
from abc import ABC, abstractmethod
import yaml
import ast
from pathlib import Path
from typing import Union, List, Dict, Callable
from backend.query import json_query
from agenttool.base_phase import BasePhase
from agenttool.tool import linux_command, locate_local_path, build_tree

# Load global config
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")


class EntryFindTool(ABC):
    def __init__(self, repo_root: Path, cleaned_readmes_path: Path, task: str) -> None:
        super().__init__()
        self.repo_root = Path(repo_root)
        self.task = task
        self.backend = "gr"
        self.cleaned_readmes_path = Path(cleaned_readmes_path)
        
        prompt_path = os.path.join(project_path, "prompts")

        entry_prompt_filepath = os.path.join(prompt_path, "entry_prompt.json")
        with open(entry_prompt_filepath, "r", encoding="utf-8") as f:
            self.entry_prompt = json.load(f)


    def has_main(self, py_file: Path) -> bool:
        """Check if file contains if __main__ block."""
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except:
            return False

        return bool(re.search(
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
            text
        ))


    def has_weights_load(self, py_file: Path) -> bool:
        """Check if file loads weights file."""
        try:
            tree = ast.parse(py_file.read_text(encoding='utf-8'))
        except Exception:
            return False

        load_keywords = [
            "torch.load",
            "load_state_dict",
            "load_checkpoint",
            "from_pretrained",
            "load_weights",
            "resume_from_checkpoint",
            "checkpoint"
        ]

        class LoadWeightVisitor(ast.NodeVisitor):
            def __init__(self):
                self.found = False

            def visit_Call(self, node):
                # case 1: a torch.load(...) call
                if isinstance(node.func, ast.Attribute):
                    full_name = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
                    if full_name in ["torch.load", "accelerator.load_state"]:
                        self.found = True
                
                # case 2: a model.load_state_dict(...) call
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "load_state_dict":
                        self.found = True
                
                # case 3: a call to a function whose name contains "checkpoint"
                if isinstance(node.func, ast.Name):
                    if "checkpoint" in node.func.id.lower():
                        self.found = True

                self.generic_visit(node)

            def visit_Assign(self, node):
                # check the config['model_path'] pattern
                if isinstance(node.value, ast.Subscript):
                    try:
                        key = node.value.slice.value.s
                        if key.lower() in ["model_path", "ckpt", "checkpoint"]:
                            self.found = True
                    except Exception:
                        pass
                self.generic_visit(node)

        visitor = LoadWeightVisitor()
        visitor.visit(tree)
        return visitor.found

    def mentioned_in_readme(self, readmes, py_file: Path):
        filename = py_file.name.lower()   # e.g. run_demo.py
        stem = py_file.stem.lower()       # e.g. run_demo

        # if the stem is too short/generic (e.g. train/run/main), use only the filename to avoid false matches
        weak_stem = len(stem) <= 3 or stem in {"main", "run", "train", "test", "demo"}

        for readme in readmes:
            text = readme["content"].lower()

            # 1) exact: full filename
            if filename in text:
                return True

            # 2) relaxed: stem as a standalone word (so "training" does not match "train")
            if not weak_stem and re.search(rf"\b{re.escape(stem)}\b", text):
                return True

            # 3) common command invocations
            if re.search(rf"(python|python3)\s+(\S*/)?{re.escape(filename)}\b", text):
                return True

        return False

    def collect_candidates(self, task: str):
        with open(self.cleaned_readmes_path, "r", encoding="utf-8") as f:
            readmes = json.load(f)
        CANDIDATE_KEYWORDS = ["main", "run", "demo", "example", "app", "server", "web", "cli", "visualize", "infer", "inference", "predict", "eval", "test"]

        keywords = set(CANDIDATE_KEYWORDS)

        candidates = []

        for file in list(self.repo_root.rglob("*.py")) + list(self.repo_root.rglob("*.ipynb")):
            name = file.name.lower()
            path = file.relative_to(self.repo_root)

            if any(k in name for k in keywords) or self.has_main(file):
                candidates.append({
                    "path": str(path),
                    "has_main": self.has_main(file),
                    "has_weights_load": self.has_weights_load(file),
                    "mentioned_in_readme": self.mentioned_in_readme(readmes, file),
                    "filename": file.name,
                    "file_type":file.suffix[1:] if file.suffix else "unknown"
                })

        return sorted(candidates, key=lambda x: x["path"])

    def assemble_prompt(self):
        prompt_parts = []

        role_prompt = "\n".join(self.entry_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        task_prompt = "\n".join(self.entry_prompt['task']).format(
            task=self.task
        )
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append(task_prompt)

        file_tree = build_tree(str(self.repo_root), max_depth=3)
        file_tree_prompt = "\n".join(self.entry_prompt['file_tree']).format(
            file_tree=json.dumps(file_tree, ensure_ascii=False, indent=2)
        )
        prompt_parts.append("\n=== FILE TREE ===")
        prompt_parts.append(file_tree_prompt)

        file_list_prompt = "\n".join(self.entry_prompt['file_list']).format(
            file_list=json.dumps(self.collect_candidates(self.task))
        )
        prompt_parts.append("\n=== CANDIDATE LIST (reference) ===")
        prompt_parts.append(file_list_prompt)

        with open(self.cleaned_readmes_path, "r", encoding="utf-8") as f:
            cleaned_readme_content = f.read()
        readme_content_prompt = "\n".join(self.entry_prompt['readme_content']).format(
            readme_content=cleaned_readme_content
        )
        prompt_parts.append("\n=== README CONTENT ===")
        prompt_parts.append(readme_content_prompt)

        rules_prompt = "\n".join(self.entry_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        output_format_prompt = json.dumps(self.entry_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)


    def find_entry(self):
        prompt = self.assemble_prompt()
        response = json_query(prompt, "entry_find", self.backend)
        if isinstance(response, str):
            response = json.loads(response)
        entry_point = str(response.get('entry_point') or "").strip()
        backup_paths = [
            str(p).strip()
            for p in (response.get('backup_path') or [])
            if str(p).strip() and str(p).strip() != entry_point
        ]
        return entry_point, backup_paths