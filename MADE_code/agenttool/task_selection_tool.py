import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
import yaml
from pathlib import Path
import string
from typing import Union, List, Dict
from agenttool.tool import *
import ast
from backend.query import json_query

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


class TaskSelectionTool(ABC):
    def __init__(self, repo_root: Path, cleaned_readmes_path: Path) -> None:
        super().__init__()
        prompt_path = os.path.join(project_path, "prompts")
        self.repo_root = repo_root
        self.backend = "gr"
        self.cleaned_readmes_path = cleaned_readmes_path
        task_prompt_filepath = os.path.join(prompt_path, "task_prompt.json")
        with open(task_prompt_filepath, "r", encoding="utf-8") as f:
            self.task_prompt = json.load(f)


    def readme_process(self):
        with open(self.cleaned_readmes_path, "r", encoding="utf-8") as f:
            readmes = json.load(f)
        return readmes
    
    def assemble_task_selection_prompt(self, readme_text: str):
        prompt_parts = []
        role_prompt = "\n".join(self.task_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        readme_text_prompt = "\n".join(self.task_prompt['readme_text']).format(
            readme_text=readme_text
        )
        prompt_parts.append("\n=== README TEXT ===")
        prompt_parts.append(readme_text_prompt)

        rules_prompt = "\n".join(self.task_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        output_format_prompt = json.dumps(self.task_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def task_selection(self):
        # readme_process returns the parsed JSON list (List[Dict[str, str]]).
        # Serialize it back to a readable JSON string before feeding it into
        # the prompt template; otherwise .format() interpolates a Python repr
        # (single quotes, escaped newlines) which the LLM cannot parse cleanly.
        readmes = self.readme_process()
        readme_text = json.dumps(readmes, ensure_ascii=False, indent=2)
        prompt = self.assemble_task_selection_prompt(readme_text)
        response = json_query(prompt, "task_selection", self.backend)
        if isinstance(response, str):
            response = json.loads(response)
        tasks = response["tasks"]

        if not tasks:
            print("[task_selection] LLM returned empty task list, defaulting to inference")
            return {"task_id": "default", "task_name": "inference", "task_description": "inference"}

        # auto-select when there is only one task; no user decision needed
        if len(tasks) == 1:
            print(f"Auto-selected task: {tasks[0]}")
            return tasks[0]
        
        letters = string.ascii_uppercase
        if len(tasks) > len(letters):
            raise ValueError("Too many tasks for letter selection")

        option_map = {letters[i]: tasks[i] for i in range(len(tasks))}

        for k, v in option_map.items():
            print(f"{k} {v}")

        while True:
            user_input = input(
                f"Please enter the task you want to enable ({'/'.join(option_map.keys())}): "
            ).strip().upper()

            if user_input in option_map:
                return option_map[user_input]
            else:
                print("Unsupported task, please try again.\n")

    def task_detection(self):
        """
        find the existing task based on the readme text.
        return:
        {
        "user_task": the task user choose,
        }
        """
        task = self.task_selection()
        return  task