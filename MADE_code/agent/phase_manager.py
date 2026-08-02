import json
import os
import re
from abc import ABC, abstractmethod
import yaml
from pathlib import Path
from typing import Union, List, Dict, Optional

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

class PhaseManager:
    def __init__(self, backend: str):
        self.backend = backend
        system_prompt_filepath = os.path.join(file_path, "phase_manager_prompt.json")
        with open(system_prompt_filepath, "r", encoding="utf-8") as f:
            self.system_prompt = json.load(f)
        gap_prompt_filepath = os.path.join(file_path, "phase_manager_gap_prompt.json")
        with open(gap_prompt_filepath, "r", encoding="utf-8") as f:
            self.gap_prompt = json.load(f)


    def init_prompt(self):
        prompt_parts = []
        role_prompt = "\n".join(self.system_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)
        system_overview_prompt = "\n".join(self.system_prompt['system_overview'])
        prompt_parts.append("\n=== SYSTEM OVERVIEW ===")
        prompt_parts.append(system_overview_prompt)
        rules_prompt = "\n".join(self.system_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)
        available_phases_prompt = json.dumps(self.system_prompt['available_phases'])
        prompt_parts.append("\n=== AVAILABLE PHASES ===")
        prompt_parts.append(available_phases_prompt)
        allowed_parallel_phases_prompt = json.dumps(self.system_prompt['allowed_parallel_phases'])
        prompt_parts.append("\n=== ALLOWED PARALLEL PHASES ===")
        prompt_parts.append(allowed_parallel_phases_prompt)
        output_format_prompt = json.dumps(self.system_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)


    def run(self, board):
        prompt = self.init_prompt() + "\n" + "=== EXCUTION HISTORY ===" + "\n" + str(board)
        response = json_query(prompt, "pm", self.backend)
        return response

    def init_gap_prompt(self):
        prompt_parts = []
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append("\n".join(self.gap_prompt["role"]))
        prompt_parts.append("\n=== SYSTEM OVERVIEW ===")
        prompt_parts.append("\n".join(self.gap_prompt["system_overview"]))
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append("\n".join(self.gap_prompt["rules"]))
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(json.dumps(self.gap_prompt["output_format"]))
        return "\n".join(prompt_parts)

    def gap_check(self, phase_description, goal, board, deployment_state=None):
        """Mid-phase information-gap check: should PM nudge a stuck EM?

        Returns the parsed pm_gap response: {reason, intervene, instruction}.
        """
        prompt = (
            self.init_gap_prompt()
            + "\n\n===PHASE DESCRIPTION===\n" + str(phase_description)
            + "\n\n===PHASE GOAL===\n" + str(goal)
        )
        if deployment_state is not None:
            prompt += "\n\n===DEPLOYMENT STATE===\n" + str(deployment_state)
        prompt += "\n\n===EM EXECUTION HISTORY (recent turns; EM may be stuck)===\n" + str(board)
        response = json_query(prompt, "pm_gap", self.backend)
        return response