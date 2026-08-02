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

class ExecutionMaster:
    def __init__(self, backend: str):
        self.backend = backend
        system_prompt_filepath = os.path.join(file_path, "execution_master_prompt.json")
        with open(system_prompt_filepath, "r", encoding="utf-8") as f:
            self.system_prompt = json.load(f)
        instruction_prompt_filepath = os.path.join(file_path, "em_instruction_prompt.json")
        with open(instruction_prompt_filepath, "r", encoding="utf-8") as f:
            self.instruction_prompt = json.load(f)

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


        output_format = self.system_prompt['output_format']
        # convert each key-value pair of the dict into a string
        output_format_prompt = json.dumps(output_format)
        # append the result to prompt_parts
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts) 

    def run(self, phase_description, goal, board, tool_list, arguments_board):
        prompt = self.init_prompt() + "\n\n===PHASE DESCRIPTION===\n" + phase_description + "\n\n===PHASE GOAL===\n" + goal + "\n\n===HISTORY===\n" + str(board) + "\n\n=== TOOL LIST ===\n"
        for tool in tool_list:
            prompt = prompt + str(tool) + "\n"
        prompt = prompt + "\n\n=== ARGUMENTS ===\n" + str(arguments_board)
        response = json_query(prompt, "em", self.backend)
        return response

    def init_instruction_prompt(self):
        prompt_parts = []
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append("\n".join(self.instruction_prompt["role"]))
        prompt_parts.append("\n=== SYSTEM OVERVIEW ===")
        prompt_parts.append("\n".join(self.instruction_prompt["system_overview"]))
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append("\n".join(self.instruction_prompt["rules"]))
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(json.dumps(self.instruction_prompt["output_format"]))
        return "\n".join(prompt_parts)

    def respond_to_instruction(self, instruction, phase_description, goal, board, tool_list):
        """EM's accept/negotiate/reject response to a PM instruction (need_retry).

        Returns the parsed em_instruction response: {reason, decision, message}.
        """
        prompt = (
            self.init_instruction_prompt()
            + "\n\n===PHASE DESCRIPTION===\n" + phase_description
            + "\n\n===PHASE GOAL===\n" + goal
            + "\n\n===PM INSTRUCTION (decide whether to accept / negotiate / reject)===\n" + str(instruction)
            + "\n\n===HISTORY (your first-hand observations)===\n" + str(board)
            + "\n\n=== TOOL LIST ===\n"
        )
        for tool in tool_list:
            prompt = prompt + str(tool) + "\n"
        response = json_query(prompt, "em_instruction", self.backend)
        return response