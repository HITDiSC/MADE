from abc import ABC
from typing import List
import os
import yaml
import json
from backend.query import json_query

allowed_parallel_phases = ["weightresolve", "dockersetup"]

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")

class ParallelDiscriminator:
    def __init__(self, backend: str):
        self.backend = backend
        system_prompt_filepath = os.path.join(file_path, "parallel_discriminator_prompt.json")
        with open(system_prompt_filepath, "r", encoding="utf-8") as f:
            self.system_prompt = json.load(f)

    def assemble_prompt(self, pm_board: List[dict]) -> str:
        prompt_parts = []
        role_prompt = "\n".join(self.system_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)
        # NOTE: previously this was `"\n".join(json.dumps(...))` which iterates
        # over the JSON string character-by-character and inserts a newline
        # between each char, producing garbled output for the LLM.
        phase_description_prompt = json.dumps(
            self.system_prompt['phase_description'], ensure_ascii=False, indent=2
        )
        prompt_parts.append("\n=== PHASE DESCRIPTION ===")
        prompt_parts.append(phase_description_prompt)
        past_phase_execution_history_prompt = "\n".join(self.system_prompt['past_phase_execution_history']).format(past_phase_execution_history=pm_board)
        prompt_parts.append("\n=== PAST PHASE EXECUTION HISTORY ===")
        prompt_parts.append(past_phase_execution_history_prompt)
        allowed_parallel_phases_prompt = "\n".join(self.system_prompt['allowed_parallel_phases']).format(allowed_parallel_phases=allowed_parallel_phases)
        prompt_parts.append("\n=== ALLOWED PARALLEL PHASES ===")
        prompt_parts.append(allowed_parallel_phases_prompt)
        output_format_prompt = json.dumps(self.system_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def run(self, pm_board: List[dict]) -> List[str]:
        prompt = self.assemble_prompt(pm_board)
        response = json_query(prompt, "pd", self.backend)
        print("pd_response", response)
        return response['phases']