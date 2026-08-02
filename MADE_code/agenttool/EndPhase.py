import json
import os
from pathlib import Path
from typing import List, Dict, Any

import yaml

from backend.query import query
from agenttool.base_phase import BasePhase

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")


class EndPhase(BasePhase):
    name: str = "EndPhase"
    description: str = "Generate final delivery artifacts and end the deployment process."
    goal: str = "prepare final delivery content and finish deployment"
    tools_schemas: List[Dict[str, any]] = [
        {"name": "generate_api_documentation", "description": "Generate final API documentation from the actual deployed service code and FastAPI app code. Use service.py and app.py as the primary evidence for request and response behavior.", "args": {"repo_root": Path, "task": str, "service_pipeline_path": Path, "fastapiapp_dir": Path, "cleaned_readmes_path": Path}},
        {"name": "end_deployment", "description": "Mark deployment as complete after final delivery artifacts such as API documentation have been prepared.", "args": {"repo_root": Path, "api_pid": str, "api_log_path": Path, "api_documentation_path": Path}},
    ]
    allowed_parallel_phases: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.tools = {
            "generate_api_documentation": self.generate_api_documentation,
            "end_deployment": self.end_deployment,
        }

        prompt_path = os.path.join(project_path, "prompts")
        api_doc_prompt_filepath = os.path.join(prompt_path, "endphase_api_doc_prompt.json")
        with open(api_doc_prompt_filepath, "r", encoding="utf-8") as f:
            self.api_doc_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        return tool_name == "end_deployment"

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "generate_api_documentation": {"repo_root": Path, "task": str, "service_pipeline_path": Path, "fastapiapp_dir": Path, "cleaned_readmes_path": Path},
            "end_deployment": {"repo_root": Path, "api_pid": str, "api_log_path": Path, "api_documentation_path": Path},
        }
        return tool_arguments_dict[tool_name]

    def assemble_api_doc_prompt(self, task: str, service_pipeline_path: Path, fastapiapp_dir: Path, cleaned_readmes_path: Path) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.api_doc_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.api_doc_prompt["instructions"])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        rules_prompt = "\n".join(self.api_doc_prompt["rules"])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        task_prompt = "\n".join(self.api_doc_prompt["task"]).format(task=task)
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append(task_prompt)

        with open(cleaned_readmes_path, "r", encoding="utf-8", errors="ignore") as f:
            cleaned_readme = f.read()
        readme_prompt = "\n".join(self.api_doc_prompt["readme_content"]).format(
            readme_content=cleaned_readme
        )
        prompt_parts.append("\n=== README CONTENT ===")
        prompt_parts.append(readme_prompt)

        with open(service_pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
            service_code = f.read()
        service_prompt = "\n".join(self.api_doc_prompt["service_code"]).format(
            service_pipeline_path=service_pipeline_path,
            service_code=service_code,
        )
        prompt_parts.append("\n=== SERVICE CODE ===")
        prompt_parts.append(service_prompt)

        app_path = Path(fastapiapp_dir) / "app.py"
        with open(app_path, "r", encoding="utf-8", errors="ignore") as f:
            app_code = f.read()
        app_prompt = "\n".join(self.api_doc_prompt["fastapi_app_code"]).format(
            app_path=app_path,
            app_code=app_code,
        )
        prompt_parts.append("\n=== FASTAPI APP CODE ===")
        prompt_parts.append(app_prompt)

        output_format_prompt = "\n".join(self.api_doc_prompt["output_format"])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)

    def generate_api_documentation(self, repo_root: Path, task: str, service_pipeline_path: Path, fastapiapp_dir: Path, cleaned_readmes_path: Path):
        try:
            prompt = self.assemble_api_doc_prompt(
                task=task,
                service_pipeline_path=service_pipeline_path,
                fastapiapp_dir=fastapiapp_dir,
                cleaned_readmes_path=cleaned_readmes_path,
            )
            api_doc_markdown = query(prompt, self.backend)

            output_dir = Path(repo_root) / ".autodeploy"
            output_dir.mkdir(parents=True, exist_ok=True)
            api_doc_path = output_dir / "API_DOCUMENTATION.md"
            api_doc_path.write_text(api_doc_markdown, encoding="utf-8")

            return [{
                "value": str(api_doc_path),
                "storage": "permanent",
                "variable_name": "api_documentation_path"
            },
            {
                "value": "API documentation generated successfully.",
                "storage": "temporary",
                "variable_name": "api_documentation_generated"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "api_documentation_error"
            }]

    def end_deployment(self, repo_root: Path, api_pid: str, api_log_path: Path, api_documentation_path: Path):
        return [{
            "value": str(repo_root),
            "storage": "permanent",
            "variable_name": "repo_root"
        },
        {
            "value": api_pid,
            "storage": "permanent",
            "variable_name": "api_pid"
        },
        {
            "value": api_log_path,
            "storage": "permanent",
            "variable_name": "api_log_path"
        },
        {
            "value": api_documentation_path,
            "storage": "permanent",
            "variable_name": "api_documentation_path"
        }]
