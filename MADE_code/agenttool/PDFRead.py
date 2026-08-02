import json
import os
import re
from abc import ABC, abstractmethod
import yaml
import pypdf
from pathlib import Path
from typing import Union, List, Dict, Callable, Optional
from backend.query import query, json_query
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

GITHUB_URL_RE = re.compile(
    r"""(?xi)
    https?://
    (?:www\.)?
    github\.com
    /[^\s)\]}>,"'.]+
    """
)

class PDFRead(BasePhase):
    name: str = "PDFRead"
    description: str = "Read the PDF file and extract the Github Link."
    goal: str = "get the Github Link"
    tools_schemas: List[Dict[str, any]] = [
        {"name": "read_pdf", "description": "Read the PDF file", "args": {"pdf_path": Path}},
        {"name": "github_link_extract_with_text", "description": "(preferred use when available)Extract the Github Link from the PDF file with text", "args": {"pdf_content": Dict[str, any]}},
        {"name": "github_link_extract_with_llm", "description": "Extract the Github Link from the PDF file with the help of LLM", "args": {"pdf_content": Dict[str, any]}},
    ]
    allowed_parallel_phases: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.tools = {
        "read_pdf": self.read_pdf,
        "github_link_extract_with_text": self.github_link_extract_with_text,
        "github_link_extract_with_llm": self.github_link_extract_with_llm,
    }
        prompt_path = os.path.join(project_path, "prompts")
        system_prompt_filepath = os.path.join(prompt_path, "pdfread_prompt.json")
        with open(system_prompt_filepath, "r", encoding="utf-8") as f:
            self.system_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        boundary_tools_list = ["github_link_extract_with_text", "github_link_extract_with_llm"]
        if tool_name in boundary_tools_list:
            return True
        else:
            return False

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "read_pdf": {"pdf_path": Path},
            "github_link_extract_with_text": {"pdf_content": Dict[str, any]},
            "github_link_extract_with_llm": {"pdf_content": Dict[str, any]},
        }
        return tool_arguments_dict[tool_name]

    def read_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        Read the PDF file and extract the text content

        Args:
            pdf_path: PDF file path
            page_numbers: the page numbers to read(from 0), if None then read all pages

        Returns:
            a dictionary containing the text content and metadata
            {
                'text': str,  # all text content
                'pages': List[str],  # the text content of each page
                'num_pages': int,  # the number of pages
            }
        """
        pdf_content = {
            'text': '',
            'pages': [],
            'num_pages': 0
        }
        try:
            # read the PDF file
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                pdf_content['num_pages'] = len(pdf_reader.pages)

                # extract the text from each page
                for page_num in range(pdf_content['num_pages']):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    pdf_content['pages'].append(page_text)
                    pdf_content['text'] += page_text + '\n'
            return [{
                "value": pdf_content, 
                "storage": "temporary", 
                "variable_name": "pdf_content"
            }]
        except FileNotFoundError:
            return [{
                "value": f"PDF file not found: {pdf_path}",
                "storage": "error", 
                "variable_name": "error_message"
            }]
        except Exception as e:
            return [{
                "value": f"Error reading PDF file: {str(e)}",
                "storage": "error", 
                "variable_name": "error_message"
            }]

    @staticmethod
    def _extract_first_github_link(text: str) -> Optional[str]:
        """Run GITHUB_URL_RE over `text`, dedupe in original order, return the first match.

        Shared by github_link_extract_with_text and github_link_extract_with_llm so
        the regex / dedup behavior is identical regardless of which path produced
        the candidate. The LLM path uses this to validate that the model actually
        returned a real github.com URL instead of a hallucinated string.
        """
        if not text:
            return None
        matches = list(dict.fromkeys(GITHUB_URL_RE.findall(text)))
        return matches[0] if matches else None

    def github_link_extract_with_text(self, pdf_content: Dict[str, any]) -> str:
        try:
            github_link = self._extract_first_github_link(pdf_content.get('text', ''))
            if github_link is not None:
                print("github_link", github_link)
                return [{
                    "value": github_link,
                    "storage": "permanent",
                    "variable_name": "github_link"
                }]
            return [{
                "value": "Text search failed in the PDF file, please use the LLM to extract the Github Link",
                "storage": "error",
                "variable_name": "error_message"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "github_link_extract_with_text_error"
            }]

    def assemble_prompt(self, pdf_content: Dict[str, any]) -> str:
        prompt_parts = []
        role_prompt = "\n".join(self.system_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)
        pdf_text_prompt = "\n".join(self.system_prompt['pdf_file']).format(
            pdf_text=pdf_content['text']
        )
        prompt_parts.append("\n=== PDF TEXT ===")
        prompt_parts.append(pdf_text_prompt)

        output_format_prompt = "\n".join(self.system_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        prompt = "\n".join(prompt_parts)
        return prompt

    def github_link_extract_with_llm(self, pdf_content: Dict[str, any]) -> str:
        try:
            prompt = self.assemble_prompt(pdf_content)
            response = json_query(prompt, "github_link_extract", self.backend)
            if isinstance(response, str):
                response = json.loads(response)
            raw_link = response.get('github_link') or ''

            # Run the LLM output through the same regex+dedup helper as the text
            # path. This catches the cases where the LLM hallucinates a non-github
            # URL, returns surrounding prose, or returns multiple links.
            github_link = self._extract_first_github_link(raw_link)
            if github_link is None:
                return [{
                    "value": (
                        "LLM did not return a valid github.com URL "
                        f"(raw response: {raw_link!r}). Please stop the phase and report the error."
                    ),
                    "storage": "error",
                    "variable_name": "error_message"
                }]
            return [{
                "value": github_link,
                "storage": "permanent",
                "variable_name": "github_link"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "github_link_extract_with_llm_error"
            }]