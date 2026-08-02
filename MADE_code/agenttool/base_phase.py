from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Callable

@dataclass
class PhaseResult:
    status: str  # "ok" | "need_retry" | "blocked" | "fatal"
    observation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    parallel_phases: Optional[List[str]] = None # support parallel execution

class BasePhase(ABC):
    name: str
    description: str
    goal: str
    tools: List[Callable]

    @abstractmethod
    def boundary_tools(self, tool_name: str) -> bool:
        pass
    
    @abstractmethod
    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        pass