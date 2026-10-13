import json
import re
from typing import List, Dict, Any, Optional
from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.llm.openrouter import OpenRouter

class StructuredExtractor:
    """Extracts structured architectural data from AUTOSAR HLD documents using fast local Ollama or cloud models."""

    def __init__(
        self,
        llm_model: str = "qwen2.5-coder:1.5b",
        openrouter_api_key: Optional[str] = None
    ):
        if llm_model.startswith("openrouter/"):
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key required for OpenRouter models.")
            llm = OpenRouter(model=llm_model, api_key=openrouter_api_key)
        else:
            llm = Ollama(model=llm_model)

        self.assistant = Assistant(
            llm=llm,
            description="You are an AUTOSAR Architect specializing in High-Level Design analysis.",
            instructions=[
                "Extract components, interfaces, ports, and signals from the provided text.",
                "Format the output as a valid JSON object.",
                "Structure: {'components': [{'name': '...', 'type': '...', 'ports': [{'name': '...', 'direction': '...', 'interface': '...'}]}]}",
                "If no entities are found, return an empty structure: {'components': []}.",
                "Only output valid JSON enclosed between braces. No conversational text."
            ]
        )

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from a block of text."""
        response = ""
        for delta in self.assistant.run(f"Extract AUTOSAR entities from this text: {text}"):
            response += delta

        # Parse JSON extraction from response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        return {"components": []}

    def merge_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results into one unique set."""
        merged = {"components": []}
        comp_map = {}

        for ext in extractions:
            for comp in ext.get("components", []):
                name = comp.get("name")
                if not name:
                    continue
                if name not in comp_map:
                    comp_map[name] = comp
                else:
                    # Merge ports
                    existing_ports = {p.get("name") for p in comp_map[name].get("ports", [])}
                    for port in comp.get("ports", []):
                        if port.get("name") not in existing_ports:
                            comp_map[name]["ports"].append(port)

        merged["components"] = list(comp_map.values())
        return merged
