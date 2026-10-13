from typing import List, Dict, Any

class ConsistencyValidator:
    """Validates consistency and identifies gaps in extracted AUTOSAR models."""

    @staticmethod
    def validate_model(model: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify missing ports, unmatched interfaces, or incomplete components."""
        findings = []
        components = model.get("components", [])

        for comp in components:
            name = comp.get("name", "Unknown")
            ports = comp.get("ports", [])

            if not ports:
                findings.append({
                    "severity": "Warning",
                    "component": name,
                    "issue": "No ports defined for this component."
                })

            for port in ports:
                if not port.get("interface"):
                    findings.append({
                        "severity": "Error",
                        "component": name,
                        "issue": f"Port '{port.get('name')}' missing interface definition."
                    })

        return findings

    @staticmethod
    def compare_revisions(old_model: Dict[str, Any], new_model: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two document revisions and return the delta."""
        diff = {
            "added_components": [],
            "removed_components": [],
            "modified_components": []
        }

        old_comps = {c["name"]: c for c in old_model.get("components", []) if "name" in c}
        new_comps = {c["name"]: c for c in new_model.get("components", []) if "name" in c}

        for name in new_comps:
            if name not in old_comps:
                diff["added_components"].append(name)
            elif new_comps[name] != old_comps[name]:
                diff["modified_components"].append(name)

        for name in old_comps:
            if name not in new_comps:
                diff["removed_components"].append(name)

        return diff
