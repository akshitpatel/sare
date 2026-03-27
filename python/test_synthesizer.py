import sys
import os
import logging
from pathlib import Path

# Setup Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# Basic logging
logging.basicConfig(level=logging.INFO)

from sare.transfer.synthesizer import TransformSynthesizer
from sare.interface.llm_bridge import llm_available

def main():
    if not llm_available():
        print("LLM is not available. Please configure API key or start Ollama.")
        return

    print("Testing LLM-guided Transform Synthesis...")
    synthesizer = TransformSynthesizer()
    
    # Try to synthesize for boolean logic
    # missing_roles = ["identity", "annihilation", "involution"]
    missing_roles = ["identity"]
    new_transforms = synthesizer.synthesize_for_domain("logic", missing_roles)
    
    print(f"Synthesized {len(new_transforms)} transforms.")
    for t in new_transforms:
        print(f" - {t.name}: {t.operator_labels} Element: {t.element_label} Action: {t.rewrite_action}")

if __name__ == "__main__":
    main()
