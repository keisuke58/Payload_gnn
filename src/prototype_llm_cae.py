# -*- coding: utf-8 -*-
"""
LLM × CAE Prototype: Natural Language → Abaqus Python Script Generation

自然言語の指示から Abaqus Python スクリプトを自動生成するプロトタイプ。
Google Gemini API / Anthropic Claude API 対応。

Usage:
    python src/prototype_llm_cae.py "片持ち梁、長さ1000mm、幅100mm、高さ50mm、SUS304、先端集中荷重100N"
    python src/prototype_llm_cae.py --interactive
    python src/prototype_llm_cae.py --backend claude "..."  # Claude API を使用
"""

import os
import sys
import ast
import json
import argparse
import subprocess
import re
from pathlib import Path

# Backend selection: prefer Gemini, fallback to Claude
_HAS_GEMINI = False
_HAS_ANTHROPIC = False

try:
    from google import genai
    _HAS_GEMINI = True
except ImportError:
    pass

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    pass

if not _HAS_GEMINI and not _HAS_ANTHROPIC:
    print("ERROR: No LLM backend found.")
    print("  Install one of: pip install google-genai / pip install anthropic")
    sys.exit(1)

# ==============================================================================
# Material Database (common engineering materials)
# ==============================================================================
MATERIAL_DB = {
    "SUS304": {"E": 193000.0, "nu": 0.3, "rho": 7.93e-9, "alpha": 1.73e-5,
               "description": "Stainless Steel 304 (tonne/mm/s/MPa)"},
    "A6061-T6": {"E": 68900.0, "nu": 0.33, "rho": 2.70e-9, "alpha": 2.36e-5,
                 "description": "Aluminum 6061-T6"},
    "Ti-6Al-4V": {"E": 113800.0, "nu": 0.342, "rho": 4.43e-9, "alpha": 8.6e-6,
                  "description": "Titanium alloy"},
    "S45C": {"E": 205000.0, "nu": 0.3, "rho": 7.85e-9, "alpha": 1.12e-5,
             "description": "Carbon Steel S45C"},
    "CFRP_UD": {"E1": 160000.0, "E2": 10000.0, "nu12": 0.3,
                "G12": 5000.0, "G13": 5000.0, "G23": 3000.0,
                "rho": 1.6e-9, "description": "CFRP Unidirectional (T1000G class)"},
}

# ==============================================================================
# System Prompt for Claude API
# ==============================================================================
SYSTEM_PROMPT = """You are an expert Abaqus Python scripting assistant.
You generate complete, runnable Abaqus Python scripts for `abaqus cae noGUI=script.py`.

## CRITICAL RULES:
1. Always start with:
   ```python
   from abaqus import *
   from abaqusConstants import *
   from caeModules import *
   from mesh import ElemType
   from driverUtils import executeOnCaeStartup
   executeOnCaeStartup()
   ```
2. Unit system: mm / tonne / s → Force: N, Stress: MPa, Density: tonne/mm³
3. NEVER use GUI operations (session.viewports, session.printToFile, etc.)
4. NEVER use interactive commands
5. Always create a Job and submit it with `job.submit()` and `job.waitForCompletion()`
6. Use `Mdb()` to create a fresh model database
7. For partitioning and meshing, use appropriate element types:
   - 3D solid: C3D8R (hex reduced integration) or C3D10 (tet quadratic)
   - 2D plane stress: CPS4R or CPS8R
   - Shell: S4R or S8R
   - Beam: B31 or B32
8. Add proper boundary conditions and loads
9. Include print statements for progress feedback
10. Handle the job name via a variable at the top of the script

## MATERIAL DATABASE (tonne/mm/s/MPa unit system):
""" + json.dumps(MATERIAL_DB, indent=2, ensure_ascii=False) + """

## OUTPUT FORMAT:
Return ONLY the Python script wrapped in ```python ... ``` markers.
No explanations before or after the code block.
The script must be complete and self-contained.
"""

# ==============================================================================
# Few-shot example
# ==============================================================================
FEW_SHOT_USER = "片持ち梁、長さ500mm、幅50mm、高さ30mm、SUS304、先端集中荷重50N、静解析"

FEW_SHOT_ASSISTANT = """```python
# -*- coding: utf-8 -*-
# Cantilever beam: L=500mm, W=50mm, H=30mm, SUS304, Tip load 50N
from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import ElemType
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()

import os
import sys

JOB_NAME = 'Cantilever_SUS304'

# --- Material Properties (SUS304, tonne/mm/s/MPa) ---
E = 193000.0   # MPa
NU = 0.3
RHO = 7.93e-9  # tonne/mm^3

# --- Geometry ---
L = 500.0  # mm
W = 50.0   # mm
H = 30.0   # mm

# --- Load ---
TIP_LOAD = -50.0  # N (downward)

print("Creating model...")
mdb = Mdb()
model = mdb.models['Model-1']

# --- Part ---
s = model.ConstrainedSketch(name='beam_sketch', sheetSize=1000.0)
s.rectangle(point1=(0.0, 0.0), point2=(L, H))
part = model.Part(name='Beam', dimensionality=THREE_D, type=DEFORMABLE_BODY)
part.BaseSolidExtrude(sketch=s, depth=W)
print("Part created.")

# --- Material ---
mat = model.Material(name='SUS304')
mat.Elastic(table=((E, NU),))
mat.Density(table=((RHO,),))

# --- Section ---
model.HomogeneousSolidSection(name='BeamSection', material='SUS304', thickness=None)
region = part.Set(cells=part.cells, name='AllCells')
part.SectionAssignment(region=region, sectionName='BeamSection')
print("Material and section assigned.")

# --- Assembly ---
assembly = model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)
instance = assembly.Instance(name='Beam-1', part=part, dependent=ON)
print("Assembly created.")

# --- Step ---
model.StaticStep(name='Load', previous='Initial', nlgeom=OFF,
                 description='Static load step')
print("Step created.")

# --- Boundary Conditions (Fixed end at x=0) ---
fixed_faces = instance.faces.getByBoundingBox(xMin=-0.1, xMax=0.1,
                                               yMin=-0.1, yMax=H+0.1,
                                               zMin=-0.1, zMax=W+0.1)
fixed_region = assembly.Set(faces=fixed_faces, name='FixedEnd')
model.EncastreBC(name='Fixed', createStepName='Initial', region=fixed_region)
print("BC applied.")

# --- Load (Tip load at x=L) ---
tip_faces = instance.faces.getByBoundingBox(xMin=L-0.1, xMax=L+0.1,
                                             yMin=-0.1, yMax=H+0.1,
                                             zMin=-0.1, zMax=W+0.1)
tip_region = assembly.Surface(side1Faces=tip_faces, name='TipSurface')
model.Pressure(name='TipLoad', createStepName='Load',
               region=tip_region, magnitude=TIP_LOAD/(H*W),
               distributionType=UNIFORM)
print("Load applied.")

# --- Mesh ---
part.seedPart(size=10.0, deviationFactor=0.1, minSizeFactor=0.1)
part.setMeshControls(regions=part.cells, elemShape=HEX, technique=STRUCTURED)
part.setElementType(regions=(part.cells,),
                    elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD),
                               ElemType(elemCode=C3D6, elemLibrary=STANDARD),
                               ElemType(elemCode=C3D4, elemLibrary=STANDARD)))
part.generateMesh()
print("Mesh generated: %d elements" % len(part.elements))

# --- Job ---
job = mdb.Job(name=JOB_NAME, model='Model-1', numCpus=4, numDomains=4)
print("Submitting job: %s" % JOB_NAME)
job.submit()
job.waitForCompletion()
print("Job completed: %s" % JOB_NAME)
```"""


# ==============================================================================
# Main Functions
# ==============================================================================

def generate_abaqus_script(prompt: str, model: str = None,
                           backend: str = "gemini") -> str:
    """Generate Abaqus Python script from natural language prompt.

    Args:
        prompt: Natural language description of the FEM model
        model: Model name override (default: auto-select per backend)
        backend: "gemini" or "claude"
    """
    if backend == "gemini" and _HAS_GEMINI:
        return _generate_gemini(prompt, model or "gemini-2.5-flash")
    elif backend == "claude" and _HAS_ANTHROPIC:
        return _generate_claude(prompt, model or "claude-sonnet-4-20250514")
    elif _HAS_GEMINI:
        return _generate_gemini(prompt, model or "gemini-2.5-flash")
    elif _HAS_ANTHROPIC:
        return _generate_claude(prompt, model or "claude-sonnet-4-20250514")
    else:
        raise RuntimeError("No LLM backend available")


def _generate_gemini(prompt: str, model: str) -> str:
    """Generate using Google Gemini API."""
    client = genai.Client()  # Uses GOOGLE_API_KEY env var

    full_prompt = (
        SYSTEM_PROMPT + "\n\n"
        "## Example:\n"
        f"User: {FEW_SHOT_USER}\n"
        f"Assistant:\n{FEW_SHOT_ASSISTANT}\n\n"
        f"## Now generate for this request:\n{prompt}"
    )

    print(f"Calling Gemini API ({model})...")
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.2,
        ),
    )

    raw_text = response.text
    return extract_code(raw_text)


def _generate_claude(prompt: str, model: str) -> str:
    """Generate using Anthropic Claude API."""
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    messages = [
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": prompt},
    ]

    print(f"Calling Claude API ({model})...")
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    raw_text = response.content[0].text
    return extract_code(raw_text)


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block, return raw text (might already be pure code)
    return text.strip()


def validate_script(code: str) -> list:
    """Basic validation of generated Abaqus script."""
    issues = []

    # Check for required imports
    if "from abaqus import" not in code:
        issues.append("Missing 'from abaqus import *'")
    if "executeOnCaeStartup" not in code:
        issues.append("Missing executeOnCaeStartup()")

    # Check for forbidden GUI operations
    gui_patterns = ["session.viewports", "session.printToFile",
                    "session.openOdb", "getDisplay"]
    for pat in gui_patterns:
        if pat in code:
            issues.append(f"Contains GUI operation: {pat}")

    # Syntax check (skip abaqus-specific imports)
    test_code = code
    for imp in ["from abaqus import", "from abaqusConstants import",
                "from caeModules import", "from mesh import",
                "from driverUtils import", "from odbAccess import"]:
        test_code = test_code.replace(imp, "# " + imp)
    # Also comment out executeOnCaeStartup() call
    test_code = test_code.replace("executeOnCaeStartup()", "# executeOnCaeStartup()")

    try:
        ast.parse(test_code)
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")

    return issues


def save_script(code: str, output_path: str) -> str:
    """Save generated script to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Script saved to: {output_path}")
    return output_path


def run_abaqus(script_path: str, work_dir: str = "abaqus_work") -> int:
    """Run generated script with Abaqus (optional)."""
    abaqus_cmd = "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/home/nishioka/libfake_x11.so"

    abs_script = os.path.abspath(script_path)
    os.makedirs(work_dir, exist_ok=True)

    cmd = [abaqus_cmd, "cae", f"noGUI={abs_script}"]
    print(f"Running: {' '.join(cmd)}")
    print(f"Working dir: {work_dir}")

    result = subprocess.run(cmd, cwd=work_dir, env=env,
                            capture_output=True, text=True)
    print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    return result.returncode


def interactive_mode():
    """Interactive mode: repeatedly ask user for input."""
    print("=" * 60)
    print("LLM × CAE: Natural Language → Abaqus Script Generator")
    print("=" * 60)
    print("Type your FEM model description in natural language.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Generate
        code = generate_abaqus_script(prompt)

        # Validate
        issues = validate_script(code)
        if issues:
            print("\n⚠ Validation warnings:")
            for issue in issues:
                print(f"  - {issue}")

        # Save
        output_name = "generated_model.py"
        output_path = save_script(code, f"abaqus_work/{output_name}")

        print(f"\nGenerated script ({len(code.splitlines())} lines):")
        print("-" * 40)
        print(code[:3000])
        if len(code) > 3000:
            print(f"... ({len(code) - 3000} more characters)")
        print("-" * 40)

        # Ask to run
        run = input("\nRun with Abaqus? [y/N]: ").strip().lower()
        if run == "y":
            rc = run_abaqus(output_path)
            print(f"Abaqus exit code: {rc}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="LLM × CAE: Natural Language → Abaqus Python Script")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Natural language description of the FEM model")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--output", "-o", default="abaqus_work/generated_model.py",
                        help="Output script path")
    parser.add_argument("--run", action="store_true",
                        help="Run generated script with Abaqus")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: auto per backend)")
    parser.add_argument("--backend", default="gemini",
                        choices=["gemini", "claude"],
                        help="LLM backend (default: gemini)")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.prompt:
        parser.print_help()
        print("\nExample:")
        print('  python src/prototype_llm_cae.py "片持ち梁、長さ1000mm、SUS304、先端荷重100N"')
        return

    # Generate
    code = generate_abaqus_script(args.prompt, model=args.model,
                                  backend=args.backend)

    # Validate
    issues = validate_script(code)
    if issues:
        print("\nValidation warnings:")
        for issue in issues:
            print(f"  - {issue}")

    # Save
    save_script(code, args.output)

    # Print
    print(f"\nGenerated script ({len(code.splitlines())} lines):")
    print(code)

    # Run
    if args.run:
        rc = run_abaqus(args.output)
        sys.exit(rc)


if __name__ == "__main__":
    main()
