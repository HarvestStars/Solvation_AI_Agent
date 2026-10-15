"""Solvation Labs screening workflow.

Steps that really execute:
  1. parse_brief          - LLM turns the customer brief into structured constraints
  2. literature_search    - LLM + OpenAI web_search tool, returns cited URLs
  3. generate_candidates  - LLM proposes electrolyte formulations grounded in step 2
     repair_smiles        - RDKit + LLM validate-and-regenerate loop for component SMILES
  4. compute_descriptors  - RDKit molecular descriptors (cheap proxy; NOT DFT)
  5. estimate_properties  - LLM estimates stability/safety/interface scores (NOT a trained model)
  6. rank                 - transparent weighted ranking
Experimental validation is out of scope and stays "pending".
"""
from __future__ import annotations

import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------- structured output schemas ----------
class Brief(BaseModel):
    cathode_voltage_v: float
    anode: str
    non_flammable_required: bool
    temperature_range_c: str
    other_constraints: list[str]
    search_query: str


class Component(BaseModel):
    name: str
    smiles: str
    role: str  # solvent / salt / additive / diluent
    amount: str


class Candidate(BaseModel):
    id: str
    name: str
    family: str
    components: list[Component]
    rationale: str
    supporting_urls: list[str]


class CandidateList(BaseModel):
    candidates: list[Candidate]


class PropertyEstimate(BaseModel):
    candidate_id: str
    stability: float
    safety: float
    interface: float
    uncertainty: str  # low / medium / high
    reasoning: str


class EstimateList(BaseModel):
    estimates: list[PropertyEstimate]


# ---------- helpers ----------
def _parse(system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, dict]:
    t0 = time.time()
    r = client().responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
    )
    return r.output_parsed, _meta(r, t0)


def _meta(r, t0: float) -> dict:
    u = r.usage
    return {"model": r.model, "response_id": r.id, "seconds": round(time.time() - t0, 1),
            "input_tokens": u.input_tokens if u else None, "output_tokens": u.output_tokens if u else None}


# ---------- workflow steps ----------
def parse_brief(form: dict, free_text: str) -> tuple[Brief, dict]:
    system = ("You are a battery R&D requirements analyst. Convert the customer brief into structured "
              "constraints. Form fields are authoritative; use free text for extra constraints. "
              "search_query should be a concise literature search query for suitable electrolytes.")
    user = f"Form fields: {form}\nFree-text requirements: {free_text or '(none)'}"
    return _parse(system, user, Brief)


def literature_search(brief: Brief) -> tuple[dict, dict]:
    t0 = time.time()
    prompt = (f"Search the scientific literature for liquid electrolyte formulations relevant to: "
              f"{brief.search_query}. Constraints: cathode {brief.cathode_voltage_v} V, anode {brief.anode}, "
              f"non-flammable required: {brief.non_flammable_required}, other: {brief.other_constraints}. "
              "Summarise 5-8 key findings (solvents, salts, additives, reported performance) as bullet points, "
              "citing a source URL for each.")
    r = client().responses.create(model=MODEL, tools=[{"type": "web_search"}], input=prompt)
    sources = {}
    for item in r.output:
        if item.type != "message":
            continue
        for part in item.content:
            for a in getattr(part, "annotations", None) or []:
                if a.type == "url_citation":
                    sources[a.url] = a.title
    return {"summary": r.output_text, "sources": [{"url": u, "title": t} for u, t in sources.items()]}, _meta(r, t0)


def generate_candidates(brief: Brief, literature: dict, n: int) -> tuple[list[Candidate], dict]:
    system = ("You are an electrolyte formulation scientist. Propose candidate liquid electrolyte formulations. "
              "Every component needs a valid, canonical SMILES (salts as ionic SMILES, e.g. LiPF6 as "
              "'[Li+].F[P-](F)(F)(F)(F)F'). Ground proposals in the literature findings and only cite URLs "
              "from the provided source list. Use ids CAND-1, CAND-2, ...")
    user = (f"Brief: {brief.model_dump_json()}\n\nLiterature findings:\n{literature['summary']}\n\n"
            f"Allowed source URLs: {[s['url'] for s in literature['sources']]}\n\nPropose {n} diverse candidates.")
    out, meta = _parse(system, user, CandidateList)
    return out.candidates, meta


class ExpectedFormula(BaseModel):
    candidate_id: str
    component_name: str
    formula: str  # Hill-style, whole salt incl. counter-ions; "unknown" if ambiguous


class ExpectedFormulaList(BaseModel):
    items: list[ExpectedFormula]


class SmilesFix(BaseModel):
    candidate_id: str
    component_name: str
    smiles: str
    note: str


class SmilesFixList(BaseModel):
    fixes: list[SmilesFix]


def smiles_check(smiles: str) -> tuple[str | None, str | None]:
    """Return (error, molecular_formula). error is None when RDKit accepts the SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "SMILES syntax error: RDKit could not parse the string", None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:  # valence / aromaticity / kekulization errors
        return f"RDKit sanitization error: {e}", None
    return None, rdMolDescriptors.CalcMolFormula(mol)


def _element_counts(formula: str) -> dict[str, int]:
    """'C2F6LiNO4S2' / 'LiC2F6NO4S2' / 'C3H6O3' -> element counts; charges and dots are ignored."""
    counts: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[el] = counts.get(el, 0) + int(n or 1)
    return counts


def repair_smiles(candidates: list[Candidate], max_rounds: int = 3) -> tuple[dict, dict]:
    """Validate-and-repair loop.

    Checks are deterministic: RDKit must parse the SMILES, and its molecular formula must match the
    formula the LLM expects from the component *name* alone (asked once, without showing the SMILES).
    Failing components are fed back to the LLM with the error and previous attempts, up to max_rounds.
    """
    t0 = time.time()
    comps = {(c.id, comp.name): comp for c in candidates for comp in c.components}
    attempts = {k: [comp.smiles] for k, comp in comps.items()}
    history, calls = [], []

    # Expected formulas from names only, so the reference is independent of the SMILES being checked.
    out, m = _parse(
        "You are a chemist. For each electrolyte component, give the molecular formula of the NAMED chemical, "
        "using its name only. For salts give the whole neutral salt including counter-ions (e.g. LiTFSI -> "
        "C2F6LiNO4S2, LiPF6 -> F6LiP). Use 'unknown' if the name is ambiguous, proprietary or a mixture. "
        "Keep candidate_id and component_name unchanged.",
        str([{"candidate_id": cid, "component_name": name, "role": c.role} for (cid, name), c in comps.items()]),
        ExpectedFormulaList)
    calls.append(m)
    expected = {(i.candidate_id, i.component_name): i.formula for i in out.items}

    def find_issues() -> dict[tuple[str, str], str]:
        issues = {}
        for key, comp in comps.items():
            err, formula = smiles_check(comp.smiles)
            exp = expected.get(key, "unknown")
            if err:
                issues[key] = err
            elif exp.lower() != "unknown" and _element_counts(formula) != _element_counts(exp):
                issues[key] = f"Formula mismatch: SMILES gives {formula}, but {key[1]} should be {exp}"
        return issues

    issues = find_issues()
    for rnd in range(1, max_rounds + 1):
        if not issues:
            break
        payload = [{"candidate_id": cid, "component_name": name, "role": comps[(cid, name)].role,
                    "expected_formula": expected.get((cid, name), "unknown"),
                    "current_smiles": comps[(cid, name)].smiles, "error": err,
                    "previous_attempts": attempts[(cid, name)]} for (cid, name), err in issues.items()]
        out, m = _parse(
            "You fix SMILES strings for electrolyte components. For each item return a corrected SMILES that "
            "exactly represents the named chemical, matches expected_formula and passes RDKit sanitization. "
            "Write salts as dot-separated ions with explicit charges, e.g. LiTFSI: "
            "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F. Do not repeat a previous failed attempt. "
            "Keep candidate_id and component_name unchanged.",
            str(payload), SmilesFixList)
        calls.append(m)
        for fix in out.fixes:
            key = (fix.candidate_id, fix.component_name)
            if key not in issues:
                continue
            old = comps[key].smiles
            comps[key].smiles = fix.smiles
            attempts[key].append(fix.smiles)
            history.append({"round": rnd, "candidate_id": key[0], "component": key[1], "problem": issues[key],
                            "old_smiles": old, "new_smiles": fix.smiles, "note": fix.note})
        issues = find_issues()
        for h in history:
            if h["round"] == rnd:
                h["resolved"] = (h["candidate_id"], h["component"]) not in issues

    unresolved = [{"candidate_id": cid, "component": name, "problem": err} for (cid, name), err in issues.items()]
    meta = {"model": MODEL, "seconds": round(time.time() - t0, 1), "llm_calls": len(calls),
            "repairs": len(history), "unresolved": len(unresolved),
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls)}
    return {"history": history, "unresolved": unresolved, "expected_formulas":
            [{"candidate_id": k[0], "component": k[1], "formula": v} for k, v in expected.items()]}, meta


def compute_descriptors(candidates: list[Candidate]) -> tuple[dict, dict]:
    """RDKit descriptors per candidate. A cheap computational proxy standing in for future DFT jobs."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    t0 = time.time()
    result = {}
    for c in candidates:
        comps, invalid = [], []
        for comp in c.components:
            mol = Chem.MolFromSmiles(comp.smiles)
            if mol is None:
                invalid.append(comp.name)
                continue
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            comps.append({
                "name": comp.name, "role": comp.role,
                "mol_wt": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
                "f_atoms": atoms.count("F"), "p_atoms": atoms.count("P"),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
        heavy = sum(x["heavy_atoms"] for x in comps) or 1
        result[c.id] = {
            "components": comps,
            "invalid_smiles": invalid,
            "fluorine_fraction": round(sum(x["f_atoms"] for x in comps) / heavy, 3),
            "contains_phosphorus": any(x["p_atoms"] for x in comps),
        }
    return result, {"tool": "rdkit", "seconds": round(time.time() - t0, 2)}


def estimate_properties(brief: Brief, candidates: list[Candidate], descriptors: dict,
                        literature: dict) -> tuple[dict[str, PropertyEstimate], dict]:
    system = ("You are an electrolyte screening assistant. For each candidate, estimate three scores in [0, 1]: "
              "oxidative/cycling stability at the target voltage, safety (non-flammability, thermal), and "
              "interface compatibility with the anode. Use the RDKit descriptors and literature findings. "
              "Be calibrated and state uncertainty (low/medium/high). These are rough expert-style estimates.")
    user = (f"Brief: {brief.model_dump_json()}\n\nLiterature:\n{literature['summary']}\n\n"
            f"Candidates: {[c.model_dump() for c in candidates]}\n\nDescriptors: {descriptors}")
    out, meta = _parse(system, user, EstimateList)
    clamp = lambda v: max(0.0, min(1.0, v))
    est = {}
    for e in out.estimates:
        e.stability, e.safety, e.interface = clamp(e.stability), clamp(e.safety), clamp(e.interface)
        est[e.candidate_id] = e
    return est, meta


def rank(candidates: list[Candidate], estimates: dict[str, PropertyEstimate], weights: dict,
         prioritize_safety: bool) -> list[dict]:
    denom = sum(weights.values()) or 1
    rows = []
    for c in candidates:
        e = estimates.get(c.id)
        if e is None:
            continue
        f = {"stability": e.stability, "safety": e.safety, "interface": e.interface}
        score = sum(weights[k] * f[k] for k in weights) / denom
        if prioritize_safety:
            score = 0.85 * score + 0.15 * e.safety
        rows.append({"Candidate": c.id, "Name": c.name, "Family": c.family, "Score": round(score, 3),
                     "Stability": round(e.stability, 2), "Safety": round(e.safety, 2),
                     "Interface": round(e.interface, 2), "Uncertainty": e.uncertainty})
    return sorted(rows, key=lambda r: (-r["Score"], r["Candidate"]))
