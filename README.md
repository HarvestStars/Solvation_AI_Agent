# Solvation Labs — AI-assisted Screening Prototype

A **Streamlit product prototype** for the YES!Delft pitch. A customer brief drives a real, multi-step workflow: LLM brief parsing → web literature search → candidate generation → RDKit descriptors → LLM property estimation → weighted ranking, with a downloadable traceable record.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Put your key in `.env` at the repo root (already git-ignored):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini   # optional, this is the default
```

One run takes ~20–30 s and makes 4 OpenAI calls (one with the `web_search` tool).

## Workflow (`workflow.py`)

| Step | What actually runs |
|---|---|
| 1. Parse brief | LLM structured output → voltage, anode, safety, extra constraints, search query |
| 2. Literature agent | LLM + OpenAI `web_search`; cited URLs are captured as sources |
| 3. Candidate generation | LLM proposes formulations (components + SMILES + amounts), citing step-2 URLs |
| 3b. SMILES repair loop | RDKit must parse each SMILES and its formula must match the formula the LLM expects from the component *name* alone; failures (with error + previous attempts) go back to the LLM, up to 3 rounds |
| 4. Descriptors | RDKit: MolWt, LogP, TPSA, F/P content; flags invalid SMILES. **Placeholder for DFT** |
| 5. Property estimation | LLM scores stability / safety / interface in [0,1] with uncertainty. **Not a trained model** |
| 6. Rank | Transparent weighted sum; changing weights re-ranks instantly without re-running steps 1–5 |
| 7. Experimental validation | Pending (partner lab) |

## Suggested walkthrough
1. Point to the prototype notice and the model name.
2. Configure 4.6 V cathode, Silicon anode, non-flammable; optionally add free-text constraints.
3. Click **Run screening** and narrate each step as its status completes.
4. Show the shortlist, then drag a weight slider to show transparent re-ranking.
5. Inspect a candidate: formulation, SMILES descriptors, literature URLs, estimate reasoning.
6. Download the JSON record (includes response IDs, token usage, timings) to illustrate traceability.

## Important limitations
- **No DFT, no trained ML model, no experimental validation.** Property scores are LLM estimates and must be presented as hypotheses for expert review, not research results.
- LLM-generated SMILES are often invalid; step 3b repairs most of them. The identity check only compares molecular formulas, so isomers are not caught, and components whose name the LLM marks "unknown" get only a syntax check. Anything left unresolved is listed in the UI.
- Literature summaries depend on web search quality; always verify cited sources.
- Next real-platform steps: replace step 4 with real DFT/xTB jobs, step 5 with a model trained on own/licensed data with calibrated uncertainty, and link partner-lab measurement IDs.

## Structure
- `app.py` — Streamlit UI.
- `workflow.py` — workflow steps and schemas.
- `requirements.txt` — dependencies.
