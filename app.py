"""Solvation Labs AI-assisted electrolyte screening prototype."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import workflow as wf

st.set_page_config(page_title="Solvation Labs | Screening Demo", page_icon="🧪", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.8rem;max-width:1200px}</style>""", unsafe_allow_html=True)
st.title("🧪 Solvation Labs | AI-assisted Electrolyte Screening")
st.warning("**PROTOTYPE** — Literature search, candidate generation and property estimates are produced by an LLM "
           f"(`{wf.MODEL}`) and RDKit descriptors. No DFT is run and no trained property model is used. "
           "Outputs are hypotheses for expert review, not validated results.")
st.markdown("**Product story:** Customer brief → literature & simulation tasks → candidate ranking → experimental validation")

with st.sidebar:
    st.header("Customer brief")
    voltage = st.select_slider("Cathode operating voltage (V)", options=[4.2, 4.3, 4.4, 4.5, 4.6, 4.7], value=4.6)
    anode = st.selectbox("Anode chemistry", ["Silicon", "Lithium metal", "Graphite", "Hard carbon (Na-ion)"])
    prioritize_safety = st.checkbox("Prioritize non-flammable candidates", value=True)
    free_text = st.text_area("Additional requirements (free text)",
                             placeholder="e.g. operate down to -20 °C, avoid LiPF6, low cost")
    n_candidates = st.slider("Number of candidates", 3, 8, 5)
    st.divider()
    wt_stability = st.slider("Weight: stability", 0, 100, 40, 5)
    wt_safety = st.slider("Weight: safety", 0, 100, 35, 5)
    wt_interface = st.slider("Weight: interface compatibility", 0, 100, 25, 5)
    run = st.button("▶ Run screening", type="primary", width="stretch")

form = {"cathode_voltage_v": voltage, "anode": anode, "prioritize_non_flammable": prioritize_safety}
weights = {"stability": wt_stability, "safety": wt_safety, "interface": wt_interface}
inputs_sig = hashlib.sha256(json.dumps({"form": form, "text": free_text, "n": n_candidates},
                                       sort_keys=True).encode()).hexdigest()[:10]


def run_pipeline() -> dict:
    log = []

    def step(label, fn, *args):
        with st.status(label, expanded=False) as s:
            try:
                out, meta = fn(*args)
            except Exception as e:  # surface API / parsing errors in the UI
                s.update(label=f"{label} — failed", state="error")
                st.exception(e)
                st.stop()
            log.append({"step": label, **meta})
            s.update(label=f"{label} — done ({meta.get('seconds', '?')}s)", state="complete")
            return out

    brief = step("1 · Parse customer brief", wf.parse_brief, form, free_text)
    lit = step("2 · Literature agent (web search)", wf.literature_search, brief)
    cands = step("3 · Candidate generation", wf.generate_candidates, brief, lit, n_candidates)
    repair = step("3b · SMILES check & repair loop (RDKit + LLM)", wf.repair_smiles, cands)
    desc = step("4 · Molecular descriptors (RDKit, DFT placeholder)", wf.compute_descriptors, cands)
    est = step("5 · Property estimation (LLM)", wf.estimate_properties, brief, cands, desc, lit)
    return {"run_id": inputs_sig, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "brief": brief, "literature": lit, "candidates": cands, "smiles_repair": repair,
            "descriptors": desc, "estimates": est, "log": log}


if run:
    st.session_state["result"] = run_pipeline()
if "result" not in st.session_state:
    st.info("Configure a customer brief on the left and click **Run screening**.")
    st.stop()

r = st.session_state["result"]
if r["run_id"] != inputs_sig:
    st.info("Brief changed. Click **Run screening** again to re-run the workflow. (Weights re-rank instantly.)")

# Ranking is cheap and deterministic, so weights re-rank without re-running the LLM steps.
ranked = wf.rank(r["candidates"], r["estimates"], weights, r["brief"].non_flammable_required or prioritize_safety)
df = pd.DataFrame(ranked)
cand_by_id = {c.id: c for c in r["candidates"]}

st.subheader("1 · Workflow orchestration")
statuses = {e["step"]: e for e in r["log"]}
for name, entry in statuses.items():
    c1, c2, c3 = st.columns([3, 1, 4])
    c1.write(f"**{name}**")
    c2.write("✅ Done")
    c3.caption(" · ".join(f"{k}: {v}" for k, v in entry.items() if k != "step" and v is not None))
c1, c2, c3 = st.columns([3, 1, 4])
c1.write("**6 · Rank and explain**"); c2.write("✅ Done"); c3.caption("Transparent weighted ranking (see weights)")
c1, c2, c3 = st.columns([3, 1, 4])
c1.write("**7 · Experimental validation**"); c2.write("⏳ Pending"); c3.caption("Partner lab measurements required")

with st.expander("Parsed brief"):
    st.json(r["brief"].model_dump())
with st.expander(f"Literature findings ({len(r['literature']['sources'])} sources)"):
    st.markdown(r["literature"]["summary"])
rep = r["smiles_repair"]
with st.expander(f"SMILES repair loop ({len(rep['history'])} repairs, {len(rep['unresolved'])} unresolved)"):
    if rep["history"]:
        st.dataframe(pd.DataFrame(rep["history"]), width="stretch", hide_index=True)
    else:
        st.write("All SMILES passed RDKit and identity checks on the first pass.")
    if rep["unresolved"]:
        st.error("Unresolved after max rounds — descriptors for these components are unreliable.")
        st.dataframe(pd.DataFrame(rep["unresolved"]), width="stretch", hide_index=True)

st.subheader("2 · Ranked candidate shortlist")
a, b, c = st.columns(3)
a.metric("Candidates", len(df))
b.metric("Top score", f"{df.iloc[0]['Score']:.3f}" if len(df) else "–")
c.metric("Literature sources", len(r["literature"]["sources"]))
st.dataframe(df, width="stretch", hide_index=True)
if len(df):
    st.bar_chart(df.set_index("Candidate")["Score"])

st.subheader("3 · Evidence and traceability")
choice = st.selectbox("Inspect candidate", df["Candidate"].tolist())
cand, est, desc = cand_by_id[choice], r["estimates"][choice], r["descriptors"][choice]
st.markdown(f"**{choice} — {cand.name}** ({cand.family})")
st.write("**Formulation**")
st.dataframe(pd.DataFrame([x.model_dump() for x in cand.components]), width="stretch", hide_index=True)
st.write(f"**Rationale:** {cand.rationale}")
st.write(f"**Estimate reasoning** (uncertainty: {est.uncertainty}): {est.reasoning}")
st.write("**RDKit descriptors**")
st.dataframe(pd.DataFrame(desc["components"]), width="stretch", hide_index=True)
st.caption(f"Fluorine fraction: {desc['fluorine_fraction']} · Contains P: {desc['contains_phosphorus']}"
           + (f" · ⚠ Invalid SMILES: {', '.join(desc['invalid_smiles'])}" if desc["invalid_smiles"] else ""))
st.write("**Literature references**")
for u in cand.supporting_urls or ["(none cited)"]:
    st.markdown(f"- {u}")
st.info("Not yet available: DFT jobs, trained-model predictions with calibrated uncertainty, partner-lab measurement IDs.")

st.subheader("4 · Export pilot record")
record = {
    "run_id": r["run_id"], "generated_at_utc": r["generated_at_utc"], "model": wf.MODEL,
    "customer_form": form, "free_text": free_text, "parsed_brief": r["brief"].model_dump(),
    "weights": weights, "literature": r["literature"], "workflow_log": r["log"],
    "smiles_repair": r["smiles_repair"],
    "candidates": [{**c.model_dump(), "descriptors": r["descriptors"][c.id],
                    "estimate": r["estimates"][c.id].model_dump() if c.id in r["estimates"] else None}
                   for c in r["candidates"]],
    "ranking": ranked,
    "method": "llm_estimates_plus_rdkit_descriptors_not_experimentally_validated",
}
st.download_button("⬇ Download JSON record", data=json.dumps(record, indent=2),
                   file_name=f"solvation_run_{r['run_id']}.json", mime="application/json")
st.download_button("⬇ Download shortlist CSV", data=df.to_csv(index=False),
                   file_name=f"solvation_shortlist_{r['run_id']}.csv", mime="text/csv")
st.caption("Solvation Labs | YES!Delft Validation Program 2026 | Prototype")
