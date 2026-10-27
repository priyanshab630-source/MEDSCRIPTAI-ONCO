"""FastAPI wrapper around the agent pipeline.

    uvicorn api.app:app --reload --port 8000
    open http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.workflow import run_pipeline
from src.core.schemas import Hypothesis
from src.tools import drugs as drug_lookup
from src.tools import llm

app = FastAPI(
    title="MEDSCRIPTAI-ONCO",
    description="Evidence-grounded, contradiction-aware drug-target hypothesis validation.",
    version="0.1.0",
)

STATIC_DIR = __file__.rsplit("/", 1)[0] + "/static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class HypothesisRequest(BaseModel):
    drug: str = Field(..., examples=["SHP099"])
    target: str = Field(..., examples=["SHP2"])
    disease: str = Field("", examples=["KRAS-mutant NSCLC"])
    smiles: str = ""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": "sarvam" if llm.available() else "rule-based fallback",
        "agents": 6,
    }


@app.get("/drugs")
def list_drugs():
    return drug_lookup.all_drugs()


@app.post("/analyze")
def analyze(request: HypothesisRequest):
    smiles = request.smiles
    if not smiles:
        record = drug_lookup.get(request.drug)
        smiles = record.get("smiles", "")
    if not smiles:
        raise HTTPException(
            status_code=422,
            detail="No SMILES for this compound: pass one explicitly in the request.",
        )

    hypothesis = Hypothesis(drug=request.drug, target=request.target,
                            disease=request.disease, smiles=smiles)
    return run_pipeline(hypothesis).to_dict()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR + "/index.html")
