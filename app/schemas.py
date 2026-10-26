from pydantic import BaseModel


class ClassProb(BaseModel):
    name: str
    prob: float


class Prediction(BaseModel):
    top_class: str
    confidence: float
    prob_table: list[ClassProb]
    width: int
    height: int
    elapsed_ms: float
    model_id: str
