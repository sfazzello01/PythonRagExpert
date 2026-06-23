from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ParseHtmlRequest(BaseModel):
    url: str
    html_text: str

class ParceRequest(BaseModel):
    """ se local = false scarica la pagina dal web altrimenti la recupera dal database"""
    url: str
    local: Optional[bool] = False
    html_text: Optional[str] = None

class Parse(BaseModel):
    url: str
    domain: str
    title: str
    html_text: str
    parsed_text: str


class Domains(BaseModel):
    domains: List[str]


class GoldStandard(BaseModel):
    url: str
    domain: str
    title: str
    html_text: str
    gold_text: str


class FullGoldStandard(BaseModel):
    gold_standard: List[GoldStandard]

class GoldStandardUrls(BaseModel):
    gold_standard_urls: List[str]

class TokenLevelEval(BaseModel):
    precision: float
    recall: float
    f1: float


class FullGsEval(BaseModel):
    token_level_eval: TokenLevelEval
    levenshtein_distance: float
    jaccard_similarity: float
    judge_score: float
    x_eval: Optional[Dict[str, Any]] = None


class EvaluateResponse(BaseModel):
    token_level_eval: TokenLevelEval
    levenshtein_distance: float
    jaccard_similarity: float
    x_eval: Optional[Dict[str, Any]] = None


class EvaluateRequest(BaseModel):
    parsed_text: str
    gold_text: str
    url: Optional[str] = None


class EvaluateJudgeResponce(BaseModel):
    model_name: str
    judge_score: int
    judge_feedback: str
    x_eval: Optional[Dict[str, Any]] = None


class AddWebResourceRequest(BaseModel):
    url: str
    html_text: str = ""
    title: str = ""
    domain: str = ""

class AddGoldStandardRequest(BaseModel):
    url: str
    gold_text: str = ""

class UrlRequest(BaseModel):
    url: str

class StatusResponce(BaseModel):
    backend: str
    database: str
    ollama: str
