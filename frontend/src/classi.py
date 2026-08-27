from pydantic import BaseModel

class EvaluateFrontendRequest(BaseModel):
    parsed_text: str
    gold_text: str
    url: str


class AddWebResourceFrontendRequest(BaseModel):
    url: str
    html_text: str = ""
    title: str = ""
    domain: str = ""


class AddGoldStandardFrontendRequest(BaseModel):
    url: str
    gold_text: str = ""


class UrlFrontendRequest(BaseModel):
    url: str

class ParseFrontendRequest(BaseModel):
    url: str
    local: bool = False
    html_text: str | None = None

class ChatFrontendRequest(BaseModel):
    message: str
    use_planner: bool = True
    use_reasoning: bool = False
