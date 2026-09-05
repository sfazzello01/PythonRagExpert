from pydantic import BaseModel


class ChatFrontendRequest(BaseModel):
    message: str
    use_planner: bool = True
    use_reasoning: bool = False
