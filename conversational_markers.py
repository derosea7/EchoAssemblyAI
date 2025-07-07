from pydantic import BaseModel, Field
from typing import Optional, List

class ConversationalMarkers(BaseModel):
    questions_and_answers: str = Field(
        description="List of questions identified, with their answers if found within the segment."
    )
    action_items: str = Field(
        description="List of tasks assigned or commitments to act."
    )