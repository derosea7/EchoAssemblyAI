from pydantic import BaseModel, Field
from typing import Optional, List

class TrendDetection(BaseModel):
    emerging_topics: str = Field(..., description="Emerging topics detected in the conversation")
    fading_topics: str = Field(..., description="Fading topics detected in the conversation")
    shifting_emphasis: str = Field(..., description="Shifting emphasis in the conversation")
