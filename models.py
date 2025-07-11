from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    userId: int
    interactedProductIds: List[int]
    topN: int = 3