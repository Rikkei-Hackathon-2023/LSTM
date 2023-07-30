from typing import List, Optional
from pydantic import BaseModel


class Data(BaseModel):
    data: List[List[float]]
    num_predict: int = 6
