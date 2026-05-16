import pandera as pa
from pandera.typing import Series
import pandas as pd

class PriceHistorySchema(pa.DataFrameModel):
    Open: Series[float] = pa.Field(ge=0)
    High: Series[float] = pa.Field(ge=0)
    Low: Series[float] = pa.Field(ge=0)
    Close: Series[float] = pa.Field(ge=0)
    Volume: Series[int] = pa.Field(ge=0, coerce=True)

    @pa.dataframe_check
    def high_is_max(cls, df: pd.DataFrame) -> Series[bool]:
        # High should be >= Low, Open, and Close
        return (df["High"] >= df["Low"]) & (df["High"] >= df["Open"]) & (df["High"] >= df["Close"])
    
    @pa.dataframe_check
    def low_is_min(cls, df: pd.DataFrame) -> Series[bool]:
        # Low should be <= High, Open, and Close
        return (df["Low"] <= df["High"]) & (df["Low"] <= df["Open"]) & (df["Low"] <= df["Close"])
