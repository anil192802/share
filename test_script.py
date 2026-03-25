import pandas as pd
from datetime import datetime, timedelta
import random

# test what a random history gives
from nepal_stock_app.technical import add_technical_indicators, evaluate_technical_signal

hist = pd.DataFrame({
    'date': pd.date_range(start='1/1/2022', periods=100),
    'open': [100 + random.random()*10 for _ in range(100)],
    'high': [110 + random.random()*10 for _ in range(100)],
    'low': [90 + random.random()*10 for _ in range(100)],
    'close': [105 + random.random()*10 for _ in range(100)],
    'volume': [1000 + random.random()*100 for _ in range(100)],
})
df = add_technical_indicators(hist)
sig = evaluate_technical_signal(df)
print("Signal:", sig.signal)
