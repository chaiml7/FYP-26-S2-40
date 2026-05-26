import yfinance as yf


def fetch_stock_history(stock_id: int, symbol: str, period: str = "6mo", interval: str = "1d"):
    ticker = yf.Ticker(symbol.upper())

    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=False
    )

    if df.empty:
        return []

    df = df.reset_index()

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "stock_id": stock_id,
            "symbol": symbol.upper(),
            "trade_date": row["Date"].date().isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "source": "yfinance"
        })

    return rows