import yfinance as yf


def fetch_stock_history(symbol: str, period: str = "6mo", interval: str = "1d"):
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
        trade_date = row["Date"].date().isoformat()

        rows.append({
            "symbol": symbol.upper(),
            "trade_date": trade_date,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "source": "yfinance"
        })

    return rows