# StockLens

StockLens is a web app that helps users review stocks in one place. It brings together price data, technical analysis, news sentiment and company financial information to produce a simple stock outlook.

Users can view each part of the analysis separately or see an overall score based on all three models.

## Features

- Shows current and historical stock prices
- Displays candlestick charts and technical indicators
- Gives technical, sentiment and financial analysis scores
- Combines the three scores into one overall outlook
- Shows recent news for each stock
- Lets users create a watchlist
- Sends optional email alerts when new analysis is ready
- Supports Free and Premium accounts
- Lets Premium users adjust the model weightages
- Allows users to submit feedback to the administrator

The administrator can manage users, stocks, feedback and parts of the analysis system.

## How it is built

StockLens uses Python and FastAPI for the website and API. The pages are built with Jinja templates, HTML, CSS and JavaScript. Supabase is used for user accounts and database storage.

The project also uses machine learning models for technical, sentiment and financial analysis. Stripe is used for Premium subscriptions.


## Important note

StockLens is an educational project. Its scores and predictions are not financial advice, and they do not guarantee future market results.
