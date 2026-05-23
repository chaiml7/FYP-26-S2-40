import { useEffect, useState } from 'react'

function App() {
  const [stocks, setStocks] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchStocks() {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/stocks`)

        if (!response.ok) {
          throw new Error('Failed to fetch stocks from backend')
        }

        const data = await response.json()
        setStocks(data)
      } catch (err) {
        setError(err.message)
      }
    }

    fetchStocks()
  }, [])

  return (
    <div style={{ padding: '40px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Webservice for Stock Market Prediction</h1>
      <h2>Final Year Project</h2>
      <p>Project ID: CSIT-26-S2-13</p>

      <p>
        This project aims to develop a webservice that provides stock market
        forecasting and recommendation using machine learning.
      </p>

      <hr style={{ margin: '30px 0' }} />

      <h2>Backend REST API Connection Test</h2>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {stocks.length === 0 && !error ? (
        <p>Loading or no data found...</p>
      ) : (
        <ul>
          {stocks.map((stock) => (
            <li key={stock.id}>
              {stock.symbol} - {stock.company_name}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default App