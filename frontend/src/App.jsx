import { useEffect, useState } from 'react'
import { supabase } from './supabaseClient'

function App() {
  const [stocks, setStocks] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchStocks() {
      const { data, error } = await supabase
        .from('stocks')
        .select('*')

      if (error) {
        setError(error.message)
      } else {
        setStocks(data)
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

      <h2>Supabase Connection Test</h2>

      {error && <p>Error: {error}</p>}

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