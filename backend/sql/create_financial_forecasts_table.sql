create table financial_forecasts (
  id uuid primary key default gen_random_uuid(),
  stock_id uuid,
  symbol text not null,
  category text not null,
  item text not null,
  next_quarter text not null,
  predicted_value numeric not null,
  confidence numeric,
  model_path text,
  model_version text,
  created_at timestamptz default now()
);

create index on financial_forecasts (symbol);
create index on financial_forecasts (category);
