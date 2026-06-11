create table if not exists public.technical_model_versions (
    id uuid primary key default gen_random_uuid(),
    model_version text not null unique,
    model_path text not null,
    metadata_path text not null,
    trained_at timestamptz not null,
    training_rows integer not null check (training_rows > 0),
    train_rows integer not null check (train_rows > 0),
    validation_rows integer not null check (validation_rows > 0),
    test_rows integer not null check (test_rows > 0),
    dataset_start date not null,
    dataset_end date not null,
    class_distribution jsonb not null default '{}'::jsonb,
    hyperparameters jsonb not null default '{}'::jsonb,
    validation_metrics jsonb not null default '{}'::jsonb,
    test_metrics jsonb not null default '{}'::jsonb,
    feature_columns jsonb not null default '[]'::jsonb,
    labels jsonb not null default '[]'::jsonb,
    return_threshold double precision not null,
    evaluation_mode text not null,
    is_active boolean not null default false,
    created_at timestamptz not null default now()
);

alter table public.technical_model_versions enable row level security;
revoke all on table public.technical_model_versions from anon, authenticated;
grant all on table public.technical_model_versions to service_role;

alter table public.direction_predictions
    add column if not exists prediction text,
    add column if not exists probabilities jsonb,
    add column if not exists raw_outlook double precision,
    add column if not exists technical_score double precision,
    add column if not exists prediction_horizon text,
    add column if not exists model_version text;

update public.direction_predictions
set model_version = 'legacy_technical_v1'
where model_version is null;

alter table public.direction_predictions
    alter column model_version set default 'legacy_technical_v1',
    alter column model_version set not null,
    alter column predicted_direction drop not null;

alter table public.direction_predictions
    drop constraint if exists direction_predictions_predicted_direction_check,
    drop constraint if exists direction_predictions_prediction_check,
    drop constraint if exists direction_predictions_raw_outlook_check,
    drop constraint if exists direction_predictions_technical_score_check,
    drop constraint if exists direction_predictions_stock_id_latest_date_key;

alter table public.direction_predictions
    add constraint direction_predictions_predicted_direction_check
        check (
            predicted_direction is null
            or predicted_direction in ('up', 'neutral', 'down')
        ),
    add constraint direction_predictions_prediction_check
        check (
            prediction is null
            or prediction in ('bullish', 'neutral', 'bearish')
        ),
    add constraint direction_predictions_raw_outlook_check
        check (
            raw_outlook is null
            or raw_outlook between -1 and 1
        ),
    add constraint direction_predictions_technical_score_check
        check (
            technical_score is null
            or technical_score between 1 and 10
        ),
    add constraint direction_predictions_stock_date_model_key
        unique (stock_id, latest_date, model_version);

create unique index if not exists technical_model_versions_active_idx
    on public.technical_model_versions (is_active)
    where is_active = true;

create index if not exists direction_predictions_symbol_date_idx
    on public.direction_predictions (symbol, latest_date desc);

notify pgrst, 'reload schema';
