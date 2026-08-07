create table if not exists public.sentiment_model_versions (
    id uuid primary key default gen_random_uuid(),
    model_version text not null unique,
    model_path text not null,
    base_model text,
    dataset_name text not null,
    trained_at timestamptz not null,
    training_rows integer not null check (training_rows > 0),
    validation_rows integer not null check (validation_rows > 0),
    test_rows integer not null check (test_rows > 0),
    class_distribution jsonb not null default '{}'::jsonb,
    hyperparameters jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    labels jsonb not null default '[]'::jsonb,
    evaluation_mode text not null,
    is_active boolean not null default false,
    created_at timestamptz not null default now(),
    constraint sentiment_model_versions_class_distribution_object
        check (jsonb_typeof(class_distribution) = 'object'),
    constraint sentiment_model_versions_hyperparameters_object
        check (jsonb_typeof(hyperparameters) = 'object'),
    constraint sentiment_model_versions_metrics_object
        check (jsonb_typeof(metrics) = 'object'),
    constraint sentiment_model_versions_labels_array
        check (jsonb_typeof(labels) = 'array')
);

alter table public.sentiment_model_versions enable row level security;
revoke all on table public.sentiment_model_versions from anon, authenticated;
grant all on table public.sentiment_model_versions to service_role;

create unique index if not exists sentiment_model_versions_active_idx
    on public.sentiment_model_versions (is_active)
    where is_active = true;

insert into public.sentiment_model_versions (
    model_version,
    model_path,
    base_model,
    dataset_name,
    trained_at,
    training_rows,
    validation_rows,
    test_rows,
    hyperparameters,
    metrics,
    labels,
    evaluation_mode,
    is_active
)
values (
    'balibpt/finbert-stocklens',
    'https://huggingface.co/balibpt/finbert-stocklens',
    'ProsusAI/finbert',
    'zeroshot/twitter-financial-news-sentiment',
    '2026-06-06T00:00:00Z',
    8352,
    1790,
    1790,
    '{"learning_rate": 0.00002, "batch_size": 16, "epochs": 3, "random_seed": 42, "max_length": 128, "fine_tuning": "full"}'::jsonb,
    '{"accuracy": 0.872, "macro_f1": 0.83, "per_class_f1": {"negative": 0.84, "neutral": 0.77, "positive": 0.87}}'::jsonb,
    '["negative", "neutral", "positive"]'::jsonb,
    'held_out_70_15_15_test_split',
    true
)
on conflict (model_version) do update
set model_path = excluded.model_path,
    base_model = excluded.base_model,
    dataset_name = excluded.dataset_name,
    trained_at = excluded.trained_at,
    training_rows = excluded.training_rows,
    validation_rows = excluded.validation_rows,
    test_rows = excluded.test_rows,
    hyperparameters = excluded.hyperparameters,
    metrics = excluded.metrics,
    labels = excluded.labels,
    evaluation_mode = excluded.evaluation_mode,
    is_active = excluded.is_active;

notify pgrst, 'reload schema';
