create table if not exists public.sentiment_sources (
    id uuid primary key default gen_random_uuid(),
    source_type text not null,
    account text not null,
    relevance text,
    is_active boolean not null default true,
    created_at timestamptz not null default now()
);

alter table public.sentiment_sources enable row level security;
revoke all on table public.sentiment_sources from anon, authenticated;
grant all on table public.sentiment_sources to service_role;

notify pgrst, 'reload schema';
