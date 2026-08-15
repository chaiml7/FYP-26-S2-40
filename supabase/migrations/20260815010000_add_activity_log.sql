-- Minimal user activity log: who did what, when. Denormalized (email stored
-- directly) to keep this simple — no FK to auth.users required to read it.
create table if not exists public.activity_log (
    id uuid primary key default gen_random_uuid(),
    email text not null,
    action text not null,
    detail text,
    created_at timestamptz not null default now()
);

create index if not exists activity_log_created_at_idx on public.activity_log (created_at desc);

alter table public.activity_log enable row level security;
revoke all on table public.activity_log from anon, authenticated;
grant all on table public.activity_log to service_role;

notify pgrst, 'reload schema';
