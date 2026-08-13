create table if not exists public.user_subscriptions (
    user_id uuid primary key references auth.users(id) on delete cascade,
    stripe_customer_id text unique,
    stripe_subscription_id text unique,
    stripe_price_id text,
    status text not null default 'inactive',
    current_period_end timestamptz,
    cancel_at_period_end boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.stripe_webhook_events (
    event_id text primary key,
    event_type text not null,
    processing_status text not null default 'processing'
        check (processing_status in ('processing', 'processed', 'failed')),
    attempts integer not null default 1 check (attempts > 0),
    error_message text,
    created_at timestamptz not null default now(),
    processed_at timestamptz
);

create index if not exists user_subscriptions_status_idx
    on public.user_subscriptions (status);

alter table public.user_subscriptions enable row level security;
alter table public.stripe_webhook_events enable row level security;

revoke all on table public.user_subscriptions from anon;
revoke all on table public.stripe_webhook_events from anon;
revoke all on table public.stripe_webhook_events from authenticated;

grant select on table public.user_subscriptions to authenticated;
grant all on table public.user_subscriptions to service_role;
grant all on table public.stripe_webhook_events to service_role;

drop policy if exists "Users can read their own subscription" on public.user_subscriptions;
create policy "Users can read their own subscription"
    on public.user_subscriptions
    for select
    to authenticated
    using ((select auth.uid()) = user_id);

notify pgrst, 'reload schema';
