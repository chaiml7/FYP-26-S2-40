drop index if exists public.technical_model_versions_active_idx;

create unique index technical_model_versions_active_idx
    on public.technical_model_versions (is_active)
    where is_active = true;
