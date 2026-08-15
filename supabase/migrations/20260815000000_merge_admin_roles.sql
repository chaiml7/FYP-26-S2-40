-- Collapse the two admin roles (frontend_admin, backend_admin) into a single
-- "admin" role. The app used to have two separate admin accounts with no
-- server-rendered UI for backend_admin and a broken post-login redirect;
-- the product decision is that there should only ever be one Admin role.

-- 1. Make sure an "admin" row exists in roles, seeded from frontend_admin's
--    row when available (falls back to a sensible default otherwise).
insert into roles (id, name, tag, desc)
select 'admin', 'Admin', coalesce(tag, 'badge-purple'),
       'Full administrative access across user management, stock database, sentiment sources, weightages, and reports.'
from roles
where id = 'frontend_admin'
on conflict (id) do nothing;

insert into roles (id, name, tag, desc)
values ('admin', 'Admin', 'badge-purple',
        'Full administrative access across user management, stock database, sentiment sources, weightages, and reports.')
on conflict (id) do nothing;

-- 2. Repoint every user currently on either admin role onto the merged role.
update user_profiles
set role_id = 'admin'
where role_id in ('frontend_admin', 'backend_admin', 'user_admin');

-- 3. Drop the now-unreferenced legacy admin role rows.
delete from roles where id in ('frontend_admin', 'backend_admin', 'user_admin');
