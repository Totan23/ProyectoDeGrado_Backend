-- ============================================================
--  Esquema de la base de datos (Supabase / PostgreSQL)
--  App móvil de vigilancia del chipo (mal de Chagas).
--
--  Cómo aplicarlo:
--    Supabase -> proyecto -> SQL Editor -> pegar este archivo -> Run.
--  Es idempotente: se puede correr varias veces sin romper nada.
--  Proyecto Supabase: zgamhsblxnfvkxisqkqc
--
--  IMPORTANTE (registro sin confirmar correo):
--    Para que el registro deje entrar de una (sin pedir confirmación
--    por correo), en el panel de Supabase ir a:
--      Authentication -> Sign In / Providers -> Email
--      -> desactivar "Confirm email".
--    Esto no se configura por SQL, es un ajuste del panel.
-- ============================================================

-- ------------------------------------------------------------
--  PERFILES
--  Supabase Auth ya guarda el correo y la contraseña en auth.users.
--  Aquí guardamos los datos extra del formulario de registro
--  (nombre de usuario, ubicación de texto, teléfono).
-- ------------------------------------------------------------
create table if not exists public.perfiles (
  id         uuid primary key references auth.users(id) on delete cascade,
  nombre     text,
  ubicacion  text,
  telefono   text,
  creado_en  timestamptz default now()
);

alter table public.perfiles enable row level security;

-- Cada usuario solo ve y edita su propio perfil.
drop policy if exists "perfiles_select_propio" on public.perfiles;
create policy "perfiles_select_propio"
  on public.perfiles for select to authenticated
  using (auth.uid() = id);

drop policy if exists "perfiles_insert_propio" on public.perfiles;
create policy "perfiles_insert_propio"
  on public.perfiles for insert to authenticated
  with check (auth.uid() = id);

drop policy if exists "perfiles_update_propio" on public.perfiles;
create policy "perfiles_update_propio"
  on public.perfiles for update to authenticated
  using (auth.uid() = id);

-- ------------------------------------------------------------
--  TRIGGER: al crear un usuario en auth.users, crear su perfil
--  copiando los metadatos que envía la app en el registro.
-- ------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.perfiles (id, nombre, ubicacion, telefono)
  values (
    new.id,
    new.raw_user_meta_data->>'nombre',
    new.raw_user_meta_data->>'ubicacion',
    new.raw_user_meta_data->>'telefono'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- El trigger sigue ejecutando la función (no depende de este privilegio),
-- pero así NO se puede invocar como RPC público (cierra un aviso de seguridad).
revoke execute on function public.handle_new_user() from anon, authenticated, public;

-- ------------------------------------------------------------
--  AUTO-CONFIRMACIÓN DE CORREO
--  Marca el correo como confirmado al crear el usuario, para que el
--  registro NO exija confirmar por email (la app inicia sesión de una).
-- ------------------------------------------------------------
create or replace function public.auto_confirm_user()
returns trigger
language plpgsql
security definer set search_path = auth, public
as $$
begin
  if new.email_confirmed_at is null then
    new.email_confirmed_at := now();
  end if;
  return new;
end;
$$;

drop trigger if exists on_auth_user_autoconfirm on auth.users;
create trigger on_auth_user_autoconfirm
  before insert on auth.users
  for each row execute function public.auto_confirm_user();

revoke execute on function public.auto_confirm_user() from anon, authenticated, public;

-- ------------------------------------------------------------
--  REPORTES
--  Cada reporte pertenece a un usuario (usuario_id).
-- ------------------------------------------------------------
create table if not exists public.reportes (
  id          uuid primary key default gen_random_uuid(),
  usuario_id  uuid references auth.users(id) on delete cascade,
  client_id   text,                       -- id local (idempotencia: evita duplicados al subir)
  especie     text not null,              -- nombre de la especie (o "Especie no identificada")
  confianza   real,                       -- confianza del modelo, 0..1
  nota        text,                       -- nota opcional del lugar
  lugar       text,                       -- lugar escrito (respaldo si no hay GPS)
  latitud     double precision,           -- ubicación (redondeada ~100 m por privacidad)
  longitud    double precision,
  foto_url    text,                       -- URL pública de la foto en Storage
  autor       text,                       -- nombre a mostrar (NULL = anónimo)
  fecha       timestamptz,                -- fecha en que se tomó la foto
  creado_en   timestamptz default now()   -- fecha de inserción en la BDD
);

-- Por si la tabla ya existía sin estas columnas:
alter table public.reportes
  add column if not exists usuario_id uuid references auth.users(id) on delete cascade;
alter table public.reportes add column if not exists foto_url  text;
alter table public.reportes add column if not exists lugar     text;
alter table public.reportes add column if not exists client_id text;
alter table public.reportes add column if not exists autor     text;

alter table public.reportes enable row level security;

-- Quitar las políticas anónimas antiguas (versión demo sin login).
drop policy if exists "insertar_reportes_anon" on public.reportes;
drop policy if exists "leer_reportes_anon" on public.reportes;

-- Cada usuario solo INSERTA sus propios reportes.
drop policy if exists "reportes_insert_propio" on public.reportes;
create policy "reportes_insert_propio"
  on public.reportes for insert to authenticated
  with check (auth.uid() = usuario_id);

-- Mapa COMUNITARIO: cualquiera puede LEER los reportes para pintarlos en el mapa.
-- (La identidad -nombre/teléfono- vive en "perfiles", que sigue siendo privado.)
drop policy if exists "reportes_select_propio"  on public.reportes;
drop policy if exists "reportes_select_publico" on public.reportes;
create policy "reportes_select_publico"
  on public.reportes for select to anon, authenticated
  using (true);

-- Índices para consultas por usuario y para el mapa.
create index if not exists reportes_usuario_idx on public.reportes(usuario_id);
create index if not exists reportes_geo_idx     on public.reportes(latitud, longitud);
-- Idempotencia: un reporte (client_id) no se puede insertar dos veces.
create unique index if not exists reportes_client_id_key on public.reportes(client_id);

-- ------------------------------------------------------------
--  TIEMPO REAL (Supabase Realtime)
--  Publica la tabla reportes para que el mapa se actualice en vivo
--  cuando alguien sube un reporte.
-- ------------------------------------------------------------
alter table public.reportes replica identity full;
do $$
begin
  if not exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    create publication supabase_realtime;
  end if;
  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'reportes'
  ) then
    alter publication supabase_realtime add table public.reportes;
  end if;
end $$;

-- ------------------------------------------------------------
--  STORAGE: bucket público para las fotos de los reportes
--  (para mostrarlas en el mapa y recolectar datos para el modelo).
-- ------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('reportes-fotos', 'reportes-fotos', true)
on conflict (id) do nothing;

drop policy if exists "fotos_insert_auth" on storage.objects;
create policy "fotos_insert_auth"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'reportes-fotos');

drop policy if exists "fotos_select_publico" on storage.objects;
create policy "fotos_select_publico"
  on storage.objects for select to anon, authenticated
  using (bucket_id = 'reportes-fotos');
