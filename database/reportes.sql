-- ============================================================
--  Esquema de la base de datos (Supabase / PostgreSQL)
--  Tabla donde la app móvil guarda los reportes de chipos.
--
--  Cómo aplicarlo:
--    Supabase → proyecto → SQL Editor → pegar este archivo → Run.
--  Proyecto Supabase: zgamhsblxnfvkxisqkqc
-- ============================================================

create table if not exists public.reportes (
  id          uuid primary key default gen_random_uuid(),
  especie     text not null,              -- nombre de la especie (o "Especie no identificada")
  confianza   real,                       -- confianza del modelo, 0..1
  nota        text,                       -- nota opcional del lugar
  latitud     double precision,           -- ubicación redondeada (3 decimales)
  longitud    double precision,
  fecha       timestamptz,                -- fecha en que se tomó la foto
  creado_en   timestamptz default now()   -- fecha de inserción en la BDD
);

-- Seguridad a nivel de fila (RLS) activada.
alter table public.reportes enable row level security;

-- ------------------------------------------------------------
--  Políticas (demo): permitir insertar y leer con la anon key.
--  NOTA: esto deja que cualquiera con la anon key escriba/lea.
--  Es suficiente para la tesis/demo. Cuando la app use el login
--  real de Supabase, cambiar a políticas por usuario (auth.uid()).
-- ------------------------------------------------------------
drop policy if exists "insertar_reportes_anon" on public.reportes;
create policy "insertar_reportes_anon"
  on public.reportes for insert to anon
  with check (true);

drop policy if exists "leer_reportes_anon" on public.reportes;
create policy "leer_reportes_anon"
  on public.reportes for select to anon
  using (true);
