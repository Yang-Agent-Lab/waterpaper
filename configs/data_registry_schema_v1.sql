-- Science project hzwwxlaxyxnwsxwpjwpt. Additive data catalogue only.
-- Raw time series stay on the server. No change to existing experiment tables.
create table public.research_datasets (
  dataset_key text primary key,
  project_slug text not null references public.research_projects(slug),
  name text not null,
  country_code text not null,
  source_url text,
  source_version text,
  data_kind text not null,
  nominal_interval_minutes numeric check (nominal_interval_minutes > 0),
  time_zone text,
  variables jsonb not null default '{}'::jsonb,
  server_uri text not null,
  source_fingerprint text not null,
  fingerprint_method text not null,
  inventory_status text not null,
  model_readiness text not null,
  planned_role text not null,
  summary jsonb not null default '{}'::jsonb check (jsonb_typeof(summary)='object'),
  observed_at timestamptz not null,
  registered_at timestamptz not null default now(),
  unique (dataset_key,project_slug)
);

create table public.research_data_entities (
  entity_key text primary key,
  project_slug text not null references public.research_projects(slug),
  dataset_key text not null,
  entity_type text not null check (entity_type in ('dataset','station','event')),
  native_id text not null,
  station_key text,
  name text,
  role_label text not null,
  time_start text,
  time_end text,
  attributes jsonb not null default '{}'::jsonb check (jsonb_typeof(attributes)='object'),
  status text not null,
  observed_at timestamptz not null,
  registered_at timestamptz not null default now(),
  unique (entity_key,dataset_key),
  unique (entity_key,project_slug),
  foreign key (dataset_key,project_slug) references public.research_datasets(dataset_key,project_slug),
  foreign key (station_key,dataset_key) references public.research_data_entities(entity_key,dataset_key) deferrable initially deferred
);

create table public.research_data_profiles (
  profile_key text primary key,
  project_slug text not null references public.research_projects(slug),
  entity_key text not null,
  dataset_key text not null,
  profile_kind text not null,
  scope text not null,
  method_version text not null,
  source_fingerprint text not null,
  fingerprint_method text not null,
  n_rows bigint check (n_rows >= 0),
  n_valid_q bigint check (n_valid_q >= 0),
  n_valid_p bigint check (n_valid_p >= 0),
  time_start text,
  time_end text,
  statistics jsonb not null default '{}'::jsonb check (jsonb_typeof(statistics)='object'),
  quality jsonb not null default '{}'::jsonb check (jsonb_typeof(quality)='object'),
  provenance jsonb not null default '{}'::jsonb check (jsonb_typeof(provenance)='object'),
  status text not null,
  computed_at timestamptz not null,
  registered_at timestamptz not null default now(),
  unique (entity_key,profile_kind,scope,method_version,source_fingerprint),
  unique (profile_key,entity_key),
  unique (profile_key,project_slug),
  foreign key (entity_key,dataset_key) references public.research_data_entities(entity_key,dataset_key),
  foreign key (dataset_key,project_slug) references public.research_datasets(dataset_key,project_slug),
  check (n_valid_q is null or n_rows is null or n_valid_q <= n_rows),
  check (n_valid_p is null or n_rows is null or n_valid_p <= n_rows)
);

create table public.research_data_classification_rules (
  rule_key text primary key,
  project_slug text not null references public.research_projects(slug),
  name text not null,
  level text not null,
  method_version text not null,
  status text not null,
  definition jsonb not null default '{}'::jsonb check (jsonb_typeof(definition)='object'),
  required_fields jsonb not null default '[]'::jsonb check (jsonb_typeof(required_fields)='array'),
  source_refs jsonb not null default '[]'::jsonb check (jsonb_typeof(source_refs)='array'),
  notes text,
  registered_at timestamptz not null default now(),
  unique (rule_key,project_slug)
);

create table public.research_data_classifications (
  assignment_key text primary key,
  project_slug text not null references public.research_projects(slug),
  entity_key text not null,
  profile_key text,
  rule_key text not null,
  label text,
  status text not null,
  evidence jsonb not null default '{}'::jsonb check (jsonb_typeof(evidence)='object'),
  assigned_at timestamptz not null,
  registered_at timestamptz not null default now(),
  foreign key (entity_key,project_slug) references public.research_data_entities(entity_key,project_slug),
  foreign key (profile_key,entity_key) references public.research_data_profiles(profile_key,entity_key),
  foreign key (rule_key,project_slug) references public.research_data_classification_rules(rule_key,project_slug)
);

create index research_datasets_country_readiness_idx on public.research_datasets(project_slug,country_code,model_readiness);
create index research_data_entities_dataset_type_idx on public.research_data_entities(dataset_key,entity_type);
create index research_data_entities_station_idx on public.research_data_entities(station_key);
create index research_data_profiles_latest_idx on public.research_data_profiles(entity_key,profile_kind,scope,computed_at desc);
create index research_data_profiles_dataset_idx on public.research_data_profiles(dataset_key);
create index research_data_classifications_rule_status_idx on public.research_data_classifications(rule_key,status);
create index research_data_classifications_entity_idx on public.research_data_classifications(entity_key);
create index research_data_classifications_profile_idx on public.research_data_classifications(profile_key);

alter table public.research_datasets enable row level security;
alter table public.research_data_entities enable row level security;
alter table public.research_data_profiles enable row level security;
alter table public.research_data_classification_rules enable row level security;
alter table public.research_data_classifications enable row level security;
revoke all on public.research_datasets, public.research_data_entities,
  public.research_data_profiles, public.research_data_classification_rules,
  public.research_data_classifications from public, anon, authenticated, service_role;
grant select,insert,update on public.research_datasets,public.research_data_entities to service_role;
grant select,insert on public.research_data_profiles,public.research_data_classification_rules,
  public.research_data_classifications to service_role;
-- No end-user policy: this catalogue is accessible through the existing
-- privileged research connection, not exposed to public or all signed-in users.

create view public.research_data_profiles_latest with (security_invoker=true) as
select distinct on (entity_key,profile_kind,scope) *
from public.research_data_profiles
order by entity_key,profile_kind,scope,computed_at desc,registered_at desc,profile_key;

create view public.research_dataset_readiness with (security_invoker=true) as
select d.dataset_key,d.project_slug,d.name,d.country_code,d.inventory_status,
       d.model_readiness,d.planned_role,d.observed_at,d.summary,
       (select count(*) from public.research_data_entities e where e.dataset_key=d.dataset_key and e.entity_type='station') as registered_stations,
       (select count(*) from public.research_data_entities e where e.dataset_key=d.dataset_key and e.entity_type='event') as registered_events,
       (select count(*) from public.research_data_profiles_latest p where p.dataset_key=d.dataset_key) as latest_profiles
from public.research_datasets d;
revoke all on public.research_data_profiles_latest,public.research_dataset_readiness from public,anon,authenticated,service_role;
grant select on public.research_data_profiles_latest,public.research_dataset_readiness to service_role;

comment on table public.research_datasets is 'Source/version inventory. Archive availability does not imply paired hourly forecasting readiness.';
comment on table public.research_data_profiles is 'Versioned statistical cache. Inventory quantiles are not fit thresholds. Keep scope, population, units and provenance.';
comment on table public.research_data_classifications is 'Evidence-bound classifications; unknown or missing inputs remain explicit. Country is not a flood type.';
