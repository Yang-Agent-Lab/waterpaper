-- Science project hzwwxlaxyxnwsxwpjwpt. Read-only entrypoints.
-- Source files need not be opened to answer these inventory questions.

-- Dataset readiness and the observation time of this assessment.
select name,country_code,inventory_status,model_readiness,planned_role,
       observed_at,registered_stations,registered_events,latest_profiles,summary
from public.research_dataset_readiness
where project_slug='waterpaper'
order by country_code,dataset_key;

-- Station distribution/coverage, preserving reference population and versions.
select d.name,e.native_id,e.name,p.scope,p.profile_kind,p.method_version,
       p.n_rows,p.n_valid_q,p.n_valid_p,p.statistics,p.quality,p.provenance,
       p.computed_at,d.observed_at,p.source_fingerprint
from public.research_data_profiles_latest p
join public.research_data_entities e using(entity_key,dataset_key)
join public.research_datasets d using(dataset_key)
where p.project_slug='waterpaper' and e.entity_type='station'
order by d.country_code,e.native_id;

-- What actually prevents the requested physical classification?
-- Keep older assignments for history; this query selects the latest assessment.
with latest_classification as (
  select distinct on (entity_key,rule_key) *
  from public.research_data_classifications
  where project_slug='waterpaper'
  order by entity_key,rule_key,assigned_at desc,registered_at desc,assignment_key
)
select d.name,e.native_id,r.name as classification_basis,c.status,c.label,c.evidence
from latest_classification c
join public.research_data_entities e using(entity_key)
join public.research_datasets d using(dataset_key)
join public.research_data_classification_rules r using(rule_key)
where c.project_slug='waterpaper' and c.status in ('NOT_EVALUABLE','partial','unresolved_station_mapping')
order by d.country_code,e.native_id,r.name;

-- Existing event morphology; do not equate author labels with independent storms.
select d.name,e.native_id,e.role_label,p.statistics,p.quality,p.provenance
from public.research_data_profiles_latest p
join public.research_data_entities e using(entity_key,dataset_key)
join public.research_datasets d using(dataset_key)
where p.project_slug='waterpaper' and e.entity_type='event'
order by e.entity_key;

-- Show previous cached versions rather than silently overwriting an old result.
select entity_key,profile_kind,scope,method_version,source_fingerprint,computed_at,status
from public.research_data_profiles
where project_slug='waterpaper'
order by entity_key,profile_kind,scope,computed_at desc;
