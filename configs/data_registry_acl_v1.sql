-- Narrow inherited default grants only for the new registry objects.
revoke all on public.research_datasets,public.research_data_entities,
  public.research_data_profiles,public.research_data_classification_rules,
  public.research_data_classifications from service_role;
grant select,insert,update on public.research_datasets,public.research_data_entities to service_role;
grant select,insert on public.research_data_profiles,public.research_data_classification_rules,
  public.research_data_classifications to service_role;
revoke all on public.research_data_profiles_latest,public.research_dataset_readiness from service_role;
grant select on public.research_data_profiles_latest,public.research_dataset_readiness to service_role;
