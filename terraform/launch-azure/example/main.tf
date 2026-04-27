module "launch_helm" {
  source = "../"

  # Infra variables
  subscription_id       = var.subscription_id
  namespace             = var.namespace
  location              = var.location
  create_resource_group = var.create_resource_group
  create_aks_cluster    = var.create_aks_cluster

  # launch-helm variables
  helm_repository              = var.helm_repository
  helm_chart_version           = var.helm_chart_version
  agent_api_key                = var.agent_api_key
  agent_image                  = var.agent_image
  agent_image_pull_policy      = var.agent_image_pull_policy
  agent_resources              = var.agent_resources
  k8s_namespace                = var.k8s_namespace
  base_url                     = var.base_url
  volcano                      = var.volcano
  git_creds                    = var.git_creds
  additional_target_namespaces = var.additional_target_namespaces

  # launch_config variables
  queues         = var.queues
  entity         = var.entity
  max_jobs       = var.max_jobs
  max_schedulers = var.max_schedulers
}
