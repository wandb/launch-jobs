provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "my-context"
}

provider "helm" {
  debug = true
  kubernetes {
    config_path = "~/.kube/config"
  }
}

module "launch_helm" {
  source = "../"

  helm_repository              = var.helm_repository
  helm_chart_version           = var.helm_chart_version
  agent_labels                 = var.agent_labels
  agent_api_key                = var.agent_api_key
  agent_image                  = var.agent_image
  agent_image_pull_policy      = var.agent_image_pull_policy
  namespace                    = var.namespace
  base_url                     = var.base_url
  additional_target_namespaces = var.additional_target_namespaces
  launch_config                = var.launch_config
  volcano                      = var.volcano
  git_creds                    = var.git_creds
  service_account_annotations  = var.service_account_annotations
  azure_storage_access_key     = var.azure_storage_access_key
}
