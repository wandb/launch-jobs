provider "azurerm" {
  subscription_id = var.subscription_id
  features {}
}

resource "random_id" "prefix" {
  byte_length = 4
}

resource "azurerm_resource_group" "launch_rg" {
  count = var.create_resource_group ? 1 : 0

  location = var.location
  name     = "${var.namespace}-${random_id.prefix.hex}-rg"
}

# locals {
#   resource_group = {
#     name     = var.create_resource_group ? azurerm_resource_group.main[0].name : var.resource_group_name
#     location = var.location
#   }
# }

module "networking" {
  source              = "./modules/networking"
  namespace           = "${var.namespace}-${random_id.prefix.hex}"
  resource_group_name = azurerm_resource_group.launch_rg.0.name
  location            = azurerm_resource_group.launch_rg.0.location

  # tags = var.tags
  depends_on = [azurerm_resource_group.launch_rg]
}

module "aks" {
  count  = var.create_aks_cluster ? 1 : 0
  source = "./modules/aks"

  name                = "${var.namespace}-${random_id.prefix.hex}-cluster"
  resource_group_name = azurerm_resource_group.launch_rg.0.name
  location            = azurerm_resource_group.launch_rg.0.location
  cluster_subnet_id   = module.networking.private_subnet.id

  azurerm_container_registry_id = module.acr.0.azurerm_container_registry_id

  node_count = 1

  depends_on = [
    module.networking.private_subnet,
    module.acr.azurerm_container_registry_id
  ]
}

module "blob" {
  count                = 1
  source               = "./modules/blob"
  storage_account_name = "${var.namespace}${random_id.prefix.hex}sa" #only lowercase letters and numbers
  container_name       = "${var.namespace}${random_id.prefix.hex}"   #only lowercase letters and numbers
  resource_group_name  = azurerm_resource_group.launch_rg.0.name
  location             = azurerm_resource_group.launch_rg.0.location
}

module "acr" {
  count               = 1
  source              = "./modules/acr"
  registry_name       = "${var.namespace}${random_id.prefix.hex}reg" #only lowercase letters and numbers
  resource_group_name = azurerm_resource_group.launch_rg.0.name
  location            = azurerm_resource_group.launch_rg.0.location
}

module "workload_identity" {
  count               = 1
  source              = "./modules/workload-identity"
  identity_name       = "${var.namespace}-${random_id.prefix.hex}-workload-id"
  resource_group_name = azurerm_resource_group.launch_rg.0.name
  location            = azurerm_resource_group.launch_rg.0.location

  storage_account_id = module.blob.0.storage_account_id
  acr_id             = module.acr.0.azurerm_container_registry_id
}

provider "kubernetes" {
  host                   = module.aks.0.cluster_host
  client_certificate     = base64decode(module.aks.0.cluster_client_certificate)
  client_key             = base64decode(module.aks.0.cluster_client_key)
  cluster_ca_certificate = base64decode(module.aks.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = module.aks.0.cluster_host
    client_certificate     = base64decode(module.aks.0.cluster_client_certificate)
    client_key             = base64decode(module.aks.0.cluster_client_key)
    cluster_ca_certificate = base64decode(module.aks.0.cluster_ca_certificate)
  }
}

locals {
  container_registry_uri  = "${module.acr.0.azurerm_container_registry_uri}${var.namespace}-${random_id.prefix.hex}"
  build_context_store_uri = module.blob.0.build_context_store_uri

  service_account_annotations = {
    "azure.workload.identity/client-id" = module.workload_identity.0.workload_identity_client_id
  }

  agent_labels = {
    "azure.workload.identity/use" = "true"
  }

  queues = join(",", var.queues)

  launch_config = templatefile("${path.module}/templates/launch-config.yaml.tpl", {
    queues                  = local.queues
    entity                  = var.entity
    max_jobs                = var.max_jobs
    max_schedulers          = var.max_schedulers
    uri                     = local.container_registry_uri
    build_context_store_uri = local.build_context_store_uri
  })
}

module "launch_helm" {
  source = "../launch-helm"

  # TF defined inputs
  service_account_annotations = local.service_account_annotations
  azure_storage_access_key    = module.blob.0.storage_access_key
  agent_labels                = local.agent_labels

  # Launch Config 
  launch_config = local.launch_config

  # Required to be defined inputs
  helm_chart_version = var.helm_chart_version
  agent_api_key      = var.agent_api_key
  agent_image        = var.agent_image
  base_url           = var.base_url

  additional_target_namespaces = var.additional_target_namespaces
  
  # Optional 
  helm_repository         = var.helm_repository
  agent_image_pull_policy = var.agent_image_pull_policy
  namespace               = var.k8s_namespace
  volcano                 = var.volcano
  git_creds               = var.git_creds
}
