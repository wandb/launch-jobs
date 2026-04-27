resource "azurerm_kubernetes_cluster" "launch_cluster" {
  name                = var.name
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = var.name
  # kubernetes_version  = var.kubernetes_version
  azure_policy_enabled      = true
  workload_identity_enabled = true # this requires an OIDC issuer and we can still use Azure AD pod-managed identities.
  oidc_issuer_enabled       = true

  automatic_channel_upgrade         = var.automatic_channel_upgrade
  role_based_access_control_enabled = var.role_based_access_control_enabled
  http_application_routing_enabled  = var.http_application_routing_enabled

  default_node_pool {
    name                = var.default_node_pool_name
    node_count          = var.node_count
    vm_size             = var.vm_size
    vnet_subnet_id      = var.cluster_subnet_id
    type                = var.type
    enable_auto_scaling = var.enable_auto_scaling
    zones               = var.zones
  }

  identity {
    type = var.identity
  }

  network_profile {
    network_plugin    = var.network_plugin
    network_policy    = var.network_policy
    load_balancer_sku = var.load_balancer_sku
  }

  tags = var.tags
}

# add the role to the identity the kubernetes cluster was assigned
resource "azurerm_role_assignment" "acrpull_role" {
  scope                = var.azurerm_container_registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.launch_cluster.kubelet_identity[0].object_id
}

# resource "azurerm_role_assignment" "blob_owner" {

#   role_definition_name = "Storage Blob Data Owner"
# }

# resource "azurerm_role_assignment" "example" {
#   scope        = data.azurerm_resource_group.example.id
#   principal_id = var.principal_id
# }
# resource "azurerm_" "name" {

# }

