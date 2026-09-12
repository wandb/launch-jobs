resource "azurerm_container_registry" "acr" {
  name                      = var.registry_name
  resource_group_name       = var.resource_group_name
  location                  = var.location
  sku                       = "Basic"
  admin_enabled             = false
  network_rule_set          = []
  quarantine_policy_enabled = false
  trust_policy {
    enabled = false
  }
  retention_policy {
    days    = 7
    enabled = false
  }

  # tags = {
  #   Environment = "Production"
  # }
}
