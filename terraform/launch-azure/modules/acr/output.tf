output "azurerm_container_registry_id" {
  description = "The ID of the Azure Container Registry"
  value       = azurerm_container_registry.acr.id
}

output "azurerm_container_registry_uri" {
  description = "The ID of the Azure Container Registry"
  value       = "https://${azurerm_container_registry.acr.login_server}/"
}

