resource "azurerm_storage_account" "storage_account" {
  name                     = var.storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = {
    environment = "staging"
  }
}

resource "azurerm_storage_container" "build_context_store" {
  name                 = var.container_name
  storage_account_name = azurerm_storage_account.storage_account.name
  # container_access_type = "private"
}
