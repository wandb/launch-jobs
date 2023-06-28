output "storage_access_key" {
  value       = azurerm_storage_account.storage_account.primary_access_key
  description = "The primary access key for the storage account associated with the container registry."
}

output "storage_account_id" {
  description = "The ID of the storage container"
  value       = azurerm_storage_account.storage_account.id
}

output "build_context_store_uri" {
  description = "The URL of the build context store container"
  value       = "${azurerm_storage_account.storage_account.primary_blob_endpoint}${azurerm_storage_container.build_context_store.name}"
}