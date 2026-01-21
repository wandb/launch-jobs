output "workload_identity_client_id" {
  description = "The Client ID of the user-assigned managed identity"
  value       = "${azurerm_user_assigned_identity.workload_identity.client_id}"
}
