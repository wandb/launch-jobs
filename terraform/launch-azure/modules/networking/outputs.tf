output "private_subnet" {
  value       = azurerm_subnet.private
  description = "The subnetwork used for W&B"
}