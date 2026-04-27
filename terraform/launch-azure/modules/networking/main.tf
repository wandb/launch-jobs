resource "azurerm_virtual_network" "default" {
  name                = "${var.namespace}-vpc"
  location            = var.location
  resource_group_name = var.resource_group_name

  address_space = [var.network_cidr]

  tags = var.tags
}

resource "azurerm_subnet" "private" {
  name                 = "${var.namespace}-private"
  resource_group_name  = var.resource_group_name
  address_prefixes     = [var.network_private_subnet_cidr]
  virtual_network_name = azurerm_virtual_network.default.name

#   service_endpoints = [
#     # "Microsoft.Sql",
#     # "Microsoft.Storage",
#     # "Microsoft.KeyVault"

#   ]
}

# resource "azurerm_subnet" "public" {
#   name                 = "${var.namespace}-public"
#   resource_group_name  = var.resource_group_name
#   address_prefixes     = [var.network_public_subnet_cidr]
#   virtual_network_name = azurerm_virtual_network.default.name
# }

resource "azurerm_subnet" "redis" {
  name                 = "${var.namespace}-redis"
  resource_group_name  = var.resource_group_name
  address_prefixes     = [var.network_redis_subnet_cidr]
  virtual_network_name = azurerm_virtual_network.default.name
}
