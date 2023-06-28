variable "resource_group_name" {
  description = "Resource Group name"
  type        = string
}

variable "name" {
  description = "AKS cluster name"
  type        = string
}

variable "location" {
  description = "value"
  type        = string
}

variable "automatic_channel_upgrade" {
  description = "Automatic channel upgrade for AKS cluster"
  type        = string
  default     = "stable"
}

variable "role_based_access_control_enabled" {
  description = "Flag to enable role-based access control (RBAC) for AKS cluster"
  type        = bool
  default     = true
}

variable "http_application_routing_enabled" {
  description = "Flag to enable HTTP application routing for AKS cluster"
  type        = bool
  default     = false
}

variable "default_node_pool_name" {
  description = "Default node pool name"
  type        = string
  default     = "default"
}

variable "azurerm_container_registry_id" {
  description = "ACR ID for the ARCPull Role attachement"
  type = string
}

variable "node_count" {
  description = "Number of nodes in the AKS cluster"
  type        = number
  default     = 3
}

variable "vm_size" {
  description = "The size of the VMs in the AKS cluster"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "cluster_subnet_id" {
  description = "The ID of the subnet where the AKS cluster is deployed"
  type        = string
}

variable "type" {
  description = "The type of AKS node pool"
  type        = string
  default     = "VirtualMachineScaleSets"
}

variable "enable_auto_scaling" {
  description = "Flag to enable auto scaling for AKS node pool"
  type        = bool
  default     = false
}

variable "zones" {
  description = "The availability zones for the AKS cluster"
  type        = list(string)
  default     = ["1", "2"]
}

variable "identity" {
  description = "The identity type for the AKS cluster"
  type        = string
  default     = "SystemAssigned"
}

variable "network_plugin" {
  description = "The network plugin to use for the AKS cluster"
  type        = string
  default     = "azure"
}

variable "network_policy" {
  description = "The network policy to use for the AKS cluster"
  type        = string
  default     = "azure"
}

variable "load_balancer_sku" {
  description = "The SKU of the load balancer for the AKS cluster"
  type        = string
  default     = "standard"
}

variable "tags" {
  description = "Tags to apply to the AKS cluster"
  type        = map(string)
  default = {
    "launch-cluster" = "true"
  }
}
