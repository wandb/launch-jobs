variable "namespace" {
  type        = string
  description = "Friendly name prefix used for tagging and naming Azure resources."
}

variable "resource_group_name" {
  type        = string
  description = "The name of the resource group in which to create the network."
}

variable "location" {
  type        = string
  description = "Specifies the supported Azure location where the resource exists."
}

variable "network_cidr" {
  default     = "10.10.0.0/16"
  type        = string
  description = "(Optional) CIDR range for network"
}

variable "network_public_subnet_cidr" {
  type        = string
  default     = "10.10.0.0/24"
  description = "(Optional) Subnet CIDR range for W&B"
}

variable "network_private_subnet_cidr" {
  type        = string
  default     = "10.10.1.0/24"
  description = "(Optional) Subnet CIDR range for frontend"
}

variable "network_redis_subnet_cidr" {
  default     = "10.10.2.0/24"
  type        = string
  description = "(Optional) Subnet CIDR range for Redis"
}

variable "network_database_subnet_cidr" {
  default     = "10.10.3.0/24"
  type        = string
  description = "The CIDR range of the database subnetwork."
}

variable "network_allow_range" {
  default     = "*"
  type        = string
  description = "(Optional) Network range to allow access to TFE"
}

variable "tags" {
  default     = {}
  type        = map(string)
  description = "Map of tags for resource"
}
