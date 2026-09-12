variable "registry_name" {
  description = "The name of the container registry."
  type        = string
}

variable "resource_group_name" {
  description = "The name of the resource group where the container registry should be created."
  type        = string
}

variable "location" {
  description = "The location/region where the container registry should be created."
  type        = string
}