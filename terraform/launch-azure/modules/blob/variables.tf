# variable "namespace" {
#   description = "The namespace to be used for naming resources"
#   type        = string
# }

variable "resource_group_name" {
  description = "Resource group in which to create the resources"
  type        = string
}

variable "storage_account_name" {
  description = "The name of the storage account"
  type        = string
}

variable "container_name" {
  description = "The name of the blob container"
  type        = string

}

variable "location" {
  description = "The location of the resources"
  type = string
}