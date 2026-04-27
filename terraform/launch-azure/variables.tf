variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "namespace" {
  description = "Unique namespace name for your resources"
  type        = string
}

variable "location" {
  description = "value"
  type        = string
}

variable "create_resource_group" {
  description = "Flag to create a RG"
  type        = bool
}

variable "create_aks_cluster" {
  description = "Flag to create a aks cluster"
  type        = bool
}

# launch-helm variables
variable "helm_repository" {
  description = "Helm repository URL"
  type        = string
  default     = "https://wandb.github.io/helm-charts"
}

variable "helm_chart_version" {
  description = "Helm chart version"
  type        = string
}

variable "agent_api_key" {
  description = "W&B API key"
  type        = string
}

variable "agent_image" {
  description = "Container image to use for the agent"
  type        = string
}

variable "agent_image_pull_policy" {
  description = "Image pull policy for agent image"
  type        = string
  default     = "Always"
}

variable "agent_resources" {
  description = "Resources block for the agent spec"
  type = object({
    limits = map(string)
  })
  default = {
    limits = {
      cpu    = "1"
      memory = "1Gi"
    }
  }
}

variable "k8s_namespace" {
  description = "Namespace to deploy launch agent into"
  type        = string
  default     = "wandb"
}

variable "base_url" {
  description = "W&B api url"
  type        = string
}

variable "volcano" {
  description = "Set to false to disable volcano install"
  type        = bool
  default     = false
}

variable "git_creds" {
  description = "The contents of a git credentials file"
  type        = string
  default     = ""
}

variable "additional_target_namespaces" {
  description = "Additional target namespaces that the launch agent can deploy into"
  type        = list(string)
  default     = ["wandb", "default"]
}

# launch_config varibles

variable "queues" {
  description = "The list of queues to be used"
  type        = list(string)
  # default     = ["azure-demo-queue"]
}

variable "entity" {
  description = "The list of entities to be used"
  type        = string
  # default     = "azure-team"
}

variable "max_jobs" {
  description = "The maximum number of jobs"
  type        = string
  default     = "5"
}

variable "max_schedulers" {
  description = "The maximum number of schedulers"
  type        = string
  default     = "8"
}

