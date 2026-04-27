variable "helm_repository" {
  description = "Helm repository URL"
  type        = string
  default     = "https://wandb.github.io/helm-charts"
}

variable "helm_chart_version" {
  description = "Helm chart version"
  type        = string
}

variable "agent_labels" {
  description = "Agent labels"
  type        = map(string)
  default = {
    "azure.workload.identity/use" = ""
  }
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

variable "namespace" {
  description = "Namespace to deploy launch agent into"
  type        = string
  default     = "wandb"
}

variable "base_url" {
  description = "W&B api url"
  type        = string
}

variable "additional_target_namespaces" {
  description = "Additional target namespaces that the launch agent can deploy into"
  type        = list(string)
  default     = ["wandb", "default"]
}

variable "launch_config" {
  description = "The literal contents of your launch agent config"
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

variable "service_account_annotations" {
  description = "Annotations for the wandb service account"
  type        = map(string)
  default = {
    "iam.gke.io/gcp-service-account"    = ""
    "azure.workload.identity/client-id" = ""
  }
}

variable "azure_storage_access_key" {
  description = "Set to access key for azure storage if using kaniko with azure"
  type        = string
  default     = ""
}
