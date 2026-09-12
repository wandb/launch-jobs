# Infra variables
subscription_id       = "636d899d-..."
namespace             = "blasczyk"
location              = "eastus"
create_resource_group = true
create_aks_cluster    = true

# launch-helm variables
# helm_repository       = https://wandb.github.io/helm-charts
helm_chart_version      = "0.6.0"
agent_api_key           = "local-46fa1..."
agent_image             = "benwandb415/agent:azure"
# agent_image_pull_policy = "Always"

# agent_resources = {
#   limits = {
#     cpu    = "1"
#     memory = "1Gi"
#   }
# }

# k8s_namespace = "wandb"
base_url  = "https://blasczyk-azure.sandbox-azure.wandb.ml"
# volcano   = false
# git_creds = ""

additional_target_namespaces = ["default"]

# launch_config variables
queues = ["azure-demo-queue"]
entity = "azure-team"
# max_jobs       = "5"
# max_schedulers = "8"


