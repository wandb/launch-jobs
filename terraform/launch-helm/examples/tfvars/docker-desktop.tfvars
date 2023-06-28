# Required
# helm_repository = "https://wandb.github.io/helm-charts"
helm_chart_version = "0.6.0"

agent_image   = "benwandb415/agent:azure"
base_url      = "http://host.docker.internal:8080"
agent_api_key = "local-9ff313e38a5575944156ef8501444c9c71792f07" # not a secret
launch_config = <<EOF
  queues: [docker-desktop]
  entity: docker-desktop
  max_jobs: 5
  max_schedulers: 8
  environment:
    type: local
  registry:
    type:
    uri:
  builder:
    type: noop
    build-context-store: 
EOF

#optional
agent_labels = {
  app = "wandb"
}
# agent_image_pull_policy = "Always"
# namespace = "wandb"
additional_target_namespaces = ["default"]

# volcano = false
# git_creds =
# service_account_annotations = {}
# azure_storage_access_key = 
# subscription_id =
