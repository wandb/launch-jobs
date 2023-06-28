# launch-helm

`launch-helm` is a Terraform module designed to deploy `wandb launch` into a Kubernetes cluster. This module simplifies the process of setting up and managing the deployment of `wandb launch` in your Kubernetes environment.

## Prerequisites

- Terraform v0.12+
- Kubernetes cluster
- kubectl
- Helm v3+

## Usage

To use the `launch-helm` module, add the following code to your Terraform configuration.


1. Create a `.tfvars` file and fill out the required fields and optionally some of the optional fields. Here's an example:

```hcl

```

2. Use the `launch-helm` module in your Terraform configuration:

```hcl
module "launch-helm" {
  source = "path/to/launch-helm"

  subscription_id = var.subscription_id
  helm_repository = var.helm_repository
  helm_chart_version = var.helm_chart_version
  agent_labels = var.agent_labels
  agent_api_key = var.agent_api_key
  agent_image = var.agent_image
  agent_image_pull_policy = var.agent_image_pull_policy
  namespace = var.namespace
  base_url = var.base_url
  additional_target_namespaces = var.additional_target_namespaces
  launch_config = var.launch_config
  volcano = var.volcano
  git_creds = var.git_creds
  service_account_annotations = var.service_account_annotations
  azure_storage_access_key = var.azure_storage_access_key
}
```

3. Run `terraform init` to initialize the Terraform working directory.
4. Run `terraform apply -var-file="your.tfvars"` to apply the changes.

Replace `path/to/launch-helm` with the path to the `launch-helm` module directory, and set the required variables according to your needs.

## Variables

The following variables are available for configuration:

### Required

- `namespace`: The Kubernetes namespace where the `wandb launch` deployment will be created.
- `release_name`: The name of the Helm release.

### Optional

- `chart_version`: The version of the `wandb launch` Helm chart to use. Default: `"latest"`
- `replica_count`: The number of replicas for the `wandb launch` deployment. Default: `1`
- `image_repository`: The repository of the `wandb launch` Docker image. Default: `"wandb/launch"`
- `image_tag`: The tag of the `wandb launch` Docker image. Default: `"latest"`
- `image_pull_policy`: The pull policy for the `wandb launch` Docker image. Default: `"IfNotPresent"`
- `resources`: The resource limits and requests for the `wandb launch` deployment. Default: `{}`
- `node_selector`: The node selector for the `wandb launch` deployment. Default: `{}`
- `tolerations`: The tolerations for the `wandb launch` deployment. Default: `[]`
- `affinity`: The affinity for the `wandb launch` deployment. Default: `{}`

## Outputs

- `helm_release_status`: The status of the Helm release.
- `helm_release_version`: The version of the Helm release.

## Examples

For more examples on how to use the `launch-helm` module, please refer to the [examples](./examples) directory.

## Contributing

We welcome contributions to the `launch-helm` module. Please submit a pull request with your changes, and ensure that your code follows the existing code style and conventions.




## Variables

The following variables are required:

- `subscription_id`: Azure subscription ID
- `helm_repository`: Helm repository URL
- `helm_chart_version`: Helm chart version
- `agent_labels`: Agent labels (map of strings)
- `agent_api_key`: W&B API key
- `agent_image`: Container image to use for the agent
- `agent_image_pull_policy`: Image pull policy for agent image
- `namespace`: Namespace to deploy launch agent into
- `base_url`: W&B api url

The following variables are optional:

- `additional_target_namespaces`: Additional target namespaces that the launch agent can deploy into (list of strings)
- `launch_config`: The literal contents of your launch agent config
- `volcano`: Set to false to disable volcano install (boolean)
- `git_creds`: The contents of a git credentials file
- `service_account_annotations`: Annotations for the wandb service account (map of strings)
- `azure_storage_access_key`: Set to access key for azure storage if using kaniko with azure
