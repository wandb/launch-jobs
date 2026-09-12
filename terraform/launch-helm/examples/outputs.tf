output "helm_release_status" {
  value       = module.launch_helm.helm_release_status
  description = "The status of the Helm release"
}
