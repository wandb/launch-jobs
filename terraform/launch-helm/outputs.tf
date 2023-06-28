output "helm_release_status" {
  value       = helm_release.launch_agent.status
  description = "The status of the Helm release"
}

output "helm_release_chart" {
  value       = helm_release.launch_agent.chart
  description = "The Helm release"
}
