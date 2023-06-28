queues: [${queues}]
entity: ${entity}
max_jobs: ${max_jobs}
max_schedulers: ${max_schedulers}
environment:
  type: azure
registry:
  type: acr
  uri: ${uri}
builder:
  type: kaniko
  build-context-store: ${build_context_store_uri}