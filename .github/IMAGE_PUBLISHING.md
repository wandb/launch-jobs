# Image publishing

The `Publish changed images` workflow builds each selected image once in Google
Artifact Registry (GAR), then mirrors the verified multi-platform manifest to
Docker Hub.

- Canonical image: `us-docker.pkg.dev/wandb-production/public/wandb/<image>`
- Compatibility mirror: `docker.io/wandb/<image>`
- Platforms: `linux/amd64` and `linux/arm64`
- Tags: the full Git commit SHA and the branch name (`main` for automatic runs)

GAR is canonical so a Docker Hub outage or expired Docker Hub token does not lose
the built artifact. Rerunning the workflow reuses an existing, revision-verified
GAR SHA image and retries the mirror without rebuilding it.

## GAR authentication

GitHub Actions authenticates to GAR with short-lived Google Workload Identity
Federation credentials. The workflow requires these repository secrets:

- `CI_WORKLOAD_IDENTITY_PROVIDER`
- `CI_WORKLOAD_IDENTITY_SERVICE_ACCOUNT`

They are managed by Terraform in `wandb/core/terraform/ci/global`. `launch-jobs`
uses the existing `tf-github-ci` service account and its writer access to the
`wandb-production/public` Artifact Registry repository. Do not create or store a
JSON service-account key in GitHub.

The workflow needs `id-token: write` only in the image-building job. Pull requests
do not run the publishing workflow.

## Docker Hub credentials

Docker Hub mirroring uses these GitHub Actions repository secrets:

- `DOCKERHUB_USERNAME`: a W&B-managed Docker Hub organization or service account
- `DOCKERHUB_TOKEN`: an access token for that account with permission to push to
  the `wandb` repositories listed in `.github/image-builds.json`

Do not use a personal account or Docker Hub password. Scope the account/token to
the 17 repositories where Docker Hub supports that restriction.

A Docker Hub organization owner must create or select the service account and
create its access token. GitHub does not expose existing secret values. Rotate
the old token rather than assuming it belongs to the account previously used by
CircleCI.

A GitHub repository administrator can store the credentials without putting them
in process arguments or shell history. Each command securely prompts for its
value:

```bash
gh secret set DOCKERHUB_USERNAME --repo wandb/launch-jobs
gh secret set DOCKERHUB_TOKEN --repo wandb/launch-jobs
```

## Initial rollout

Push-triggered publishing is disabled unless the repository variable
`IMAGE_PUBLISHING_ENABLED` is exactly `true`. Manual runs remain available while
publishing is disabled.

After the WIF infrastructure change has applied and the Docker Hub secrets are
configured:

1. Run `Publish changed images` manually for `job_hello_world`.
2. Confirm its SHA and `main` tags exist in both GAR and Docker Hub and contain
   `linux/amd64` and `linux/arm64` manifests.
3. Run it manually for `job_inspect_ai_evals_api_model` to exercise the shared
   build context.
4. Rerun one image and confirm the workflow reuses its canonical GAR SHA image.
5. Enable push-triggered publishing:

   ```bash
   gh variable set IMAGE_PUBLISHING_ENABLED --repo wandb/launch-jobs --body true
   ```

6. Merge a narrowly scoped image change and confirm only the affected image is
   published.
7. After at least one successful push-triggered run, remove the CircleCI config,
   disable the CircleCI project/integration, and revoke its publishing token.

Disable automatic publishing without affecting manual runs by setting the
variable to `false`.

## Failure and retry behavior

- If the GAR build fails, no Docker Hub image is published.
- If Docker Hub login or mirroring fails, the workflow fails but leaves the
  canonical GAR SHA image available. Rerun after fixing Docker Hub; the build is
  reused.
- If `main` advances while an image is building, immutable SHA tags remain but
  both registries' mutable `main` tags are skipped.
- Per-image concurrency prevents older in-flight runs from promoting over newer
  runs.

The mirror validation compares sorted platform descriptors rather than requiring
top-level index digests to match, because registries may rewrite index metadata.

## Docker Hub credential rotation

1. Create a new access token for the service account. Do not revoke the old token
   yet.
2. Replace `DOCKERHUB_TOKEN` with `gh secret set` as shown above. Update
   `DOCKERHUB_USERNAME` too if the account changed.
3. Run the workflow manually for `job_hello_world`.
4. Confirm the GAR image was reused and the Docker Hub SHA and `main` tags match
   its platform manifests.
5. Revoke the old Docker Hub token.

During the CircleCI parity period, determine whether CircleCI uses the same token
before revoking it. If it does, retain the old token until CircleCI is disabled.

Repository administrators can verify secret presence and rotation timestamps,
but not secret values:

```bash
gh api repos/wandb/launch-jobs/actions/secrets \
  --jq '.secrets[] | select(.name | test("^(CI_WORKLOAD_IDENTITY_|DOCKERHUB_)")) | [.name, .updated_at] | @tsv'
```
