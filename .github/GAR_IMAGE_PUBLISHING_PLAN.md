# GAR-backed image publishing plan

## Goal

Make Google Artifact Registry (GAR) the canonical destination for images built by
`wandb/launch-jobs`, while retaining Docker Hub as a compatibility mirror. GitHub
Actions will authenticate to GAR with short-lived Workload Identity Federation
(WIF) credentials. Docker Hub remains the only destination requiring a static
credential.

This follows the model used by `wandb/bufstream`: build or reuse a verified image
in GAR, then copy its manifest to Docker Hub rather than rebuilding separately for
each registry.

## Proposed registry layout

Use the existing public GAR repository:

```text
us-docker.pkg.dev/wandb-production/public/wandb/<image>:<tag>
```

Keep the existing Docker Hub names and tags:

```text
docker.io/wandb/<image>:<tag>
```

For example:

```text
us-docker.pkg.dev/wandb-production/public/wandb/job_hello_world:<git-sha>
docker.io/wandb/job_hello_world:<git-sha>
```

Each selected image receives:

- an immutable full Git SHA tag;
- the mutable `main` tag, promoted only after confirming that the remote `main`
  ref still points to the workflow's commit.

The 17 image names and build contexts remain defined in
`.github/image-builds.json`.

## Phase 1: Extend the shared CI WIF access

This work belongs in the infrastructure repository that manages W&B's GitHub WIF
bindings and GitHub Actions secrets (currently `wandb/core/terraform/ci/global`).
Follow the existing Bufstream setup rather than creating another service account:

1. Add the exact GitHub OIDC subject for `wandb/launch-jobs` on `main` to the WIF
   binding for the existing `tf-github-ci` service account. Use a subject principal
   restricted to `repo:wandb/launch-jobs:ref:refs/heads/main`, rather than a
   repository-wide principal set.
2. Reuse the service account's existing Artifact Registry writer access to the
   `wandb-production/public` repository. Do not add new project-wide roles or
   registry permissions for `launch-jobs`.
3. Manage these repository secrets through Terraform, using the same WIF provider
   and service-account values as Bufstream:
   - `CI_WORKLOAD_IDENTITY_PROVIDER`
   - `CI_WORKLOAD_IDENTITY_SERVICE_ACCOUNT`
4. Confirm that forks, pull requests, and workflows on other branches cannot
   obtain an identity token accepted for service-account impersonation.

Acceptance criteria:

- A workflow on `main` can authenticate and push a disposable test tag.
- A workflow from any other branch/ref cannot impersonate the shared service
  account as `wandb/launch-jobs`.
- No new service account, JSON service-account key, or broader GAR grant is
  created.

## Phase 2: Add GAR authentication and canonical publishing

In this repository:

1. Add a local composite action equivalent to Bufstream's
   `.github/actions/gar-docker-login/action.yml`:
   - use `google-github-actions/auth` with WIF;
   - install/configure `gcloud`;
   - configure Docker for `us-docker.pkg.dev`;
   - pin third-party actions to full commit SHAs.
2. Add `id-token: write` to `.github/workflows/publish-images.yml`, retaining
   `contents: read`.
3. For each selected matrix entry, derive both its GAR and Docker Hub references.
4. Build the multi-platform image once and push the SHA tag to GAR. Preserve the
   current `linux/amd64` and `linux/arm64` platforms and `provenance: false`
   behavior.
5. Before building, inspect the GAR SHA tag:
   - if absent, build and push it;
   - if present, reuse it so retries are idempotent;
   - fail rather than overwrite if its recorded source revision does not match the
     requested commit.
6. Validate the GAR image after build/reuse:
   - both expected platforms exist;
   - the image carries `org.opencontainers.image.revision=<git-sha>` on the index
     and/or platform manifests;
   - the source revision is unambiguous.
7. Continue using the GitHub Actions build cache initially. Moving cache state to
   GAR is a separate optimization and should not block the credential migration.

Acceptance criteria:

- Both a normal build context and the shared `inspect_ai_evals` context publish a
  valid GAR SHA image.
- Rerunning the same commit reuses the canonical GAR image safely.
- No Docker Hub credentials are needed to complete the canonical build.

## Phase 3: Mirror verified manifests to Docker Hub

1. Rotate/provision a W&B-managed Docker Hub organization or service-account
   token scoped to the 17 existing repositories.
2. Store its identity as `DOCKERHUB_USERNAME` and its token as
   `DOCKERHUB_TOKEN`. Do not reuse Bufstream's repository-scoped token.
3. Authenticate to Docker Hub only after the GAR image has been validated.
4. Copy the canonical GAR manifest with `docker buildx imagetools create`; do not
   rebuild the image for Docker Hub.
5. Validate the mirror by comparing the sorted platform manifest descriptors
   between GAR and Docker Hub. Do not require the top-level index digest to be
   identical because registries may rewrite index metadata.
6. Fail the workflow if mirroring fails, while clearly reporting that the
   canonical GAR artifact is available for a retry.
7. After both SHA references are valid, re-check the remote `main` ref and promote
   the GAR and Docker Hub `main` tags only if it still matches `GITHUB_SHA`.
8. Keep per-image concurrency so an older run cannot replace a newer image's
   mutable tag.

Acceptance criteria:

- Docker Hub's SHA tag contains exactly the same platform manifests as GAR.
- A retry after a Docker Hub outage reuses GAR and performs only the mirror step.
- An obsolete workflow run cannot move either registry's `main` tag backwards.

## Phase 4: Controlled rollout

Use the existing `IMAGE_PUBLISHING_ENABLED` gate.

1. Keep push-triggered publishing disabled.
2. Manually publish `job_hello_world` and verify its GAR and Docker Hub SHA and
   `main` tags.
3. Manually publish `job_inspect_ai_evals_api_model` to exercise the shared build
   context.
4. Rerun one of those jobs to verify idempotent GAR reuse.
5. Temporarily test failure recovery by withholding or replacing the Docker Hub
   credential: confirm the GAR image succeeds first, then restore the credential
   and rerun to mirror without rebuilding.
6. Set `IMAGE_PUBLISHING_ENABLED=true` and merge a narrowly scoped image change.
7. Confirm path selection builds only the affected image(s), and observe at least
   one successful push-triggered run before retiring CircleCI.

Record the test run URLs, image references, manifest comparison, and credential
owners in `.github/IMAGE_PUBLISHING.md` or the migration PR.

## Phase 5: Decommission CircleCI

After the rollout criteria pass:

1. Delete `.circleci/config.yml`.
2. Update `.github/IMAGE_PUBLISHING.md` to describe GAR as canonical, Docker Hub
   as a mirror, credential rotation, and recovery from mirror failures.
3. Disable the CircleCI project and remove its GitHub integration/webhook.
4. Remove any required CircleCI status checks from repository rules or branch
   protection.
5. Revoke CircleCI's Docker Hub credential/context after verifying that no other
   project consumes it.
6. Verify a subsequent `main` change produces no CircleCI pipeline or CircleCI
   commit statuses.

## Operational behavior and rollback

- **GAR unavailable:** fail before attempting Docker Hub; no unverified image is
  published.
- **Docker Hub unavailable or token expired:** fail after GAR publication. Rerun
  later; the workflow reuses the immutable GAR SHA image and retries the mirror.
- **Branch advances during a build:** retain immutable SHA tags but skip both
  mutable `main` promotions.
- **Selector bug:** manually dispatch one image or `all`; the manifest remains the
  source of truth.
- **Rollback during rollout:** set `IMAGE_PUBLISHING_ENABLED=false`. Manual runs
  remain available for diagnosis. Keep CircleCI enabled until the end-to-end push
  criteria pass.

## Follow-up improvements

These are useful but not required to remove CircleCI:

- Add retention rules for old SHA tags and build caches, preserving any tags still
  referenced by users.
- Add secret scanning before canonical publication, following the reusable
  publisher in `wandb/core`.
- Pin every third-party action in this repository to a full commit SHA and let
  Renovate manage updates.
- Evaluate moving consumers from Docker Hub to public GAR; once compatibility no
  longer requires Docker Hub, remove the remaining static publishing credential.
- Add scheduled monitoring that verifies representative `main` manifests exist in
  both registries and still match by platform digest.

## External dependencies and owners

Before implementation starts, assign owners for:

- the `wandb/core` Terraform/WIF binding and repository-secret change;
- GAR retention policy (the shared account already has repository IAM);
- creation and rotation of the Docker Hub organization/service-account token;
- repository administration needed to disable CircleCI and remove status checks.

The infrastructure change must land before the repository workflow can be tested;
CircleCI removal is deliberately the final change.
