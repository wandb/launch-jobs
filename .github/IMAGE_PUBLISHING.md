# Image publishing credentials

The `Publish changed images` workflow publishes to the `wandb` namespace on Docker Hub. It uses these GitHub Actions repository secrets:

- `DOCKERHUB_USERNAME`: a W&B-managed Docker Hub service account
- `DOCKERHUB_TOKEN`: an access token for that account with permission to push to the `wandb` repositories listed in `.github/image-builds.json`

Do not use a personal account or a Docker Hub password.

## Initial setup

A Docker Hub organization owner must create or select the service account and create its access token. GitHub does not expose existing secret values, and a workflow must not create its own long-lived publishing credential.

A GitHub repository administrator can then store the credentials without putting them in process arguments or shell history. Each command securely prompts for its value:

```bash
gh secret set DOCKERHUB_USERNAME --repo wandb/launch-jobs
gh secret set DOCKERHUB_TOKEN --repo wandb/launch-jobs
```

The repository had a `DOCKERHUB_TOKEN` secret before this migration, but its value cannot be inspected. Rotate it rather than assuming it belongs to the service account used by CircleCI.

## Initial rollout

Push-triggered publishing is disabled unless the repository variable `IMAGE_PUBLISHING_ENABLED` is exactly `true`. Manual runs remain available while publishing is disabled, preventing the merge that adds the image manifest from publishing all images immediately.

After configuring the secrets:

1. Run `Publish changed images` manually for `job_hello_world`.
2. Run it manually for `job_inspect_ai_evals_api_model` to exercise the shared context.
3. Confirm both SHA and branch tags, digests, and multi-platform manifests in Docker Hub.
4. Enable push-triggered publishing:

   ```bash
   gh variable set IMAGE_PUBLISHING_ENABLED --repo wandb/launch-jobs --body true
   ```

Disable automatic publishing without affecting manual runs by setting the variable to `false`.

## Rotation

1. Create a new access token for the service account in Docker Hub. Do not revoke the old token yet.
2. Replace `DOCKERHUB_TOKEN` with `gh secret set` as shown above. Update `DOCKERHUB_USERNAME` too if the account changed.
3. Run `Publish changed images` manually for one normal image, such as `job_hello_world`.
4. Run it for one shared-context image, such as `job_inspect_ai_evals_api_model`.
5. Confirm the SHA and branch tags, digests, and multi-platform manifests in Docker Hub.
6. Revoke the old Docker Hub token.

During the CircleCI parity period, determine whether CircleCI uses the same token before revoking it. If it does, either update CircleCI as well or retain the old token until CircleCI is disabled.

Repository administrators can verify secret presence and rotation timestamps, but not secret values:

```bash
gh api repos/wandb/launch-jobs/actions/secrets \
  --jq '.secrets[] | select(.name | startswith("DOCKERHUB_")) | [.name, .updated_at] | @tsv'
```
