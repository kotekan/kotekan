# GitHub Actions Workflows

- `build_push_docker.yaml`: Reusable workflow that conditionally builds and publishes the CPU/GPU/Intel Docker images
- `iwyu.yaml`: Manual or scheduled IWYU linter runner
- `main.yaml`: Main CI jobs, builds Docker images and runs the kotekan tests.
  It runs for `develop`, the long-lived `pilotproxy-dtv-detector` integration
  branch, pull requests targeting either branch, and manual dispatches. The
  first job verifies the vendored PilotProxy core before any build starts;
  both GPU YAML builds also export a pinned runtime bundle and run the
  file-based detector pipeline plus its bit-exact offline verifier. The stable
  `CI success` aggregate is intended to be the required branch-protection check;
  it permits the GPU matrix to be skipped while no self-hosted runner is enabled
  and for fork PRs, which must not execute on the privileged telescope runner.
- `manual_docker.yaml`: Manually trigger Docker image builds
- `publish_docker.yaml`: Rebuilds and publishes Docker images after a PR is merged into `develop`
- `schedule.yaml`: Daily cron entry point (runs from the repository's default branch) that dispatches `scheduled_tasks.yaml` and `iwyu.yaml`
- `scheduled_tasks.yaml`: Tasks dispatched by the scheduler on the repository's default branch. Its self-hosted probe is skipped unless the `ENABLE_SELF_HOSTED_TASKS` repository variable is `true`.
- `test_kotekan_build.yaml`: Reusable workflow that builds kotekan and runs post-build commands
