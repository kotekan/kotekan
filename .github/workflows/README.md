# GitHub Actions Workflows

- `build_push_docker.yaml`: Reusable workflow that conditionally builds and publishes the CPU/GPU/Intel Docker images
- `iwyu.yaml`: Manual or scheduled IWYU linter runner
- `main.yaml`: Main CI jobs, builds Docker images and runs the kotekan tests.
  It runs for `develop`, the long-lived `pilotproxy-dtv-detector` integration
  branch, pull requests targeting either branch, and manual dispatches. The
  first job verifies the vendored PilotProxy core before any build starts;
  both GPU YAML builds also export a pinned runtime bundle and run the
  file-based detector pipeline plus its bit-exact offline verifier.
- `manual_docker.yaml`: Manually trigger Docker image builds
- `publish_docker.yaml`: (Re)Builds and publishes Docker images for PRs or pushes into `develop`
- `schedule.yaml`: Daily cron entry point (runs from the default branch) that dispatches `scheduled_tasks.yaml` and `iwyu.yaml`
- `scheduled_tasks.yaml`: Tasks dispatched by the scheduler on the default branch (`develop`)
- `test_kotekan_build.yaml`: Reusable workflow that builds kotekan and runs post-build commands
