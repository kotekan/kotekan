# GitHub Actions Workflows

- `build_push_docker.yml`: Reusable workflow that conditionally builds and publishes the CPU/GPU Docker images
- `iwyu.yaml`: Manual or scheduled IWYU linter runner
- `main.yml`: Main CI jobs, builds Docker images and runs the kotekan tests
- `manual_docker.yaml`: Manually trigger Docker image builds
- `publish_docker.yml`: (Re)Builds and publishes Docker images for PRs or pushes into `develop`/`chord`
- `scheduled_tasks.yml`: Tasks that may be run by the scheduler on the default branch (currently `develop`)
- `test_kotekan_build.yml`: Reusable workflow that builds kotekan and runs post-build commands
