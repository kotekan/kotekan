# Kotekan CI YAML Tests

Config files in this directory are intended to be run as part of CI. The `run_tests.sh` script runs Kotekan tests against all YAML configuration files in a provided directory: everything in the `batch` directory will be run in one of several CI jobs, while configs in `standalone` are intended for tests that need more tailoring (e.g. take a while, or expect specific compile flags).

## Usage

```bash
./run_tests.sh <kotekan_binary> <timeout_duration> <test_config_dir>
```

**Example:**
```bash
./run_tests.sh ./build-2404/kotekan/kotekan 2m config/ci-tests/batch
```

## Output Exampe

```
======================================
Test Summary
======================================
Total tests: 10
Passed: 8
Failed: 1
Timed out: 1

Failed tests:
  - config/ci-tests/batch/test_broken.yaml (exit code: 1)
```
