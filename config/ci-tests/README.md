# Kotekan CI YAML Tests

Config files in this directory are intended to be run as part of CI. The `run_tests.sh` script runs Kotekan tests against all YAML configuration files. Individual tests can also be run as part of CI if desired.

## Usage

```bash
./run_tests.sh <kotekan_binary> <timeout_duration> <test_config_dir>
```

**Example:**
```bash
./run_tests.sh ./build-2404/kotekan/kotekan 2m config/ci-tests
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
  - config/ci-tests/test_broken.yaml (exit code: 1)
```
