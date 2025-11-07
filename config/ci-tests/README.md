# Kotekan CI YAML Tests

Runs Kotekan tests against multiple YAML configuration files with configurable timeouts.

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
