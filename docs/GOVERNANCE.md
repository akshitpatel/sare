# Governance Baseline

## Repository policy
- Primary branch: `main`
- All changes should go through pull requests.
- CI must pass before merge.

## Required status check
Set this required check in repository branch protection settings:
- `C++ Build and Tests`

## Local verification command
Run before pushing:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
