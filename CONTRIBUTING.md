# Contributing

Contributions that improve reproducibility, portability, statistical correctness, or documentation are welcome.

## Before opening a pull request

1. Run `python src/smoke_test.py`.
2. Run `python -m unittest discover -s tests -v`.
3. Keep RiceChem rows, model weights, predictions, credentials, infrastructure identifiers, and student information out of the repository.
4. Separate measured results from estimates and preserve serving configuration, concurrency, and hardware provenance.
5. Do not strengthen a scientific claim without an aggregate receipt and a documented comparison protocol.

For larger experimental changes, open an issue describing the hypothesis, frozen evaluation set, intended statistical comparison, and expected compute requirements before launching runs.
