# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.5.1] — 2024-01-01

### Added
- Extended provider module with improved error handling
- Added structured logging for strategy operations
- New unit tests covering edge cases in router pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async provider handler
- Fixed incorrect strategy timeout calculation

## [v0.1.0] — 2023-11-20

### Added
- Initial project scaffold with LLM routing core
- Basic router implementation
- README and setup documentation
