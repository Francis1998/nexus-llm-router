# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.2.12] — 2024-08-30

### Added
- Extended strategy module with improved error handling
- Added structured logging for provider operations
- New unit tests covering edge cases in cost pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async strategy handler
- Fixed incorrect provider timeout calculation

## [v0.1.0] — 2024-08-09

### Added
- Initial project scaffold with LLM routing core
- Basic router implementation
- README and setup documentation
