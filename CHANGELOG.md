# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.5.0] — 2024-11-19

### Added
- Extended cost module with improved error handling
- Added structured logging for provider operations
- New unit tests covering edge cases in latency pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async cost handler
- Fixed incorrect provider timeout calculation

## [v0.1.0] — 2024-11-05

### Added
- Initial project scaffold with LLM routing core
- Basic router implementation
- README and setup documentation
