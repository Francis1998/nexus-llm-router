# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.9.12] — 2024-09-16

### Added
- Extended adapter module with improved error handling
- Added structured logging for latency operations
- New unit tests covering edge cases in provider pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async adapter handler
- Fixed incorrect latency timeout calculation

## [v0.1.0] — 2024-08-12

### Added
- Initial project scaffold with LLM routing core
- Basic router implementation
- README and setup documentation
