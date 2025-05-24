# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.1.18] — 2025-05-24

### Added
- Extended router module with improved error handling
- Added structured logging for latency operations
- New unit tests covering edge cases in adapter pipeline

### Changed
- Refactored retry logic to use exponential backoff with jitter
- Improved type annotations across core modules
- Updated dependency pins to latest stable versions

### Fixed
- Resolved race condition in async router handler
- Fixed incorrect latency timeout calculation

## [v0.1.0] — 2025-04-26

### Added
- Initial project scaffold with LLM routing core
- Basic router implementation
- README and setup documentation
