# Testing Documentation

This directory contains all tests for the AI Chatbot Backend application.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_schema.py      # Schema generation and validation tests
│   ├── test_database.py    # Database operations tests
│   └── test_workflow.py    # Workflow logic tests
└── integration/             # Integration tests (future)
```

## Running Tests

### Quick Start

Run all unit tests:
```bash
./run_tests.sh
```

Or using pytest directly:
```bash
pytest tests/unit/ -v
```

### Run Specific Test Files

```bash
# Test schema module
pytest tests/unit/test_schema.py -v

# Test database module
pytest tests/unit/test_database.py -v

# Test workflow module
pytest tests/unit/test_workflow.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/unit/test_schema.py::TestRequestSchema -v

# Run specific test function
pytest tests/unit/test_schema.py::TestRequestSchema::test_schema_has_required_fields -v
```

### Coverage Reports

Generate coverage report:
```bash
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term-missing
```

View HTML coverage report:
```bash
open htmlcov/index.html
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions and classes in isolation using mocks.

**test_schema.py** - Tests for schema generation:
- Dynamic schema creation from config
- Field validation
- PII field exclusion
- Schema-to-dict conversion

**test_database.py** - Tests for database operations:
- PostgreSQL client initialisation
- Fuzzy candidate pre-filter (pg_trgm)
- Vector similarity search (pgvector)
- Request saving

**test_workflow.py** - Tests for workflow logic:
- Workflow initialization
- PII parsing and formatting
- Node functions (collect_pii, confirm_pii, etc.)
- Routing logic
- State management

### Integration Tests (`tests/integration/`)

Test multiple components working together (future implementation).

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `mock_llm` - Mock LLM for testing
- `mock_embedding_model` - Mock embedding model
- `sample_request_data` - Sample request with PII
- `sample_request_data_no_pii` - Sample request without PII
- `sample_embedding` - Sample embedding vector
- `sample_conversation_state` - Sample conversation state

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_is_being_tested>`

### Example Test Structure

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

class TestMyFeature:
    """Test MyFeature functionality."""
    
    @pytest.fixture
    def my_fixture(self):
        """Create test fixture."""
        return MagicMock()
    
    def test_basic_functionality(self, my_fixture):
        """Test basic functionality."""
        result = my_function(my_fixture)
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, my_fixture):
        """Test async functionality."""
        result = await my_async_function(my_fixture)
        assert result == expected_value
```

### Async Tests

Use `@pytest.mark.asyncio` decorator for async tests:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Mocking

Use `unittest.mock` for mocking:

```python
from unittest.mock import MagicMock, AsyncMock, patch

# Mock synchronous function
mock_obj = MagicMock()
mock_obj.method.return_value = "result"

# Mock async function
mock_obj = AsyncMock()
mock_obj.method.return_value = "result"

# Patch module
with patch('module.function') as mock_func:
    mock_func.return_value = "result"
    # Test code here
```

## Test Coverage Goals

- **Unit Tests**: Aim for >80% code coverage
- **Critical Paths**: 100% coverage for:
  - PII handling
  - Duplicate detection
  - Data validation
  - Routing logic

## Continuous Integration

Tests should be run:
- Before committing code
- In CI/CD pipeline
- Before deploying to production

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running tests from the project root:
```bash
cd ai-chatbot-backend
pytest tests/unit/
```

### Async Test Failures

Ensure `pytest-asyncio` is installed:
```bash
uv add --dev pytest-asyncio
```

### Mock Not Working

Verify the import path in the patch decorator matches the actual import in the code:
```python
# If code imports: from src.core.database import MongoDBClient
# Then patch: @patch('src.core.database.MongoDBClient')
```

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Share common setup code via fixtures
3. **Mock External Dependencies**: Don't make real API calls or database connections
4. **Test Edge Cases**: Test both success and failure scenarios
5. **Clear Assertions**: Use descriptive assertion messages
6. **Fast Tests**: Unit tests should run quickly (<1s each)
7. **Readable Tests**: Tests are documentation - make them clear

## Future Enhancements

- [ ] Add integration tests
- [ ] Add E2E tests for API endpoints
- [ ] Add performance tests
- [ ] Add load tests
- [ ] Add security tests
- [ ] Increase coverage to >90%