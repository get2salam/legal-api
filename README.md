# Legal API ⚖️

FastAPI REST backend for legal case law search. A generic, jurisdiction-agnostic API for building legal research applications.

## Features

- 🔍 **Full-text search** with relevance ranking
- 🏛️ **Filter by court**, date range, citation
- 📄 **Pagination** for large result sets
- 🔐 **API key authentication** (optional)
- 📊 **Statistics endpoints** for analytics
- 🚀 **Fast** — async FastAPI with connection pooling

## Quick Start

```bash
# Clone and install
git clone https://github.com/get2salam/legal-api.git
cd legal-api
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Run
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Search

```http
GET /api/v1/search?q=constitutional+rights&court=supreme&year=2024
```

**Parameters:**
- `q` (required): Search query
- `court`: Filter by court name
- `year`: Filter by year
- `date_from`, `date_to`: Date range
- `page`, `per_page`: Pagination

**Response:**
```json
{
  "total": 142,
  "page": 1,
  "per_page": 20,
  "results": [
    {
      "id": "case_001",
      "title": "Smith v. State",
      "citation": "2024 SC 445",
      "court": "Supreme Court",
      "date": "2024-03-15",
      "snippet": "...constitutional rights were violated...",
      "relevance": 0.95
    }
  ]
}
```

### Get Case

```http
GET /api/v1/cases/{case_id}
```

**Response:**
```json
{
  "id": "case_001",
  "title": "Smith v. State",
  "citation": "2024 SC 445",
  "court": "Supreme Court",
  "date": "2024-03-15",
  "judges": ["Justice A", "Justice B"],
  "headnote": "Brief summary...",
  "text": "Full judgment text..."
}
```

### List Courts

```http
GET /api/v1/courts
```

### Statistics

```http
GET /api/v1/stats
GET /api/v1/stats/courts
GET /api/v1/stats/years
```

## Configuration

Create a `.env` file:

```env
# Database
DATABASE_URL=sqlite:///./legal.db
# Or PostgreSQL: postgresql://user:pass@localhost/legal

# API Settings
API_TITLE=Legal Case Law API
API_VERSION=1.0.0
PER_PAGE_DEFAULT=20
PER_PAGE_MAX=100

# Authentication (optional)
API_KEY_ENABLED=false
API_KEY=your-secret-key

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

## Data Loading

Load your case law data:

```bash
# From JSON files
python scripts/load_data.py --source data/cases/

# From JSONL
python scripts/load_data.py --source data/cases.jsonl

# From CSV
python scripts/load_data.py --source data/cases.csv
```

### Expected Data Format

```json
{
  "id": "unique_id",
  "title": "Case Title",
  "citation": "2024 ABC 123",
  "court": "Court Name",
  "date": "2024-01-15",
  "judges": ["Judge 1", "Judge 2"],
  "headnote": "Brief summary",
  "text": "Full judgment text"
}
```

## Project Structure

```
legal-api/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── database.py          # Database connection
├── routers/
│   ├── search.py        # Search endpoints
│   ├── cases.py         # Case CRUD
│   └── stats.py         # Statistics
├── services/
│   ├── search.py        # Search logic
│   └── ranking.py       # Relevance ranking
├── scripts/
│   └── load_data.py     # Data loading utility
└── tests/
    └── test_api.py      # API tests
```

## Deployment

### Docker

```bash
docker build -t legal-api .
docker run -p 8000:8000 legal-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db/legal
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=legal
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE)

---

Built with ⚖️ by [Abdul Salam](https://github.com/get2salam)
