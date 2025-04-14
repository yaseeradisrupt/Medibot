medical_bot/
├── __init__.py
├── main.py                   # Entry point 
├── config.py                 # Configuration and environment variables
├── utils/
│   ├── __init__.py
│   ├── chat_history.py       # Functions for managing chat history
│   └── message_formatting.py # Message handling utilities
├── datasources/
│   ├── __init__.py
│   ├── vectorstore.py        # Vector database interactions
│   ├── bigquery.py           # BigQuery database operations
│   └── web_search.py         # Web search operations (Tavily)
├── models/
│   ├── __init__.py
│   ├── router.py             # Question routing logic
│   ├── sql_generator.py      # SQL generation functions
│   └── intent_extractor.py   # Intent extraction from queries
└── graph/
    ├── __init__.py
    └── state_graph.py 