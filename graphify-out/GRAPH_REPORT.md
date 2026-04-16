# Graph Report - .  (2026-04-10)

## Corpus Check
- Large corpus: 308 files · ~605,074 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 3012 nodes · 5629 edges · 149 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 1733 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `MARK5DatabaseManager` - 116 edges
2. `AdvancedFeatureEngine` - 97 edges
3. `MARK5Predictor` - 76 edges
4. `MarketStatusChecker` - 72 edges
5. `DataProvider` - 68 edges
6. `PortfolioRiskAnalyzer` - 59 edges
7. `DataPipeline` - 56 edges
8. `MARK5MLTrainer` - 56 edges
9. `TradingSignalGenerator` - 50 edges
10. `ISEAdapter` - 50 edges

## Surprising Connections (you probably didn't know these)
- `Non-vectorized version for comparison.` --uses--> `AdvancedFeatureEngine`  [INFERRED]
  tests/verify_frac_diff_small.py → core/models/features.py
- `Non-vectorized version for comparison.` --uses--> `AdvancedFeatureEngine`  [INFERRED]
  tests/verify_frac_diff.py → core/models/features.py
- `Maestro: Dashboard Fixes` --semantically_similar_to--> `Client`  [INFERRED] [semantically similar]
  docs/maestro/state/archive/dashboard-fixes.md → graphify-3/worked/httpx/README.md
- `Verifies model exists, prompts to train if missing.` --uses--> `RobustBacktester`  [INFERRED]
  dashboard.py → core/models/tcn/backtester.py
- `Verifies model exists, prompts to train if missing.` --uses--> `DataPipeline`  [INFERRED]
  dashboard.py → core/data/data_pipeline.py

## Hyperedges (group relationships)
- **MARK5 High Priority Findings** — sec_01, log_01, perf_01 [EXTRACTED 1.00]
- **graphify Core Technologies** — leiden_clustering, tree_sitter_ast, claude_vision [EXTRACTED 1.00]
- **HTTPX Core Abstractions** — httpx_client, httpx_async_client, httpx_response [EXTRACTED 0.95]
- **MARK5 Training Pipeline Flow** — mark5_arch_financial_engineer, mark5_arch_ml_trainer, mark5_arch_predictor [EXTRACTED 1.00]
- **MARK5 High Priority Findings** — sec_01, log_01, perf_01 [EXTRACTED 1.00]
- **graphify Core Technologies** — leiden_clustering, tree_sitter_ast, claude_vision [EXTRACTED 1.00]
- **Comprehensive Stock Validation Charts** — reports_itc_chart, reports_sbin_chart, reports_infy_chart, reports_bhartiartl_chart, reports_tcs_chart, reports_hdfcbank_chart, reports_icicibank_chart, reports_kotakbank_chart, reports_reliance_chart, reports_hindunilvr_chart [INFERRED 0.90]
- **Aggregate Validation Reports** — reports_portfolio_dashboard, reports_cross_stock_heatmap, reports_portfolio_equity_curve, reports_sector_performance, reports_stock_rankings [INFERRED 0.90]
- **Dashboard UI Assets** — ui_favicon, ui_icons, ui_vite_logo, ui_hero_graphic, ui_react_logo [INFERRED 0.90]

## Communities

### Community 0 - "Autonomous Trading & Risk Management"
Cohesion: 0.02
Nodes (168): AlertLevel, AlertManager, AlertType, MARK5 ALERT MANAGER v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Hot-reloadable configuration., Structured alert creation — accepts enum values from autonomous.py.         Conv, Non-blocking alert dispatch.         Puts payload into queue and returns control, Consumer loop. Processes alerts one by one to prevent network congestion. (+160 more)

### Community 1 - "Data Engineering & ML Infrastructure"
Cohesion: 0.02
Nodes (153): BaseFeed, BulkDealsProvider, MARK5 Bulk & Block Deals Data Provider, Historical large deals endpoint on NSE is disabled/404.         For Phase 1 vali, Compute:         1. institutional_buy_5d: 5d rolling sum of buy-side value / 20d, Fetch today's bulk/block deals summary., Fetch daily live bulk/block deals and append to local cache., Process JSON to parquet records append. (+145 more)

### Community 2 - "HTTP Client & API Adapters"
Cohesion: 0.03
Nodes (101): Auth, BasicAuth, BearerAuth, DigestAuth, NetRCAuth, Authentication handlers. Auth objects are callables that modify a request before, Load credentials from ~/.netrc based on the request host., Base class for all authentication handlers. (+93 more)

### Community 3 - "Order Execution & Validation"
Cohesion: 0.03
Nodes (85): BaseExecutor, MARK5 BASE EXECUTOR v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Standard Interface for all Broker Adapters.     Now enforces Schema return types, CRITICAL: Rounds price to nearest 0.05 (NSE Standard).         Prevents 'InputEx, Send order to exchange. Return True if accepted., Reconciliation: Get daily order book converted to Internal Schema., Reconciliation: Get net positions converted to Internal Schema., BaseExecutor (+77 more)

### Community 4 - "Persistence & Time-series Data"
Cohesion: 0.02
Nodes (47): bootstrap_system(), MARK5 SYSTEM BOOTSTRAPPER v9.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Verify that Kite API credentials and access token are present and valid., Wire all core services into the dependency-injection container.      Call order, _validate_kite_session(), get_timescale_manager(), Efficient Bulk Insert using COPY or execute_values, Read operations can be synchronous or async depending on need.         For strat (+39 more)

### Community 5 - "Language-specific Extraction Tests"
Cohesion: 0.03
Nodes (48): _calls(), _labels(), Tests for language extractors: Java, C, C++, Ruby, C#, Kotlin, Scala, PHP, Swift, Methods on the same receiver type must share one canonical type node., Type node id should be scoped to directory, not file stem., _relations(), test_c_finds_functions(), test_c_finds_includes() (+40 more)

### Community 6 - "Market Data Feed & Adapters"
Cohesion: 0.04
Nodes (43): BaseFeed, MARK5 BASE DATA FEED v9.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Universal tick structure.     Every adapter MUST convert raw exchange data into, Abstract base class for all data feed adapters.     Implements the Observer patt, Register a strategy callback to receive tick updates., Push normalized ticks to all registered observers.          FIX L-06: Was using, TickData, DataValidator (+35 more)

### Community 7 - "Core Processing & Utilities"
Cohesion: 0.03
Nodes (44): Base, Server, LinearAlgebra, add(), Animal, -initWithName, -speak, ApiClient (+36 more)

### Community 8 - "Source Code AST Extraction"
Cohesion: 0.04
Nodes (78): _check_tree_sitter_version(), _csharp_extra_walk(), extract(), extract_c(), extract_cpp(), extract_csharp(), extract_elixir(), _extract_generic() (+70 more)

### Community 9 - "File Parsing & Document Storage"
Cohesion: 0.04
Nodes (67): handle_delete(), handle_enrich(), handle_get(), handle_list(), handle_search(), handle_upload(), API module - exposes the document pipeline over HTTP. Thin layer over parser, va, Accept a list of file paths, run the full pipeline on each,     and return a sum (+59 more)

### Community 10 - "File Detection & Graphify Validation"
Cohesion: 0.05
Nodes (47): classify_file(), convert_office_file(), count_words(), detect(), detect_incremental(), docx_to_markdown(), extract_pdf_text(), FileType (+39 more)

### Community 11 - "Position Sizing & Regime Detection"
Cohesion: 0.05
Nodes (34): MarketRegime, MARK5 DECISION ENGINE v8.0 - PRODUCTION GRADE (REACTIVE) ━━━━━━━━━━━━━━━━━━━━━━━, Legacy helper for autonomous.py, Main Reactor Loop. Called whenever new data arrives.         data_snapshot: Data, almgren_chriss_slippage(), MARK5 VOLATILITY AWARE POSITION SIZER v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━, Calculate optimal position size (number of shares).          Args:             s, Estimate one-way execution slippage using the Almgren-Chriss framework, (+26 more)

### Community 12 - "Skill & Agent Installation Tests"
Cohesion: 0.05
Nodes (51): _agents_install(), _agents_uninstall(), _install(), Tests for graphify install --platform routing., Claude platform install writes CLAUDE.md; others do not., Installing twice does not duplicate the section., Installs into an existing AGENTS.md without overwriting other content., Uninstall keeps pre-existing content. (+43 more)

### Community 13 - "ISE Feature Engineering & Signaling"
Cohesion: 0.06
Nodes (40): graphify - extract · build · cluster · analyze · report., _load_json(), _month_key(), MARK5 INDIAN STOCK API ADAPTER v1.1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  Provide, Return current token budget: {used, remaining, limit}., Core fetch with:           - Budget gate (raise before any network hit), Raised when monthly 500-request budget would be breached., _save_json() (+32 more)

### Community 14 - "Main Entry Points & System Health"
Cohesion: 0.07
Nodes (38): _agents_install(), _agents_uninstall(), _check_skill_version(), claude_install(), claude_uninstall(), _cursor_install(), _cursor_uninstall(), gemini_install() (+30 more)

### Community 15 - "Common Utilities & Caching"
Cohesion: 0.06
Nodes (34): ensure_directory(), generate_cache_key(), get_logger(), MARK5 COMMON UTILITIES v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Safely save data to pickle file          Args:         data: Data to save, Setup logger with consistent formatting and duplicate prevention          Args:, Get existing logger or create with default settings          Args:         name:, Reset logger initialization state          Args:         name: Logger name to re (+26 more)

### Community 16 - "Bhavcopy & FnO Data Processing"
Cohesion: 0.07
Nodes (25): BhavCopyFetcher, _bs_price(), FNOFeatureEngine, _futures_basis_raw(), _implied_vol(), _iv_skew_raw(), _max_pain_distance(), _near_expiry() (+17 more)

### Community 17 - "Configuration & System Validation"
Cohesion: 0.11
Nodes (27): BaseModel, get_config(), get_database_config(), get_execution_config(), get_redis_config(), get_risk_config(), get_timescale_config(), MARK5 CONFIG MANAGER v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (+19 more)

### Community 18 - "Hyperparameter Optimization Engine"
Cohesion: 0.08
Nodes (17): HyperparameterOptimizer, MARK5 HYPERPARAMETER OPTIMIZER v9.0 - MULTICLASS EDITION C-01 FIX: All objective, Optimizes LightGBM for MULTICLASS LogLoss., Optimizes RandomForest for MULTICLASS classification., Optimizes CatBoost for MULTICLASS LogLoss with ordered boosting., Multi-model hyperparameter optimizer using Optuna.     C-01 FIX: All objectives, Optimizes all ensemble models (XGBoost, LightGBM, RandomForest, CatBoost)., Atomic save of optimized parameters. (+9 more)

### Community 19 - "Backtesting & Pipeline Orchestration"
Cohesion: 0.09
Nodes (18): IndianTaxConfig, CHANAKYA: Advanced Indian Market Backtesting Engine (NSE/BSE) Architected for TC, True Range calculation handling gaps, The Core Simulation Loop.         CRITICAL: Executions happen at OPEN of i+1 bas, Exact Tax Structure for NSE Equity/F&O (As of 2024-25), Calculates exact breakdown of charges, RobustBacktester, Trade (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.09
Nodes (9): ABC, MARK5Launcher, MARK5 SYSTEM LAUNCHER v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, BaseWorker, ExecutionWorker, get_process_manager(), InferenceWorker, ProcessManager (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (18): main(), ProductionGate, Validate backtest performance meets requirements., Validate paper trading performance., Validate training data has balanced signals., Validate model accuracy on out-of-sample data., Comprehensive production readiness validation.          Requirements for live de, Validate risk management systems are in place. (+10 more)

### Community 22 - "Community 22"
Cohesion: 0.09
Nodes (33): _cross_community_surprises(), _cross_file_surprises(), _file_category(), god_nodes(), graph_diff(), _is_concept_node(), _is_file_node(), _node_community_map() (+25 more)

### Community 23 - "Community 23"
Cohesion: 0.07
Nodes (19): PathValidationError, MARK5 ROBUST REGISTRY v8.0 - HARDENED EDITION Revisions:     v7.0: Atomic writes, Normalize and validate ticker symbol., Validate that metadata is JSON-serializable., Load registry from file.                  Returns:             Registry dict, Atomically save registry to disk.                  Uses temp file + rename patte, Compute SHA-256 checksum of file.                  Args:             file_path:, Generate unique model ID using UUID. (+11 more)

### Community 24 - "Community 24"
Cohesion: 0.07
Nodes (18): After merging multiple files, no internal edges should be dangling., Call-graph pass must produce INFERRED calls edges., AST-resolved call edges are deterministic and should be EXTRACTED/1.0., Same input always produces same output., run_analysis() calls compute_score() - must appear as a calls edge., Analyzer.process() calls run_analysis() - cross class→function calls edge., Same caller→callee pair must appear only once even if called multiple times., All edge sources must reference a known node (targets may be external imports). (+10 more)

### Community 25 - "Community 25"
Cohesion: 0.07
Nodes (25): Tests for graphify/cache.py., Non-.md files are still hashed by their full content., _body_content correctly strips YAML frontmatter., _body_content returns content unchanged when no frontmatter present., Same file gives same hash on repeated calls., Different file contents give different hashes., Save then load returns the same result dict., After file content changes, load_cached returns None. (+17 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (14): main(), Audit data collection and processing pipeline., Audit ML model implementation., Audit trading signal generation and risk management., Comprehensive system audit and gap analysis., Audit backtesting and validation frameworks., Audit production readiness., Audit for HOLD bias in training data. (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.1
Nodes (24): compute_atr(), compute_frac_diff_weights(), compute_mfi(), compute_rsi(), compute_volatility_clustering(), engineer_features_df(), engineer_features_tensor(), engineer_tcn_features_df() (+16 more)

### Community 28 - "Community 28"
Cohesion: 0.11
Nodes (23): make_graph(), _make_simple_graph(), Tests for analyze.py., Code↔paper edge should score higher than code↔code edge., Helper: build a small nx.Graph from node/edge specs., Multi-file graph: should find cross-file edges between real entities., Concept nodes (empty source_file) must not appear in surprises., Single-file graph: should return cross-community edges, not empty list. (+15 more)

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (17): _call_pairs(), _confidences(), _labels(), Tests for multi-language AST extraction: JS/TS, Go, Rust., test_go_emits_calls(), test_go_finds_constructor(), test_go_finds_methods(), test_go_finds_struct() (+9 more)

### Community 30 - "Community 30"
Cohesion: 0.08
Nodes (25): Tests for graphify claude install / uninstall commands., claude_install also writes .claude/settings.json with PreToolUse hook., Running claude_install twice does not duplicate the PreToolUse hook., Creates CLAUDE.md when none exists., claude_uninstall removes the PreToolUse hook from settings.json., Written section includes the three rules., Appends to an existing CLAUDE.md without clobbering it., Running install twice does not duplicate the section. (+17 more)

### Community 31 - "Community 31"
Cohesion: 0.09
Nodes (13): Generate trading signal from prediction.                  Args:             pred, Determine signal type and strength from prediction, Create a HOLD signal with given reasons, Calculate ATR-based stop loss.                  Args:             return_pct: Pr, Calculate fallback stop loss when ATR not available, Calculate take profit level, Calculate risk-reward ratio, Calculate recommended position size as fraction (+5 more)

### Community 32 - "Community 32"
Cohesion: 0.1
Nodes (6): _make_mock_response(), Tests for graphify/security.py - URL validation, safe fetch, path guards, label, test_safe_fetch_raises_on_non_2xx(), test_safe_fetch_returns_bytes(), test_safe_fetch_text_decodes_utf8(), test_safe_fetch_text_replaces_bad_bytes()

### Community 33 - "Community 33"
Cohesion: 0.09
Nodes (23): Top 8 Features IC - ADANIENT.NS, Top 8 Features IC - ASIANPAINT.NS, Top 8 Features IC - AXISBANK.NS, Top 8 Features IC - BAJFINANCE.NS, Top 8 Features IC - BHARTIARTL.NS, Top 8 Features IC - DIXON.NS, Top 8 Features IC - DLF.NS, Top 8 Features IC - HDFCBANK.NS (+15 more)

### Community 34 - "Community 34"
Cohesion: 0.11
Nodes (20): attach_hyperedges(), _cypher_escape(), _html_script(), _html_styles(), _hyperedge_script(), push_to_neo4j(), Store hyperedges in the graph's metadata dict., Escape a string for safe embedding in a Cypher single-quoted literal. (+12 more)

### Community 35 - "Community 35"
Cohesion: 0.16
Nodes (21): _detect_url_type(), _download_binary(), _fetch_arxiv(), _fetch_html(), _fetch_tweet(), _fetch_webpage(), _html_to_markdown(), ingest() (+13 more)

### Community 36 - "Community 36"
Cohesion: 0.13
Nodes (11): CircuitBreakerDetector, MARK5 Market Utilities v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Calculate total cost for buy + sell (complete trade), Calculate net P&L after all transaction costs, 🔥 BUG FIX #5: Circuit Breaker Detection     Detects when stocks hit circuit limi, Check if stock has hit circuit breaker limits                  Args:, Check if stock is approaching circuit breaker (within threshold %), 🔥 BUG FIX #4: Transaction Cost Modeling     Calculates real-world trading costs (+3 more)

### Community 37 - "Community 37"
Cohesion: 0.18
Nodes (17): _make_graph(), Tests for serve.py - MCP graph query helpers (no mcp package required)., test_bfs_depth_1(), test_bfs_depth_2(), test_bfs_disconnected(), test_bfs_returns_edges(), test_communities_from_graph_basic(), test_communities_from_graph_isolated() (+9 more)

### Community 38 - "Community 38"
Cohesion: 0.17
Nodes (19): _make_graph(), Tests for graphify.wiki — Wikipedia-style article generation., God node with bad ID should not crash., Communities with more than 25 nodes show a truncation notice., test_article_navigation_footer(), test_community_article_has_audit_trail(), test_community_article_has_cross_links(), test_community_article_shows_cohesion() (+11 more)

### Community 39 - "Community 39"
Cohesion: 0.16
Nodes (18): _body_content(), cache_dir(), cached_files(), check_semantic_cache(), clear_cache(), file_hash(), load_cached(), Strip YAML frontmatter from Markdown content, returning only the body. (+10 more)

### Community 40 - "Community 40"
Cohesion: 0.14
Nodes (17): build_graph(), cluster(), cohesion_score(), _partition(), Leiden community detection on NetworkX graphs. Splits oversized communities. Ret, Run a second Leiden pass on a community subgraph to split it further., Context manager to suppress stdout/stderr during library calls.      graspologic, Ratio of actual intra-community edges to maximum possible. (+9 more)

### Community 41 - "Community 41"
Cohesion: 0.14
Nodes (17): _make_extraction(), Tests for confidence_score on edges., Edges lacking confidence_score get sensible defaults in to_json., Report summary line should include avg confidence for INFERRED edges., Surprising connections section shows confidence score next to INFERRED edges., Return a minimal extraction dict with one edge of each confidence type., EXTRACTED edges must have confidence_score == 1.0., INFERRED edges must have confidence_score between 0.0 and 1.0. (+9 more)

### Community 42 - "Community 42"
Cohesion: 0.14
Nodes (8): _make_report(), Tests for hyperedge support in graphify., Write graph.json then reload it - hyperedges must survive., test_hyperedges_roundtrip_via_json_file(), test_report_includes_hyperedge_node_list(), test_report_includes_hyperedges_section(), test_report_skips_hyperedges_section_when_empty(), test_report_skips_hyperedges_section_when_key_missing()

### Community 43 - "Community 43"
Cohesion: 0.18
Nodes (16): _make_extraction_with_semantic_edge(), _make_graph_with_semantic_edge(), _make_report_with_semantic_surprise(), _make_two_edge_graph(), Tests for semantically_similar_to edge support., Two nodes in separate files connected by a semantically_similar_to edge., Non-semantic edges must not get the [semantically similar] tag., Graph with one semantically_similar_to edge and one references edge, both cross- (+8 more)

### Community 44 - "Community 44"
Cohesion: 0.12
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 0.23
Nodes (14): _make_git_repo(), Tests for hooks.py - git hook install/uninstall., test_install_appends_to_existing_hook(), test_install_creates_hook(), test_install_creates_post_checkout_hook(), test_install_idempotent(), test_install_is_executable(), test_install_post_checkout_is_executable() (+6 more)

### Community 46 - "Community 46"
Cohesion: 0.17
Nodes (13): _build_opener(), _NoFileRedirectHandler, Fetch *url* and return decoded text (UTF-8, replacing bad bytes).      Wraps saf, Resolve *path* and verify it stays inside *base*.      *base* defaults to the `g, Strip control characters and cap length.      Safe for embedding in JSON data (i, Raise ValueError if *url* is not http or https, or targets a private/internal IP, Redirect handler that re-validates every redirect target.      Prevents open-red, Fetch *url* and return raw bytes.      Protections applied:     - URL scheme val (+5 more)

### Community 47 - "Community 47"
Cohesion: 0.26
Nodes (14): make_graph(), test_to_cypher_contains_merge_statements(), test_to_cypher_creates_file(), test_to_graphml_creates_file(), test_to_graphml_has_community_attribute(), test_to_graphml_valid_xml(), test_to_html_contains_legend_with_labels(), test_to_html_contains_nodes_and_edges() (+6 more)

### Community 48 - "Community 48"
Cohesion: 0.29
Nodes (13): _make_graph(), Tests for graphify/benchmark.py., test_print_benchmark_no_crash(), test_query_bfs_expands_neighbors(), test_query_returns_positive_for_matching_question(), test_query_returns_zero_for_no_match(), test_run_benchmark_corpus_tokens_proportional(), test_run_benchmark_error_on_empty_graph() (+5 more)

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (6): MARK5 Training Analytics v8.0 - PRODUCTION GRADE (FINANCIAL GRADE) ━━━━━━━━━━━━━, Finds the best performing model for each market regime., Did we improve over last week?, Generates the Architect's Report, ResultsAnalyzer, TrainingResultsAnalyzer

### Community 50 - "Community 50"
Cohesion: 0.15
Nodes (8): get_system_info(), MARK5 SYSTEM MONITOR v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━, Log an error and update stats., Get overall system status., Get comprehensive system information          Returns:         Dictionary contai, Centralized monitor for system health, resources, and error tracking., Check system resources and return metrics., SystemHealthMonitor

### Community 51 - "Community 51"
Cohesion: 0.18
Nodes (9): _communities_from_graph(), _find_node(), _load_graph(), Return node IDs whose label or ID matches the search term (case-insensitive)., Start the MCP server. Requires pip install mcp., Reconstruct community dict from community property stored on nodes., Render subgraph as text, cutting at token_budget (approx 3 chars/token)., serve() (+1 more)

### Community 52 - "Community 52"
Cohesion: 0.22
Nodes (12): _git_root(), install(), _install_hook(), Walk up to find .git directory., Install a single git hook, appending if an existing hook is present., Remove graphify section from a git hook using start/end markers., Install graphify post-commit and post-checkout hooks in the nearest git repo., Remove graphify post-commit and post-checkout hooks. (+4 more)

### Community 53 - "Community 53"
Cohesion: 0.23
Nodes (9): make_graph(), Clustering should not emit ANSI escape codes or other output.      graspologic's, Same as above but for stderr — ANSI codes can go to either stream., test_cluster_covers_all_nodes(), test_cluster_does_not_write_to_stderr(), test_cluster_does_not_write_to_stdout(), test_cluster_returns_dict(), test_cohesion_score_range() (+1 more)

### Community 54 - "Community 54"
Cohesion: 0.26
Nodes (12): End-to-end pipeline test: detect → extract → build → cluster → analyze → report, Second run on unchanged corpus should produce identical node/edge counts., Run the full pipeline on the fixtures directory. Returns a dict of outputs., run_pipeline(), test_pipeline_all_nodes_have_community(), test_pipeline_detection_finds_code_and_docs(), test_pipeline_extraction_confidence_labels(), test_pipeline_graph_has_edges() (+4 more)

### Community 55 - "Community 55"
Cohesion: 0.17
Nodes (13): Autonomous Trader Module, Database Manager Module, Execution Engine Module, Features Engineering Module, Kite Adapter Module, Dashboard Main Application, Phase 1: Foundation & Persistence Relocation, Phase 2: Database Schema & Indexing (+5 more)

### Community 56 - "Community 56"
Cohesion: 0.2
Nodes (5): Zero-copy write if input is compatible, otherwise fast copy.         Advanced: U, HYBRID SPIN-YIELD STRATEGY         Returns a READ-ONLY view of the data. DO NOT, Multi-Consumer Observer Method.         Reads all available data from last_head, Architectural Grade: HFT     Mechanism: Single-Producer / Single-Consumer (SPSC), ZeroCopyRingBuffer

### Community 57 - "Community 57"
Cohesion: 0.17
Nodes (6): AlphaFeatureEngineer, ALPHA FEATURE CORE: TCN-Optimized Feature Engineering (DEPRECATED) ━━━━━━━━━━━━━, Legacy Wrapper for TCN Feature Engineering.     Now routes to the unified pure t, No-op: Unified engine is stateless., No-op: Unified engine is stateless., Generates features using the unified pure engine.         Local normalization en

### Community 58 - "Community 58"
Cohesion: 0.27
Nodes (11): Tests for rationale/docstring extraction in extract.py., # NOTE: must run before compile() or linker will fail, Trivial docstrings under 20 chars should not become rationale nodes., test_class_docstring_extracted(), test_function_docstring_extracted(), test_module_docstring_extracted(), test_rationale_comment_extracted(), test_rationale_confidence_is_extracted() (+3 more)

### Community 59 - "Community 59"
Cohesion: 0.17
Nodes (0): 

### Community 60 - "Community 60"
Cohesion: 0.22
Nodes (9): build(), build_from_json(), Build a NetworkX graph from an extraction dict.      directed=True produces a Di, Merge multiple extraction results into one graph., Merge multiple extraction results into one graph.      directed=True produces a, assert_valid(), Validate an extraction JSON dict against the graphify schema.     Returns a list, Raise ValueError with all errors if extraction is invalid. (+1 more)

### Community 61 - "Community 61"
Cohesion: 0.38
Nodes (9): make_inputs(), test_report_contains_ambiguous_section(), test_report_contains_communities(), test_report_contains_corpus_check(), test_report_contains_god_nodes(), test_report_contains_header(), test_report_contains_surprising_connections(), test_report_shows_raw_cohesion_scores() (+1 more)

### Community 62 - "Community 62"
Cohesion: 0.2
Nodes (1): Tests for watch.py - file watcher helpers (no watchdog required).

### Community 63 - "Community 63"
Cohesion: 0.2
Nodes (1): Tests for graphify.ingest.save_query_result

### Community 64 - "Community 64"
Cohesion: 0.28
Nodes (8): _estimate_tokens(), print_benchmark(), _query_subgraph_tokens(), Token-reduction benchmark - measures how much context graphify saves vs naive fu, Print a human-readable benchmark report., Run BFS from best-matching nodes and return estimated tokens in the subgraph con, Measure token reduction: corpus tokens vs graphify query tokens.      Args:, run_benchmark()

### Community 65 - "Community 65"
Cohesion: 0.36
Nodes (8): _community_article(), _cross_community_links(), _god_node_article(), _index_md(), Return (community_label, edge_count) pairs for cross-community connections, sort, Generate a Wikipedia-style wiki from the graph.      Writes:       - index.md, _safe_filename(), to_wiki()

### Community 66 - "Community 66"
Cohesion: 0.39
Nodes (5): Analyzer, compute_score(), normalize(), Fixture: functions and methods that call each other - for call-graph extraction, run_analysis()

### Community 67 - "Community 67"
Cohesion: 0.61
Nodes (7): backtest_menu(), ensure_model_ready(), ise_intelligence_menu(), main(), run_backtest(), show_banner(), training_menu()

### Community 68 - "Community 68"
Cohesion: 0.36
Nodes (7): _has_non_code(), _notify_only(), Re-run AST extraction + build + cluster + report for code files. No LLM needed., Write a flag file and print a notification (fallback for non-code-only corpora)., Watch watch_path for new or modified files and auto-update the graph.      For c, _rebuild_code(), watch()

### Community 69 - "Community 69"
Cohesion: 0.43
Nodes (6): load_extraction(), test_ambiguous_edge_preserved(), test_build_from_json_edge_count(), test_build_from_json_node_count(), test_edges_have_confidence(), test_nodes_have_label()

### Community 70 - "Community 70"
Cohesion: 0.29
Nodes (4): get_pnl(), get_stock(), Returns equity curve. Linked to DB, using mock generation for now due to lack of, Integrates real-time Kite Connect quote and ISE API fundamental data.

### Community 71 - "Community 71"
Cohesion: 0.4
Nodes (3): Mock background backtest process to simulate real one., run_backtest(), simulate_backtest()

### Community 72 - "Community 72"
Cohesion: 0.33
Nodes (3): EnsembleWeighter, MARK5 ENSEMBLE WEIGHTER v2.0 - ARCHITECT EDITION Revisions: 1. DECORRELATION BOO, Calculates weights based on Confidence, Regime, and Correlation.

### Community 73 - "Community 73"
Cohesion: 0.33
Nodes (6): Claude Vision, graphify Processing Pipeline, graphify AI coding assistant skill, /raw folder workflow, Leiden Community Detection, tree-sitter AST

### Community 74 - "Community 74"
Cohesion: 0.4
Nodes (2): Direct slot assignment.          Crash early if service name is invalid (Strict, ServiceContainer

### Community 75 - "Community 75"
Cohesion: 0.7
Nodes (4): get_w(), Non-vectorized version for comparison., slow_frac_diff_ffd(), test_frac_diff()

### Community 76 - "Community 76"
Cohesion: 0.4
Nodes (5): Attention Mechanism Notes, Layer Normalization, Multi-Head Attention, Positional Encoding, Transformer Architecture

### Community 77 - "Community 77"
Cohesion: 0.5
Nodes (1): MARK5 PRECISION DATABASE MANAGER v9.0 - PRODUCTION GRADE (The Vault) ━━━━━━━━━━━

### Community 78 - "Community 78"
Cohesion: 0.67
Nodes (3): Non-vectorized version for comparison., slow_frac_diff_ffd(), test_frac_diff()

### Community 79 - "Community 79"
Cohesion: 0.5
Nodes (1): DashboardHealthCheck

### Community 80 - "Community 80"
Cohesion: 0.5
Nodes (4): Adapter Pattern, MARK5 Codebase Audit Report, MARK3 Financial Analytics System, Service Container Pattern

### Community 81 - "Community 81"
Cohesion: 0.67
Nodes (0): 

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (2): main(), run_test()

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (2): get_w(), test_frac_diff()

### Community 84 - "Community 84"
Cohesion: 0.67
Nodes (3): CPCV Training, Kite Data Feed, ABB.NS Training Log

### Community 85 - "Community 85"
Cohesion: 0.67
Nodes (3): Portfolio Dashboard, Portfolio Equity Curve, RELIANCE.NS Validation Report

### Community 86 - "Community 86"
Cohesion: 0.67
Nodes (3): Cross-Stock Performance Heatmap, Sector Performance Breakdown, Stock Rankings by Sharpe Ratio

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (0): 

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (0): 

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (2): SEC-01: Dashboard API lacks authentication, Rationale for SEC-01

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (2): LOG-01: Volatile Trading State, Rationale for LOG-01

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (2): Lookahead Probe Failure, MARK5 Phase 0 Summary

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (2): Multi-Head Attention, Transformer Architecture

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (2): FlashAttention Paper, GPT (nanoGPT)

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (2): DigestAuth, Response

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (2): architecture.md, storage.py

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (2): Experiment A: 20-Day Hold, Final Optimized Backtest

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (2): FinancialEngineer, MARK5MLTrainer

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (2): Client, Maestro: Dashboard Fixes

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (2): React Logo, Vite Logo

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (1): Robust Factory Method for Deserialization.

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (1): Round price to valid tick size.                  Utility for order entry UIs.

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (1): Return the nearest expiry date >= d.

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (1): Put/Call OI ratio. Validated: [0.1, 5.0].

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (1): Raw basis = near-month futures settle - spot.         Sign: positive = futures a

### Community 105 - "Community 105"
Cohesion: 1.0
Nodes (1): IV skew proxy = median(OTM PE close) - median(OTM CE close), divided by spot.

### Community 106 - "Community 106"
Cohesion: 1.0
Nodes (1): sign(Δspot_return) × sign(Δfutures_OI).         +1: price up + OI up = bullish (

### Community 107 - "Community 107"
Cohesion: 1.0
Nodes (1): Computes max pain strike = strike that causes maximum aggregate loss to

### Community 108 - "Community 108"
Cohesion: 1.0
Nodes (1): Establish connection and build instrument map. Returns True on success.

### Community 109 - "Community 109"
Cohesion: 1.0
Nodes (1): Close connection and release resources.

### Community 110 - "Community 110"
Cohesion: 1.0
Nodes (1): Fetch historical OHLCV data.          Returns:             DataFrame with tz-awa

### Community 111 - "Community 111"
Cohesion: 1.0
Nodes (1): Subscribe to real-time ticks for the given symbols.

### Community 112 - "Community 112"
Cohesion: 1.0
Nodes (1): Unsubscribe from real-time ticks.

### Community 113 - "Community 113"
Cohesion: 1.0
Nodes (1): O(1) symbol → instrument token lookup.

### Community 114 - "Community 114"
Cohesion: 1.0
Nodes (1): Return a health status dict for monitoring.

### Community 115 - "Community 115"
Cohesion: 1.0
Nodes (1): Validates data WITHOUT modifying it.          Rejects the dataset if critical fl

### Community 116 - "Community 116"
Cohesion: 1.0
Nodes (1): Ensures no data exists outside 09:15 - 15:30 IST.

### Community 117 - "Community 117"
Cohesion: 1.0
Nodes (0): 

### Community 118 - "Community 118"
Cohesion: 1.0
Nodes (0): 

### Community 119 - "Community 119"
Cohesion: 1.0
Nodes (0): 

### Community 120 - "Community 120"
Cohesion: 1.0
Nodes (0): 

### Community 121 - "Community 121"
Cohesion: 1.0
Nodes (0): 

### Community 122 - "Community 122"
Cohesion: 1.0
Nodes (1): graphify knowledge graph

### Community 123 - "Community 123"
Cohesion: 1.0
Nodes (1): PERF-01: Missing DB Indexes

### Community 124 - "Community 124"
Cohesion: 1.0
Nodes (1): Value (micrograd)

### Community 125 - "Community 125"
Cohesion: 1.0
Nodes (1): AsyncClient

### Community 126 - "Community 126"
Cohesion: 1.0
Nodes (1): attention_notes.md

### Community 127 - "Community 127"
Cohesion: 1.0
Nodes (1): MARK5Predictor

### Community 128 - "Community 128"
Cohesion: 1.0
Nodes (1): Maestro: ML Pipeline Refactor

### Community 129 - "Community 129"
Cohesion: 1.0
Nodes (1): Post Earnings Drift

### Community 130 - "Community 130"
Cohesion: 1.0
Nodes (1): ATR Regime

### Community 131 - "Community 131"
Cohesion: 1.0
Nodes (1): RSI (14)

### Community 132 - "Community 132"
Cohesion: 1.0
Nodes (1): Relative Strength vs Nifty

### Community 133 - "Community 133"
Cohesion: 1.0
Nodes (1): Sector Relative Strength

### Community 134 - "Community 134"
Cohesion: 1.0
Nodes (1): Distance to 52W High

### Community 135 - "Community 135"
Cohesion: 1.0
Nodes (1): Volume Z-Score

### Community 136 - "Community 136"
Cohesion: 1.0
Nodes (1): Gap Significance

### Community 137 - "Community 137"
Cohesion: 1.0
Nodes (1): ITC.NS Validation Report

### Community 138 - "Community 138"
Cohesion: 1.0
Nodes (1): SBIN.NS Validation Report

### Community 139 - "Community 139"
Cohesion: 1.0
Nodes (1): INFY.NS Validation Report

### Community 140 - "Community 140"
Cohesion: 1.0
Nodes (1): BHARTIARTL.NS Validation Report

### Community 141 - "Community 141"
Cohesion: 1.0
Nodes (1): TCS.NS Validation Report

### Community 142 - "Community 142"
Cohesion: 1.0
Nodes (1): HDFCBANK.NS Validation Report

### Community 143 - "Community 143"
Cohesion: 1.0
Nodes (1): ICICIBANK.NS Validation Report

### Community 144 - "Community 144"
Cohesion: 1.0
Nodes (1): KOTAKBANK.NS Validation Report

### Community 145 - "Community 145"
Cohesion: 1.0
Nodes (1): HINDUNILVR.NS Validation Report

### Community 146 - "Community 146"
Cohesion: 1.0
Nodes (1): Dashboard Favicon

### Community 147 - "Community 147"
Cohesion: 1.0
Nodes (1): UI Icon Sprites

### Community 148 - "Community 148"
Cohesion: 1.0
Nodes (1): Hero Graphic

## Knowledge Gaps
- **761 isolated node(s):** `Mock background backtest process to simulate real one.`, `Returns equity curve. Linked to DB, using mock generation for now due to lack of`, `Integrates real-time Kite Connect quote and ISE API fundamental data.`, `MARK5 ALERT MANAGER v8.0 - PRODUCTION GRADE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, `MARK5 High-Performance Asynchronous Alert System.     Architecture: Producer-Con` (+756 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 87`** (2 nodes): `test_vectorization_fix.py`, `test_vectorization()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (2 nodes): `report.py`, `generate()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (2 nodes): `SEC-01: Dashboard API lacks authentication`, `Rationale for SEC-01`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (2 nodes): `LOG-01: Volatile Trading State`, `Rationale for LOG-01`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (2 nodes): `Lookahead Probe Failure`, `MARK5 Phase 0 Summary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (2 nodes): `Multi-Head Attention`, `Transformer Architecture`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (2 nodes): `FlashAttention Paper`, `GPT (nanoGPT)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (2 nodes): `DigestAuth`, `Response`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (2 nodes): `architecture.md`, `storage.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (2 nodes): `Experiment A: 20-Day Hold`, `Final Optimized Backtest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (2 nodes): `FinancialEngineer`, `MARK5MLTrainer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (2 nodes): `Client`, `Maestro: Dashboard Fixes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (2 nodes): `React Logo`, `Vite Logo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (1 nodes): `Robust Factory Method for Deserialization.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `Round price to valid tick size.                  Utility for order entry UIs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (1 nodes): `Return the nearest expiry date >= d.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (1 nodes): `Put/Call OI ratio. Validated: [0.1, 5.0].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (1 nodes): `Raw basis = near-month futures settle - spot.         Sign: positive = futures a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (1 nodes): `IV skew proxy = median(OTM PE close) - median(OTM CE close), divided by spot.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 106`** (1 nodes): `sign(Δspot_return) × sign(Δfutures_OI).         +1: price up + OI up = bullish (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 107`** (1 nodes): `Computes max pain strike = strike that causes maximum aggregate loss to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 108`** (1 nodes): `Establish connection and build instrument map. Returns True on success.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 109`** (1 nodes): `Close connection and release resources.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 110`** (1 nodes): `Fetch historical OHLCV data.          Returns:             DataFrame with tz-awa`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 111`** (1 nodes): `Subscribe to real-time ticks for the given symbols.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 112`** (1 nodes): `Unsubscribe from real-time ticks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 113`** (1 nodes): `O(1) symbol → instrument token lookup.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 114`** (1 nodes): `Return a health status dict for monitoring.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 115`** (1 nodes): `Validates data WITHOUT modifying it.          Rejects the dataset if critical fl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 116`** (1 nodes): `Ensures no data exists outside 09:15 - 15:30 IST.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 117`** (1 nodes): `jbdf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 118`** (1 nodes): `test_tracing.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 119`** (1 nodes): `manifest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 120`** (1 nodes): `vite.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 121`** (1 nodes): `eslint.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 122`** (1 nodes): `graphify knowledge graph`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 123`** (1 nodes): `PERF-01: Missing DB Indexes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 124`** (1 nodes): `Value (micrograd)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 125`** (1 nodes): `AsyncClient`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 126`** (1 nodes): `attention_notes.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 127`** (1 nodes): `MARK5Predictor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 128`** (1 nodes): `Maestro: ML Pipeline Refactor`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 129`** (1 nodes): `Post Earnings Drift`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 130`** (1 nodes): `ATR Regime`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 131`** (1 nodes): `RSI (14)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 132`** (1 nodes): `Relative Strength vs Nifty`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 133`** (1 nodes): `Sector Relative Strength`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 134`** (1 nodes): `Distance to 52W High`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 135`** (1 nodes): `Volume Z-Score`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 136`** (1 nodes): `Gap Significance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 137`** (1 nodes): `ITC.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 138`** (1 nodes): `SBIN.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 139`** (1 nodes): `INFY.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 140`** (1 nodes): `BHARTIARTL.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 141`** (1 nodes): `TCS.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 142`** (1 nodes): `HDFCBANK.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 143`** (1 nodes): `ICICIBANK.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 144`** (1 nodes): `KOTAKBANK.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 145`** (1 nodes): `HINDUNILVR.NS Validation Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 146`** (1 nodes): `Dashboard Favicon`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 147`** (1 nodes): `UI Icon Sprites`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 148`** (1 nodes): `Hero Graphic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TCNTradingModel` connect `Data Engineering & ML Infrastructure` to `Autonomous Trading & Risk Management`, `Configuration & System Validation`, `Backtesting & Pipeline Orchestration`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `DataProvider` connect `Autonomous Trading & Risk Management` to `Data Engineering & ML Infrastructure`, `Community 20`, `Persistence & Time-series Data`, `Market Data Feed & Adapters`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `MARK5Predictor` connect `Autonomous Trading & Risk Management` to `Data Engineering & ML Infrastructure`, `Community 20`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 92 inferred relationships involving `MARK5DatabaseManager` (e.g. with `TradingSignal` and `AutonomousTrader`) actually correct?**
  _`MARK5DatabaseManager` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `AdvancedFeatureEngine` (e.g. with `TradingSignal` and `AutonomousTrader`) actually correct?**
  _`AdvancedFeatureEngine` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 69 inferred relationships involving `MARK5Predictor` (e.g. with `Verifies model exists, prompts to train if missing.` and `Core backtest execution logic.`) actually correct?**
  _`MARK5Predictor` has 69 INFERRED edges - model-reasoned connections that need verification._
- **Are the 61 inferred relationships involving `MarketStatusChecker` (e.g. with `TradingSignal` and `AutonomousTrader`) actually correct?**
  _`MarketStatusChecker` has 61 INFERRED edges - model-reasoned connections that need verification._