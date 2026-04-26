# =============================================================================
# RAG 教学示例 - 综合测试脚本
# =============================================================================
# 测试所有文件的语法和函数结构

import sys
import importlib.util

# 测试结果记录
test_results = []

def test_file_syntax(file_path, module_name):
    """测试文件语法"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        # 只检查语法，不执行
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误：{e}"
    except Exception as e:
        return False, f"导入错误：{e}"

def test_function_exists(file_path, expected_functions):
    """检查期望的函数/类是否存在"""
    found = []
    missing = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for func in expected_functions:
        # 检查函数或类定义
        if f"def {func}(" in content or f"class {func}" in content:
            found.append(func)
        else:
            missing.append(func)

    return found, missing

# ==================== 01_milvus_basics ====================
print("=" * 60)
print("测试模块：01_milvus_basics")
print("=" * 60)

modules_01 = [
    ("01_milvus_basics/01_connect_milvus.py", [
        "connect_local_milvus", "connect_remote_milvus",
        "connect_with_auth", "connection_parameters_explained",
        "check_connection_health"
    ]),
    ("01_milvus_basics/02_create_collection.py", [
        "create_simple_collection", "create_custom_collection",
        "create_multi_vector_collection", "metric_types_explained",
        "collection_operations"
    ]),
    ("01_milvus_basics/03_insert_data.py", [
        "generate_mock_embeddings", "prepare_test_data",
        "insert_single_data", "insert_batch_data",
        "insert_with_custom_fields", "insert_with_custom_id",
        "insert_best_practices"
    ]),
    ("01_milvus_basics/04_create_index.py", [
        "generate_mock_embeddings", "prepare_test_data",
        "insert_test_data", "create_flat_index",
        "create_ivf_flat_index", "create_hnsw_index",
        "index_types_explained", "index_management",
        "index_performance_comparison"
    ]),
]

for file_path, expected_funcs in modules_01:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== 02_document_chunking ====================
print("\n" + "=" * 60)
print("测试模块：02_document_chunking")
print("=" * 60)

modules_02 = [
    ("02_document_chunking/01_fixed_chunking.py", [
        "fixed_char_chunking", "demo_fixed_char_chunking",
        "fixed_paragraph_chunking", "demo_fixed_paragraph_chunking",
        "fixed_sentence_chunking", "demo_fixed_sentence_chunking",
        "chunk_file", "demo_file_chunking",
        "fixed_chunking_with_overlap", "demo_overlap_chunking",
        "fixed_chunking_summary"
    ]),
    ("02_document_chunking/02_sliding_window.py", [
        "sliding_window_chunking", "demo_basic_sliding_window",
        "visualize_sliding_window", "demo_visualization",
        "sliding_window_by_words", "demo_word_sliding_window",
        "sliding_window_with_sentence_boundary",
        "demo_sentence_boundary_sliding",
        "compare_sliding_window_params", "demo_parameter_comparison",
        "sliding_window_summary"
    ]),
    ("02_document_chunking/03_ai_chunking.py", [
        "mock_ai_chunking", "demo_mock_ai_chunking",
        "ai_chunking_with_llm", "demo_real_ai_chunking",
        "sentence_clustering_chunking", "demo_sentence_clustering",
        "langchain_text_splitter_demo",
        "compare_chunking_methods", "demo_method_comparison",
        "ai_chunking_best_practices"
    ]),
    ("02_document_chunking/04_summary_chunking.py", [
        "mock_summary_generation", "demo_mock_summary",
        "generate_summary_with_llm", "demo_llm_summary",
        "batch_generate_summaries", "demo_batch_summaries",
        "search_by_summary", "demo_summary_search",
        "rag_with_summary_retrieval", "demo_rag_with_summary",
        "compare_summary_vs_direct_chunking",
        "hybrid_search_with_summary", "demo_hybrid_search"
    ]),
    ("02_document_chunking/05_chunking_comparison.py", [
        "fixed_chunking", "sliding_window_chunking",
        "ai_chunking", "summary_chunking",
        "compare_all_methods", "demo_all_methods",
        "evaluate_chunk_quality", "demo_quality_evaluation",
        "simulate_search", "demo_search_comparison",
        "compare_file_chunking", "demo_file_comparison",
        "comprehensive_comparison_table",
        "chunking_selection_guide"
    ]),
]

for file_path, expected_funcs in modules_02:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== 03_retrieval_methods ====================
print("\n" + "=" * 60)
print("测试模块：03_retrieval_methods")
print("=" * 60)

modules_03 = [
    ("03_retrieval_methods/01_scalar_query.py", [
        "prepare_test_collection", "basic_scalar_query",
        "compound_scalar_query", "range_query",
        "scalar_plus_vector_search", "scalar_query_best_practices"
    ]),
    ("03_retrieval_methods/02_vector_search.py", [
        "prepare_test_collection", "basic_vector_search",
        "metric_type_comparison", "batch_vector_search",
        "search_params_explained", "search_with_real_embedding",
        "vector_search_best_practices"
    ]),
    ("03_retrieval_methods/03_keyword_search.py", [
        "simple_keyword_match", "demo_bm25_search",
        "bm25_with_library", "keyword_vs_vector_search",
        "bm25_parameter_tuning", "keyword_search_best_practices"
    ]),
    ("03_retrieval_methods/04_hybrid_search.py", [
        "simple_hybrid_search", "rrf_fusion",
        "HybridSearcher", "demo_hybrid_searcher",
        "hybrid_search_best_practices", "milvus_hybrid_search_info"
    ]),
    ("03_retrieval_methods/05_rerank.py", [
        "mock_rerank_pipeline", "bge_reranker_demo",
        "cross_encoder_rerank", "rerank_performance_comparison",
        "complete_rag_rerank_pipeline", "rerank_best_practices"
    ]),
]

for file_path, expected_funcs in modules_03:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== 04_rag_api ====================
print("\n" + "=" * 60)
print("测试模块：04_rag_api")
print("=" * 60)

modules_04 = [
    ("04_rag_api/rag_retrieval_api.py", [
        "RAGRetriever", "demo_basic_retrieval",
        "demo_advanced_retrieval", "demo_load_from_file",
        "api_parameters_explained"
    ]),
    ("04_rag_api/rag_qna_api.py", [
        "RAGQnA", "demo_basic_qna", "demo_streaming_qna",
        "demo_custom_prompt", "rag_pipeline_explained"
    ]),
]

for file_path, expected_funcs in modules_04:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== 05_rag_pipeline ====================
print("\n" + "=" * 60)
print("测试模块：05_rag_pipeline")
print("=" * 60)

modules_05 = [
    ("05_rag_pipeline/rag_full_pipeline.py", [
        "RAGPipeline", "demo_full_pipeline", "pipeline_summary"
    ]),
    ("05_rag_pipeline/rag_minimal.py", [
        "SimpleRAG"
    ]),
    ("05_rag_pipeline/rag_step_by_step.py", [
        "step1_load_documents", "step2_document_chunking",
        "step3_embedding", "step4_store_to_milvus",
        "step5_vector_search", "step6_rag_qna",
        "full_pipeline_summary"
    ]),
]

for file_path, expected_funcs in modules_05:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== embedding_examples ====================
print("\n" + "=" * 60)
print("测试模块：embedding_examples")
print("=" * 60)

modules_emb = [
    ("embedding_examples/01_embedding_basics.py", [
        "simulate_embedding_process", "cosine_similarity_demo",
        "embedding_applications", "visualize_embeddings",
        "embedding_models_explained"
    ]),
    ("embedding_examples/02_aliyun_embedding.py", [
        "basic_embedding_with_sdk", "batch_embedding",
        "AliyunEmbedding", "demo_embedding_tool",
        "similarity_applications", "error_handling_and_best_practices"
    ]),
    ("embedding_examples/03_local_embedding.py", [
        "basic_embedding_with_sentence_transformers",
        "batch_embedding", "similarity_calculation",
        "semantic_search_demo", "chinese_embedding_models",
        "local_model_best_practices"
    ]),
    ("embedding_examples/04_embedding_comparison.py", [
        "model_comparison_table", "similarity_comparison_simulated",
        "speed_quality_tradeoff", "cost_comparison",
        "model_selection_decision_tree", "benchmark_recommendations"
    ]),
]

for file_path, expected_funcs in modules_emb:
    syntax_ok, syntax_msg = test_file_syntax(file_path, file_path.replace("/", "_").replace(".", "_"))
    found, missing = test_function_exists(file_path, expected_funcs)

    status = "[OK]" if syntax_ok and not missing else "[WARN]"
    print(f"\n{status} {file_path}")
    print(f"   语法：{syntax_msg}")
    print(f"   函数：{len(found)}/{len(expected_funcs)} 已实现")
    if missing:
        print(f"   缺失：{missing}")

    test_results.append({
        "file": file_path,
        "syntax_ok": syntax_ok,
        "functions_found": len(found),
        "functions_expected": len(expected_funcs),
        "missing_functions": missing
    })

# ==================== 汇总报告 ====================
print("\n" + "=" * 60)
print("测试汇总报告")
print("=" * 60)

total_files = len(test_results)
syntax_ok_files = sum(1 for r in test_results if r["syntax_ok"])
all_funcs_ok = sum(1 for r in test_results if not r["missing_functions"])

print(f"""
测试文件总数：{total_files}
语法正确：{syntax_ok_files}/{total_files}
函数完整：{all_funcs_ok}/{total_files}
""")

# 按模块统计
modules = {}
for r in test_results:
    module = r["file"].split("/")[0]
    if module not in modules:
        modules[module] = {"total": 0, "syntax_ok": 0, "funcs_ok": 0}
    modules[module]["total"] += 1
    if r["syntax_ok"]:
        modules[module]["syntax_ok"] += 1
    if not r["missing_functions"]:
        modules[module]["funcs_ok"] += 1

print("\n按模块统计:")
for module, stats in sorted(modules.items()):
    print(f"  {module}: {stats['syntax_ok']}/{stats['total']} 语法正确，{stats['funcs_ok']}/{stats['total']} 函数完整")

# 有问题需要修复的文件
issues = [r for r in test_results if not r["syntax_ok"] or r["missing_functions"]]
if issues:
    print("\n需要关注的文件:")
    for r in issues:
        if not r["syntax_ok"]:
            print(f"  [ERROR] {r['file']} - 语法错误")
        elif r["missing_functions"]:
            print(f"  [WARN] {r['file']} - 缺失函数：{r['missing_functions']}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
