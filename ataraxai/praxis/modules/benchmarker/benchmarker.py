from ataraxai.hegemonikon_py import HegemonikonQuantizedModelInfo, HegemonikonBenchmarkMetrics, HegemonikonBenchmarkResult, HegemonikonBenchmarkParams, HegemonikonLlamaModelParams, LlamaBenchmarker  # type: ignore


class BenchmarkRunner:
    def __init__(self, model_info, benchmark_params, llama_model_params):
        self.model_info = model_info
        self.benchmark_params = benchmark_params
        self.llama_model_params = llama_model_params

    def run(self):
        pass