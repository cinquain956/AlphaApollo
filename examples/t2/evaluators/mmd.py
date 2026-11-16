from alphaapollo.experiments.t2.evaluation import evaluate_program_path


INSTANCE = {
    "n": 14,
    "dimensions": 3,
    "benchmark": 0.2400470626477107,
}


def evaluate(program_path):
    return evaluate_program_path(program_path, "mmd", INSTANCE, 30.0)
