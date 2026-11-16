from alphaapollo.experiments.t2.evaluation import evaluate_program_path


INSTANCE = {
    "n": 26,
    "container_type": "rectangle",
    "width": 1.0,
    "height": 1.0,
    "tolerance": 1.0e-6,
    "benchmark": 2.635,
}


def evaluate(program_path):
    return evaluate_program_path(program_path, "circle_packing", INSTANCE, 30.0)
