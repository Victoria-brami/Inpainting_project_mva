from code.parser.evaluation import parser
from code.completion_and_patchmatch_eval import comparison_evaluate
import os

import numpy as np#os.system('pip install torch==1.9.1')

def main():
    parameters = parser()
    print("\n", parameters)
    print("Parser defined")
    # Eval = ModelEvaluation()
    np.random.seed(parameters["seed"])
    comparison_evaluate(parameters)
    print("\n   Evaluation DONE !!!")


if __name__ == '__main__':
    main()