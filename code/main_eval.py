from code.parser.evaluation import parser
from code.completion_eval import evaluate
import os
#os.system('pip install torch==1.9.1')

def main():
    parameters = parser()
    print("Parser defined")
    # Eval = ModelEvaluation()
    print("DEFINED EVAL CLASS")
    evaluate(parameters)
    print("Evaluation DONE, please see in {}".format())


if __name__ == '__main__':
    main()