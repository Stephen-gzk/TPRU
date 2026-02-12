import re
from typing import Dict, List

def format_reward(predict: str) -> float:
    """
    Check if the prediction follows the required format with <think> and <answer> tags.
    The answer can be a single uppercase letter (for MCQ) or a sequence of 
    uppercase letters (for ordering tasks).
    """
    # Modified the regex to accept one or more uppercase letters ([A-Z]+) in the answer tag.
    # This pattern now matches both "C" and "CABD".
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>\s*([A-Z]+)\s*</answer>$", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    # The prompt implies a binary reward for format correctness.
    # We will maintain the 0.2 reward if format is correct.
    return 0.2 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Check if the answer inside <answer> tags matches the ground truth.
    This function is already general enough to work for both MCQ and ordering tasks.
    """
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict, re.DOTALL)
    if not answer_match:
        return 0.0
    
    # Extracts the answer, which could be "C" or "CABD", and compares to ground truth.
    answer = answer_match.group(1).strip().strip('"\'')
    return 1.0 if answer == ground_truth else 0.0

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute scores for each prediction based on format and accuracy.
    This function remains unchanged as its logic correctly calls the reward functions.
    """
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        # Normalize whitespace around tags for more robust matching
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
