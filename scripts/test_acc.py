#!/usr/bin/env python
"""
Generative evaluation, stage 2 of 2: score the responses scripts/gen_vllm.py sampled.

The answer-extraction functions below are the migration/test_acc.py reference, verbatim: they
ARE the metric, so any "cleanup" here would silently move every number this repo reports.
Only the reporting around them is new — a per-task accuracy table plus a JSON summary so
results land next to the other benchmark artifacts instead of only in a log.

  python scripts/test_acc.py --input_file results/commonsense/saltq_int3.jsonl \
      --output_json results/commonsense/saltq_int3.json
"""

import argparse
import json
import os
import re
from fractions import Fraction
from collections import Counter, defaultdict
from datetime import datetime

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string
    
def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        #pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
    
def process_math_results(completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        return False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def extract_commonsense_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['siqa', 'arc_challenge', 'arc_easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


COMMONSENSE_TASKS = [
    'boolq', 'piqa', 'siqa', 'arc_challenge', 'arc_easy', 'openbookqa',
    'hellaswag', 'winogrande',
]


def score_record(data: dict) -> bool:
    """Score one generation against its gold answer, dispatching on the task type."""
    task = data['type']
    if task == 'gsm8k':
        y_pred = extract_answer_number(data['output'])
        return y_pred is not None and float(y_pred) == float(data['answer'])
    if task == 'math':
        return process_math_results(data['output'], data['answer'])
    if task in COMMONSENSE_TASKS:
        y_pred = extract_commonsense_answer(task, data['output'])
        return y_pred == data['answer']
    raise ValueError(f"No scorer for task type '{task}'.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='The .jsonl written by scripts/gen_vllm.py.')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Where to write the accuracy summary. Default: alongside '
                             '--input_file with a .json suffix.')
    parser.add_argument('--model_path', type=str, default='',
                        help='The model these responses came from. Pass the PRE-fold export dir '
                             '(...-deploy-eval): scripts/collect_saltq_results.py recovers bits / '
                             'group_size / the freedom split from the artifacts next to it, and '
                             'the folded -vllm dir deliberately carries none of them.')
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    results = defaultdict(list)
    outputs = defaultdict(list)      # raw generations, for the degeneracy check below
    golds = defaultdict(list)
    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results[data['type']].append(score_record(data))
                outputs[data['type']].append((data.get('output') or '').strip())
                golds[data['type']].append((data.get('answer') or '').strip())

    summary = {}
    collapsed = []
    for task, hits in sorted(results.items()):
        acc = sum(hits) / len(hits)
        # A model that emits ONE answer for every question scores the frequency of that answer in
        # the gold labels, which looks like a real number and is not one. The INT3 g64 commonsense
        # run reported 37.03% mean this way: always "true" on boolq (62.2% = the true-rate),
        # always "answer1" on arc_challenge (22.7%), and so on for all 8 tasks. Nothing in a
        # per-task accuracy table shows this, so it is measured here instead of eyeballed.
        n_distinct = len(set(outputs[task]))
        majority = max(Counter(golds[task]).values()) / len(golds[task]) if golds[task] else 0.0
        summary[task] = {'n': len(hits), 'acc': acc,
                         'n_distinct_outputs': n_distinct,
                         'majority_class_acc': majority}
        flag = ''
        if n_distinct <= 2:
            collapsed.append(task)
            flag = f'  <-- only {n_distinct} distinct output(s)'
        print(f"{task:16s} n={len(hits):6d}  acc={acc * 100:6.2f}  "
              f"(majority-class {majority * 100:5.2f}, {n_distinct} distinct){flag}")

    total = sum(v['n'] for v in summary.values())
    if total:
        # Unweighted mean over tasks — the convention the commonsense literature reports, and
        # not the same as pooled accuracy, since the task sizes differ by ~20x.
        mean_acc = sum(v['acc'] for v in summary.values()) / len(summary)
        summary['_mean'] = {'n': total, 'acc': mean_acc}
        print(f"{'MEAN (unweighted)':16s} n={total:6d}  acc={mean_acc * 100:6.2f}")

    if collapsed:
        summary['_degenerate_tasks'] = sorted(collapsed)
        print()
        print(f"*** WARNING: {len(collapsed)} of {len(results)} tasks got <=2 distinct outputs "
              f"across the whole split: {', '.join(sorted(collapsed))}")
        print("*** The model learned the RESPONSE TEMPLATE and a per-task prior, not the task. "
              "Its accuracy is the majority-class rate and means nothing.")
        print("*** Note the training loss will look healthy: the response is ~8 tokens of which "
              "~7 are template, so missing only the answer token costs ~0.09 nats.")

    out_json = args.output_json or os.path.splitext(args.input_file)[0] + '.json'
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    payload = {
        # `source` is the discriminator scripts/collect_saltq_results.py dispatches on: this
        # file and an lm-eval result file are both results/**/*.json and are NOT the same shape.
        'source': 'vllm-generative',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'model_path': args.model_path,
            'dataset': args.dataset,
            'input_file': args.input_file,
        },
        'results': summary,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n[acc] summary -> {out_json}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
