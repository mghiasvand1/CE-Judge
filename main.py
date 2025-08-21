from deep_translator import GoogleTranslator
from collections import defaultdict
from scipy.stats import kendalltau
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import argparse
import json
import re


with open("prompts/prompts.json", "r") as f:
    PROMPTS = json.load(f)

API_KEY = None
client = None

def inference(prompt, judge=False, _type=None):
    global client
    if judge:
        if _type == "pointwise":
            system_prompt = "As a literary translation critic, your role is to identify errors and evaluate the translation’s quality. Focus on the subtleties of literary style, emotional impact, and creative expression. An excellent translation captures the original work’s aesthetic qualities, tone, and cultural nuances, rather than adhering to a word-for-word approach. Translations that are excessively literal and fail to adapt to the target language’s literary conventions and natural flow should be critiqued accordingly."
        elif _type == "pairwise":
            system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    chat_completion_res = client.chat.completions.create(
        model="qwen/qwen2.5-7b-instruct",
        messages=messages,
        max_tokens=8192,
        temperature=0,
        top_p=1,
        seed=42,
    )
    response = chat_completion_res.choices[0].message.content
    return response

def concept(_type, task, inputs):
    concepts = []
    for _input in tqdm(inputs):
        prompt = PROMPTS["concepts"][_type][task].replace("[INPUT]", _input)
        concepts.append(inference(prompt))
    return concepts

def translate(inputs, lang):
    if lang == "en":
        return inputs
    lang = lang if lang != "zh" else "zh-CN"
    inputs = [text[:5000] if text not in [None, ""] else "None" for text in inputs]
    translator = GoogleTranslator(source=lang, target="en")
    translations = []
    for i in range(0, len(inputs), 10):
        batch = inputs[i:i + 10]
        try:
            translated_batch = translator.translate_batch(batch)
            for item in translated_batch:
                translations.append(item) 
        except Exception as batch_error:
            for text in batch:
                try:
                    translations.append(translator.translate(text))
                except Exception as single_error:
                    translations.append("None")
    return translations

def checklist(_type, task, inputs, concepts, lang, _case):
    checklists = []
    inputs = translate(inputs, lang)
    for i, _input in enumerate(tqdm(inputs)):
        prompt = PROMPTS["checklist"][_type][task][_case] \
        .replace("[INPUT]", _input) \
        .replace("[CONCEPTS]", concepts[i])
        checklists.append(inference(prompt))
    return checklists

def judge(_type, task, inputs, responses, checklists_ri, checklists_ir):
    pred = []
    if _type == "pointwise":
        checklists = [checklists_ri[0][i] + "\n" + checklists_ir[0][i] for i in range(len(checklists_ri[0]))]
    elif _type == "pairwise":
        checklists_a = [checklists_ri[0][i] + "\n" + checklists_ir[0][i] for i in range(len(checklists_ri[0]))]
        checklists_b = [checklists_ri[1][i] + "\n" + checklists_ir[1][i] for i in range(len(checklists_ri[1]))]
    for i, _input in enumerate(tqdm(inputs)):
        prompt = PROMPTS["judge"][_type][task] \
        .replace("[INPUT]", _input) 
        if _type == "pointwise":
            prompt = prompt.replace("[RESPONSE]", responses[0][i]).replace("[CHECKLIST]", checklists[i])
        elif _type == "pairwise":
            prompt = prompt.replace("[RESPONSE_A]", responses[0][i]).replace("[RESPONSE_B]", responses[1][i])
            prompt = prompt.replace("[CHECKLIST_A]", checklists_a[i]).replace("[CHECKLIST_B]", checklists_b[i])

        res = inference(prompt, judge=True, _type=_type)

        if _type == "pointwise":
            match = re.search(r"Score: (\d+)", res, re.IGNORECASE)
        elif _type == "pairwise":
            match = re.search(r"Therefore, I choose\s+([A-B])\s+as the better response", res, re.IGNORECASE)

        if match:
            pred.append(match.group(1))  
        else:
            pred.append("None")
            print(res)

    return pred

def evaluate(label_file, _type):
    pairs = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if not (line := line.strip()):
                continue
            if "," not in line:
                continue
            a, b = map(str.strip, line.split(",", 1))
            pairs.append((a, b))
    preds, golds = zip(*pairs)

    if _type == "pointwise":
        p = list(map(int, preds))
        g = list(map(int, golds))
        print(f"{kendalltau(p, g)[0]:.3f}")

    elif _type == "pairwise":
        print(f"{sum(p == g for p, g in zip(preds, golds)) / len(preds):.3f}")

def main(dataset, lang, _type):
    if dataset == "mmeval":
        test_data = load_dataset("prometheus-eval/MM-Eval", split="test") \
            .filter(lambda example: example["language"] == lang) \
            .remove_columns(["language", "chosen_model", "rejected_model", "id", "__index_level_0__"])
        data_list = test_data.to_list()
        grouped_data = defaultdict(list)
        for row in data_list:
            key = row["subset"]
            if key in ["reasoning", "chat"]:
                grouped_data[key].append(row)
        grouped_data = dict(grouped_data)
        for subset_name, rows in grouped_data.items():
            inst = [r["prompt"] for r in rows]
            ch = [r["chosen"] for r in rows]
            rj = [r["rejected"] for r in rows]

            concepts_i = concept(_type, subset_name, inst)
            concepts_r_a = concept(_type, subset_name, ch)
            concepts_r_b = concept(_type, subset_name, rj)
            
            checklists_ri_a =  checklist(_type, subset_name, ch, concepts_i, lang, _case = "R->I")
            checklists_ir_a = checklist(_type, subset_name, inst, concepts_r_a, lang, _case = "I->R")
            checklists_ri_b = checklist(_type, subset_name, rj, concepts_i, lang, _case = "R->I")
            checklists_ir_b = checklist(_type, subset_name, inst, concepts_r_b, lang, _case = "I->R")

            pred = judge(_type, subset_name, inst, [ch, rj], [checklists_ri_a, checklists_ri_b], [checklists_ir_a, checklists_ir_b])

            filename = f"labels/{dataset}_{lang}_{subset_name}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for p in pred:
                    f.write(f"{p}, A\n")
            
    elif dataset == "liteval":
        names = ["de-en", "de-zh", "en-de", "en-zh"]
        for name in names:  
            if name in ["de-zh", "en-de"]:
                df = pd.read_csv(f"{name}.csv", delimiter=';', encoding="utf-8")
            elif name in ["de-en", "en-zh"]:
                df = pd.read_csv(f"{name}.csv", encoding="utf-8")
            sources = df['source'].tolist()
            targets = df['tgt'].tolist()
            ratings = df['rating'].tolist()

            concepts_i = concept(_type, "mt", sources)
            concepts_r = concept(_type, "mt", targets)

            checklists_ri =  checklist(_type, "mt", targets, concepts_i, name.split("-")[1], _case = "R->I")
            checklists_ir = checklist(_type, "mt", sources, concepts_r, name.split("-")[0], _case = "I->R")

            pred = judge(_type, "mt", sources, [targets], [checklists_ri], [checklists_ir])
            
            filename = f"labels/{dataset}_{name}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for i, p in enumerate(pred):
                    f.write(f"{p}, {ratings[i]}\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FC-Judge")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    inf_parser = subparsers.add_parser("inference")
    inf_parser.add_argument("--dataset", choices=["mmeval", "liteval"], required=True)
    inf_parser.add_argument("--lang", type=str, required=True)
    inf_parser.add_argument("--type", choices=["pointwise", "pairwise"], required=True)
    inf_parser.add_argument("--api_key", type=str, required=True)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--label_file", type=str, required=True)
    eval_parser.add_argument("--type", choices=["pointwise", "pairwise"], required=True)

    args = parser.parse_args()

    if args.mode == "inference":
        API_KEY = args.api_key
        client = OpenAI(base_url="https://api.novita.ai/v3/openai", api_key=API_KEY)
        main(dataset=args.dataset, lang=args.lang, _type=args.type)

    elif args.mode == "evaluate":
        evaluate(label_file=args.label_file, _type=args.type)

