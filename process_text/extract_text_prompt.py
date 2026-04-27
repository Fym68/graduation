"""
从 metadata_3d.csv 的 Imaging Description 中，用 LLM API（OpenAI 兼容接口）提取每个 T2S 病人的
宫颈肿瘤分割相关文本描述（一句话），保存为 JSON 供后续 BiomedCLIP 编码使用。

Usage:
    python extract_text_prompt.py --api_key sk-xxxxx [--model deepseek-chat] [--dry_run]

Output format (patient_text_prompts.json):
    {
        "10146710": "Cervical soft tissue mass approximately 49mm × 33mm, T2 slightly hyperintense, ...",
        "10147939": "Cervical mucosal thickening approximately 5mm, no definite mass formation.",
        ...
    }
"""

import argparse
import json
import os
import time

import pandas as pd

METADATA_PATH = "/home/fym/Nas/fym/datasets/graduation/metadata_3d.csv"
OUTPUT_DIR = "/home/fym/graduation/process_text"

SYSTEM_PROMPT = """\
You are a radiology report parser specialized in cervical cancer MRI.

IMPORTANT CONTEXT: All patients in this dataset are confirmed cervical cancer cases. Even if the imaging description uses uncertain language (e.g., "considering", "suspicious for", "cannot rule out"), the patient has been clinically diagnosed with cervical cancer. Your extraction should reflect this — describe the lesion as a cervical cancer tumor, not as a suspected finding.

Your task: given a full MRI imaging description of a cervical cancer patient, extract ONLY the information useful for segmenting the cervical tumor on T2-weighted sagittal (T2S) slices.

## Output rules
- Output exactly ONE English sentence, no more than 50 words.
- The sentence should read like a concise radiology finding focused on the cervical lesion.
- Do NOT output any explanation, preamble, or formatting — just the sentence itself.

## What to extract (if mentioned)
1. Tumor location within the cervix (anterior/posterior lip, lateral wall, etc.)
2. Approximate tumor size
3. T2 signal characteristics
4. Invasion extent (vaginal involvement, uterine body involvement, serosal breakthrough)
5. Boundary clarity or margin description

## What to EXCLUDE
- T1 signal, DWI, contrast enhancement findings
- Lymph nodes, bladder, rectum, bone, adnexa, brain, liver, spleen, kidney findings
- Any organ or structure unrelated to the cervical lesion itself

## Edge cases
- If the description mentions only subtle cervical changes (e.g., mucosal thickening, wall irregularity) without a clear mass, describe those changes — they represent the tumor in early stage.
- If the description contains NO cervical findings at all (e.g., only brain or abdominal content), output exactly: "Cervical cancer confirmed but no cervical lesion detail in this imaging report."
- If T2 signal is not explicitly mentioned, omit it from your output rather than guessing.
- If tumor size is given in multiple dimensions, keep at most the two largest dimensions.\
"""

USER_TEMPLATE = """\
Patient imaging description:
{description}\
"""


def build_user_message(desc):
    return USER_TEMPLATE.format(description=desc)


def call_llm(client, model, user_msg, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                stream=False,
            )
            if isinstance(resp, str):
                return resp.strip().strip('"')
            return resp.choices[0].message.content.strip().strip('"')
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise

"""
python extract_text_prompt.py --api_key sk-HKIVHfm2bTl6BgoY5nwQyjMkKOOQD6QNwZgfNVT4cpHzOQJj --model claude-sonnet-4-5-20250929

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--base_url", type=str, default="https://api.kwwai.top/v1")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--output", type=str, default="patient_text_prompts.json")
    parser.add_argument("--dry_run", action="store_true", help="Print prompt for first patient without calling API")
    args = parser.parse_args()

    df = pd.read_csv(METADATA_PATH)
    t2s = df[df["has_T2S"] == True].copy()
    print(f"Total T2S patients: {len(t2s)}")

    output_path = os.path.join(OUTPUT_DIR, args.output)

    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results, will skip them")

    remaining = t2s[~t2s["Patient ID"].astype(str).isin(existing.keys())]
    print(f"Remaining to process: {len(remaining)}")

    if len(remaining) == 0:
        print("All done!")
        return

    if args.dry_run:
        row = remaining.iloc[0]
        msg = build_user_message(row["Imaging Description"])
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== SAMPLE USER MESSAGE ===")
        print(msg)
        print(f"\n=== Estimated input tokens: ~{(len(SYSTEM_PROMPT) + len(msg)) // 4}")
        return

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    all_results = dict(existing)

    for i, (_, row) in enumerate(remaining.iterrows()):
        pid = str(row["Patient ID"]).strip()
        desc = row["Imaging Description"]
        user_msg = build_user_message(desc)
        print(f"[{i+1}/{len(remaining)}] {pid}...", end=" ", flush=True)

        result = call_llm(client, args.model, user_msg)
        all_results[pid] = result
        print(f"OK: {result[:80]}...")

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(all_results)} patients saved to {output_path}")


if __name__ == "__main__":
    main()
