import os
import pandas as pd
from typing import Literal
from pydantic import BaseModel
from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt

client = genai.Client()
MODEL_ID = 'gemini-1.5-flash'

AllObjectParts = Literal[
    'front_bumper', 'rear_bumper', 'door', 'hood', 'windshield', 'side_mirror', 'headlight', 'taillight', 'fender', 'quarter_panel', 'body',
    'screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base',
    'box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item', 
    'unknown'
]

class ClaimReviewOutput(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: Literal['dent', 'scratch', 'crack', 'glass_shatter', 'broken_part', 'missing_part', 'torn_packaging', 'crushed_packaging', 'water_damage', 'stain', 'none', 'unknown']
    object_part: AllObjectParts
    claim_status: Literal['supported', 'contradicted', 'not_enough_information']
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: Literal['none', 'low', 'medium', 'high', 'unknown']

def load_data():
    claims = pd.read_csv('dataset/claims.csv')
    history = pd.read_csv('dataset/user_history.csv').set_index('user_id')
    reqs = pd.read_csv('dataset/evidence_requirements.csv')
    return claims, history, reqs

def get_user_history(user_id, history_df):
    if user_id in history_df.index:
        row = history_df.loc[user_id]
        return f"Past Claims: {row['past_claim_count']}, Rejected: {row['rejected_claim']}, Flags: {row['history_flags']}"
    return "No prior history."

def get_evidence_reqs(claim_object, reqs_df):
    relevant_reqs = reqs_df[(reqs_df['claim_object'] == claim_object) | (reqs_df['claim_object'] == 'all')]
    return "\n".join([f"- If issue is '{row['applies_to']}': {row['minimum_image_evidence']}" for _, row in relevant_reqs.iterrows()])

def prepare_images(image_paths_str):
    if pd.isna(image_paths_str): return []
    paths = [p.strip() for p in image_paths_str.split(';')]
    return [os.path.join('dataset', p) if not p.startswith('dataset/') else p for p in paths]

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def process_claim_with_llm(user_claim, claim_object, history_context, reqs_context, image_filepaths):
    system_instruction = """
    You are an expert fraud and damage claim reviewer. Verify damage claims using images, claim context, user history, and evidence requirements.
    The IMAGES are the primary source of truth.
    1. If images do not meet the minimum requirements, set evidence_standard_met to false and claim_status to not_enough_information.
    2. Extract image IDs from filenames (e.g., 'dataset/images/test/case_001/img_1.jpg' -> 'img_1'). Use semicolons for multiples. Use 'none' if none support.
    3. Ensure risk_flags are semicolon-separated.
    """
    prompt = f"Object: {claim_object}\nClaim: {user_claim}\nHistory: {history_context}\nRequirements: {reqs_context}"
    
    uploaded_files = [client.files.upload(file=p) for p in image_filepaths if os.path.exists(p)]
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt] + uploaded_files,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ClaimReviewOutput,
                temperature=0.1
            ),
        )
        for f in uploaded_files: client.files.delete(name=f.name)
        return response.parsed
    except Exception as e:
        for f in uploaded_files: client.files.delete(name=f.name)
        raise e

def main():
    claims_df, history_df, reqs_df = load_data()
    results = []
    
    for index, row in claims_df.iterrows():
        print(f"Processing {index + 1}/{len(claims_df)}...")
        history_context = get_user_history(row['user_id'], history_df)
        reqs_context = get_evidence_reqs(row['claim_object'], reqs_df)
        image_filepaths = prepare_images(row['image_paths'])
        
        try:
            llm_result = process_claim_with_llm(row['user_claim'], row['claim_object'], history_context, reqs_context, image_filepaths)
            result_dict = row.to_dict()
            result_dict.update(llm_result.model_dump())
            results.append(result_dict)
        except Exception as e:
            print(f"Failed row {index}: {e}")
            fallback = row.to_dict()
            fallback.update({
                'evidence_standard_met': False, 'evidence_standard_met_reason': 'Error', 'risk_flags': 'manual_review_required',
                'issue_type': 'unknown', 'object_part': 'unknown', 'claim_status': 'not_enough_information',
                'claim_status_justification': 'Failed', 'supporting_image_ids': 'none', 'valid_image': False, 'severity': 'unknown'
            })
            results.append(fallback)

    pd.DataFrame(results).to_csv("output.csv", index=False)
    print("Completed! Saved to output.csv")

if __name__ == "__main__":
    main()
