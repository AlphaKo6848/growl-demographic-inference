##### Block 1 : Import external tools and Access to Claude API #####
import anthropic
import csv
import json
import os
from collections import defaultdict

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

##### Block 2 : Define IAB Audience Taxonomy #####
AGE_SEGMENTS = {
    "3": "Age: 18-20", "4": "Age: 21-24", "5": "Age: 25-29",
    "6": "Age: 30-34", "7": "Age: 35-39", "8": "Age: 40-44",
    "9": "Age: 45-49", "10": "Age: 50-54", "11": "Age: 55-59",
    "12": "Age: 60-64", "13": "Age: 65-69", "14": "Age: 70-74", "15": "Age: 75+"
}

GENDER_SEGMENTS = {
    "50": "Gender: Male", "49": "Gender: Female", "51": "Gender: Other"
}

IAB_CATEGORIES = {
    "IAB1": "Arts & Entertainment", "IAB2": "Automotive", "IAB3": "Business",
    "IAB4": "Careers", "IAB5": "Education", "IAB6": "Family & Parenting",
    "IAB7": "Health & Fitness", "IAB8": "Food & Drink", "IAB9": "Hobbies & Interests",
    "IAB10": "Home & Garden", "IAB11": "Law, Government, & Politics", "IAB12": "News",
    "IAB13": "Personal Finance", "IAB14": "Society", "IAB15": "Science",
    "IAB16": "Pets", "IAB17": "Sports", "IAB18": "Style & Fashion",
    "IAB19": "Technology & Computing", "IAB20": "Travel", "IAB21": "Real Estate",
    "IAB22": "Shopping", "IAB23": "Religion & Spirituality", "IAB24": "Uncategorized",
    "IAB25": "Non-Standard Content", "IAB26": "Illegal Content"
}

IAB_SUBCATEGORIES = {
    "IAB1-1": "Books & Literature", "IAB1-2": "Celebrity Fan/Gossip", "IAB1-3": "Fine Art",
    "IAB1-4": "Humor", "IAB1-5": "Movies", "IAB1-6": "Music", "IAB1-7": "Television",
    "IAB2-1": "Auto Parts", "IAB2-2": "Auto Repair", "IAB2-3": "Buying/Selling Cars",
    "IAB2-4": "Car Culture", "IAB2-15": "Motorcycles", "IAB2-17": "Performance Vehicles",
    "IAB7-1": "Exercise", "IAB7-18": "Depression", "IAB7-32": "Nutrition",
    "IAB7-37": "Psychology/Psychiatry", "IAB7-42": "Substance Abuse", "IAB7-44": "Weight Loss",
    "IAB7-45": "Women's Health", "IAB8-1": "American Cuisine", "IAB8-8": "Desserts & Baking",
    "IAB8-9": "Dining Out", "IAB9-5": "Board Games/Puzzles", "IAB9-25": "Roleplaying Games",
    "IAB9-26": "Sci-Fi & Fantasy", "IAB9-30": "Video & Computer Games",
    "IAB10-5": "Home Repair", "IAB10-7": "Interior Decorating", "IAB10-9": "Remodeling & Construction",
    "IAB11-1": "Immigration", "IAB11-2": "Legal Issues", "IAB11-4": "Politics",
    "IAB13-1": "Beginning Investing", "IAB13-2": "Credit/Debt & Loans",
    "IAB13-4": "Financial Planning", "IAB13-7": "Investing", "IAB13-12": "Tax Planning",
    "IAB14-1": "Dating", "IAB14-3": "Gay Life", "IAB14-4": "Marriage",
    "IAB15-6": "Physics", "IAB15-7": "Space/Astronomy",
    "IAB17-12": "Football", "IAB17-18": "Hunting/Shooting", "IAB17-26": "Pro Basketball",
    "IAB17-44": "World Soccer", "IAB18-1": "Beauty", "IAB18-3": "Fashion",
    "IAB19-4": "C/C++", "IAB19-6": "Cell Phones", "IAB19-8": "Computer Networking",
    "IAB19-18": "Internet Technology", "IAB19-20": "JavaScript", "IAB19-25": "Network Security",
    "IAB19-27": "PC Support", "IAB19-34": "Web Design/HTML",
    "IAB20-1": "Adventure Travel", "IAB20-3": "Air Travel", "IAB20-7": "Business Travel",
    "IAB23-5": "Christianity", "IAB23-7": "Islam",
    "IAB25-2": "Extreme Graphic/Explicit Violence", "IAB25-3": "Pornography",
    "IAB25-4": "Profane Content", "IAB26-1": "Illegal Content"
}

##### Block 3 : Read CSV and Group by user_id #####
conversations = defaultdict(list)

with open('user-chats-spark_original.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = row['user_id']
        conversations[user_id].append({
            'chat_id': row['chat_id'],
            'role': row['role'],
            'text': row['text'],
            'created_at': row['created_at']
        })

##### Block 4 : Resume logic - skip already processed users #####
OUTPUT_FILE = 'user_profiles_final.csv'
processed_users = set()

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_users.add(row['user_id'])
    print(f"Already processed users: {len(processed_users)} users - skipped")


##### Block 5 : First inference prompt #####
def build_first_pass_prompt(conversation_text):
    return f"""
You are analyzing a chatbot conversation to infer the user's age, gender, and content category for ad targeting.
Only analyze the USER's messages, not the assistant's responses.

Conversation:
{conversation_text}

=== AGE INFERENCE RULES ===
A-rule 1: Use of slang, internet abbreviations, or generational references
A-rule 2: Typing style — very short/fragmented suggests younger; longer formal suggests older
A-rule 3: Spelling patterns — intentional informal suggests younger; accidental errors may suggest older
A-rule 4: Vocabulary complexity — simpler may suggest younger or less educated
A-rule 5: Time of conversation — late night suggests younger; early morning suggests older/working adults
A-rule 6: Topics — school/gaming/anime suggests teens; career/mortgage/parenting suggests 30s-40s; health/retirement suggests 50s+
A-rule 7: Emoji usage — heavy emoji suggests younger
A-rule 8: Formality/honorifics — suggests older in cultures where this applies
A-rule 9: Language-specific age markers — Arabic, Japanese, Korean grammar forms
A-rule 10: Message gaps — longer gaps may suggest slower typing = older users

=== GENDER INFERENCE RULES ===
G-rule 1: Grammatical gender markers — verb forms, adjective agreement (Arabic, French, Spanish, etc.)
G-rule 2: Topic — firearms/military suggests male; pregnancy/beauty/childcare suggests female
G-rule 3: Direct self-identification — name, pronoun, role mention
G-rule 4: Relationship references — "my boyfriend/husband" suggests female; "my girlfriend/wife" suggests male
G-rule 5: Occupational references — statistically gendered professions
G-rule 6: Emotional expression — direct emotional language more common in female
G-rule 7: Writing tone — aggressive/transactional weakly suggests male; nurturing/relational weakly suggests female
G-rule 8: Request type — technical/gaming skews male; appearance/emotional support skews female

=== CONTENT CATEGORY INFERENCE RULES ===
C-rule 1: Identify the dominant topic of the entire conversation
C-rule 2: Use most specific subcategory available
C-rule 3: If multiple topics, pick the dominant one
C-rule 4: Sexually explicit → IAB25-3; violent → IAB25-2; illegal → IAB26-1
C-rule 5: Too generic/unclear → IAB24

=== OUTPUT FORMAT ===
Age options: {json.dumps(AGE_SEGMENTS, ensure_ascii=False)}
Gender options: {json.dumps(GENDER_SEGMENTS, ensure_ascii=False)}
IAB Categories: {json.dumps(IAB_CATEGORIES, ensure_ascii=False)}
IAB Subcategories: {json.dumps(IAB_SUBCATEGORIES, ensure_ascii=False)}

Return ONLY a JSON object:
{{
    "age_distribution": {{"10s": 0, "20s": 0, "30s": 0, "40s": 0, "50s": 0, "60s+": 0}},
    "age_id": "...",
    "age_label": "Age: XX-XX",
    "age_confidence": "high/medium/low",
    "age_inference_keywords": ["keyword1", "keyword2", "keyword3"],
    "gender_id": "...",
    "gender_label": "Gender: Male/Female/Other",
    "gender_confidence": "high/medium/low",
    "gender_inference_keywords": ["keyword1", "keyword2", "keyword3"],
    "content_cat": "IAB category code",
    "content_subcat": "IAB subcategory code",
    "content_keywords": ["keyword1", "keyword2", "keyword3"],
    "reasoning": "step-by-step reasoning referencing A-rules, G-rules, C-rules"
}}
"""

##### Block 6 : Second inference prompt (Newly added) #####
# Condition: age_confidence = low + gender_confidence in [high, medium]
# P(age | gender=X, content=Y, conversation) -> re-infer user's age
def build_reestimate_prompt(conversation_text, first_pass):
    return f"""
You are re-estimating a user's AGE ONLY using Bayesian reasoning.

The following were already determined with confidence in a first pass — treat these as FIXED PRIORS:
- Gender: {first_pass['gender_label']} (confidence: {first_pass['gender_confidence']})
- Content Category: {first_pass['content_cat']} / {first_pass['content_subcat']}
- Gender signals: {first_pass['gender_inference_keywords']}

The first-pass age estimate had LOW confidence: {first_pass['age_label']}

Given these fixed priors, apply conditional reasoning:
P(age | gender={first_pass['gender_label']}, content={first_pass['content_cat']}, conversation)

Ask yourself:
- Who typically uses this content category AND is this gender?
- What age group most commonly exhibits this gender+content combination?
- Does the conversation style (length, vocabulary, language) narrow the age further?

Conversation:
{conversation_text}

Age options: {json.dumps(AGE_SEGMENTS, ensure_ascii=False)}

Return ONLY a JSON object (age fields only):
{{
    "age_distribution": {{"10s": 0, "20s": 0, "30s": 0, "40s": 0, "50s": 0, "60s+": 0}},
    "age_id": "...",
    "age_label": "Age: XX-XX",
    "age_confidence": "medium or high",
    "age_inference_keywords": ["keyword supporting age given gender+content", "keyword2", "keyword3"],
    "reestimate_reasoning": "Bayesian reasoning: P(age | gender, content) = ..."
}}
"""

##### Block 7 : Inference Loops #####
results = []

for user_id, messages in conversations.items():

    # Resume: 이미 처리된 유저 스킵
    if user_id in processed_users:
        continue

    messages.sort(key=lambda x: x['created_at'])
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg['role']} : {msg['text']}\n\n"

    ### First inference
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # 비용 절감: haiku 사용
        max_tokens=1000,
        messages=[{"role": "user", "content": build_first_pass_prompt(conversation_text)}]
    )

    raw = response.content[0].text
    try:
        json_str = raw[raw.index('{'):raw.rindex('}') + 1]
        parsed = json.loads(json_str) ### parsed will be used for the second inference as first_pass
    except:
        parsed = {"error": raw}

    ### Second inference
    ### Run only if age_confidence=low and gender_confidence=high/medium
    reestimated = False
    if (parsed.get('age_confidence') == 'low' and parsed.get('gender_confidence') in ['high', 'medium'] and 'error' not in parsed):
        re_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": build_reestimate_prompt(conversation_text, parsed)}]
        )

        re_raw = re_response.content[0].text
        try:
            re_json_str = re_raw[re_raw.index('{'):re_raw.rindex('}') + 1]
            re_parsed = json.loads(re_json_str)

            ### Update age by the second inference  (gender/content is maintained)
            parsed['age_distribution'] = re_parsed.get('age_distribution', parsed['age_distribution'])
            parsed['age_id'] = re_parsed.get('age_id', parsed['age_id'])
            parsed['age_label'] = re_parsed.get('age_label', parsed['age_label'])
            parsed['age_confidence'] = re_parsed.get('age_confidence', parsed['age_confidence'])
            parsed['age_inference_keywords'] = re_parsed.get('age_inference_keywords', parsed['age_inference_keywords'])
            parsed['reasoning'] += " | REESTIMATED: " + re_parsed.get('reestimate_reasoning', '')
            reestimated = True

        except:
            pass  ### If fails to re-infer the age, it will keep the result from the first inference

    results.append({
        "user_id": user_id,
        "age_distribution": json.dumps(parsed.get("age_distribution", {})),
        "age_id": parsed.get("age_id", ""),
        "age_label": parsed.get("age_label", ""),
        "age_confidence": parsed.get("age_confidence", ""),
        "age_inference_keywords": "/".join(parsed.get("age_inference_keywords", [])),
        "gender_id": parsed.get("gender_id", ""),
        "gender_label": parsed.get("gender_label", ""),
        "gender_confidence": parsed.get("gender_confidence", ""),
        "gender_inference_keywords": "/".join(parsed.get("gender_inference_keywords", [])),
        "content_cat": parsed.get("content_cat", ""),
        "content_subcat": parsed.get("content_subcat", ""),
        "content_keywords": "/".join(parsed.get("content_keywords", [])),
        "reestimated": reestimated,   # 재추정 여부 플래그
        "reasoning": parsed.get("reasoning", "")
    })

    print(f"Done: {user_id} | age={parsed.get('age_label')}({parsed.get('age_confidence')}) | gender={parsed.get('gender_label')}({parsed.get('gender_confidence')}) | reestimated={reestimated}")

##### Block 8 : Save CSV file (Resume mode is available using append) #####
file_exists = os.path.exists(OUTPUT_FILE)
with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    if not file_exists:
        writer.writeheader()
    writer.writerows(results)

print(f"\n Save complete: {OUTPUT_FILE} ({len(results)} users added)")












