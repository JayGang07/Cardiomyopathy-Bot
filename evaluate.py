from bert_score import score

# --- SCRIPT CONFIGURATION ---
CANDIDATE_FILENAME = "symptom_analyzer_BERTscore/symptom_report_xyz.txt"
REFERENCE_FILENAME = "symptom_analyzer_BERTscore/reference_summary.txt"
# --------------------------

try:
    with open(CANDIDATE_FILENAME, 'r', encoding='utf-8') as f:
        candidate_text = f.read()

    with open(REFERENCE_FILENAME, 'r', encoding='utf-8') as f:
        reference_text = f.read()

except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Please make sure '{CANDIDATE_FILENAME}' and '{REFERENCE_FILENAME}' are in the same folder as this script.")
    exit()

# Check if either text is empty before scoring
if not candidate_text.strip() or not reference_text.strip():
    print("\n--- BERTScore Results ---")
    print("Error: One of the text files is empty. Cannot calculate score.")
    exit()

# BERTScore expects a list of candidates and a list of references
candidates = [candidate_text]
references = [reference_text]

# Calculate the scores
P, R, F1 = score(
    candidates,
    references,
    lang="en",
    model_type="bert-base-uncased",
    verbose=True
)

# Print the results
print("\n--- BERTScore Results ---")
print(f"Comparing '{CANDIDATE_FILENAME}' (Candidate) vs. '{REFERENCE_FILENAME}' (Reference)")
print(f"Precision (P): {P.mean():.4f}")
print(f"Recall (R):    {R.mean():.4f}")
print(f"F1 Score (F1): {F1.mean():.4f}")
