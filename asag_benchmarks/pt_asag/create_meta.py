import pandas as pd
import json
import os

def create_unified_meta(pt_asag_dir):
    # Paths to the CSV files
    questions_path = os.path.join(pt_asag_dir, 'questions.csv')
    ref_answers_path = os.path.join(pt_asag_dir, 'reference_answers.csv')
    ref_answers_ext_path = os.path.join(pt_asag_dir, 'reference_answers_extended.csv')
    concepts_path = os.path.join(pt_asag_dir, 'concepts.csv')
    student_answers_path = os.path.join(pt_asag_dir, 'student_answers.csv')

    # Load the CSV files
    df_questions = pd.read_csv(questions_path)
    df_ref_answers = pd.read_csv(ref_answers_path)
    df_ref_answers_ext = pd.read_csv(ref_answers_ext_path)
    df_concepts = pd.read_csv(concepts_path)
    df_student_answers = pd.read_csv(student_answers_path)

    # Union reference answers
    df_ref_all = pd.concat([df_ref_answers, df_ref_answers_ext]).drop_duplicates()

    meta_data = {}

    for _, row in df_questions.iterrows():
        q_id = str(row['question_id'])
        q_text = row['question_text']

        # Get sample solutions (reference answers)
        solutions = df_ref_all[df_ref_all['question_id'] == row['question_id']]['refans_text'].tolist()
        
        # Get concepts
        concepts = df_concepts[df_concepts['question_id'] == row['question_id']]['concept_text'].tolist()
        
        # Get grades/levels
        grades = df_student_answers[df_student_answers['question_id'] == row['question_id']]['grade'].unique()
        # Sort grades and convert to standard python types
        sorted_grades = sorted([int(g) for g in grades if pd.notnull(g)])
        
        meta_data[q_id] = {
            "question": q_text,
            "sample_solutions": solutions,
            "number_of_levels": len(sorted_grades),
            "labels": sorted_grades, # Including labels as per scientsbank style
            "concepts": concepts
        }

    output_path = os.path.join(pt_asag_dir, 'questions_meta.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created unified meta file at: {output_path}")

if __name__ == "__main__":
    pt_asag_dir = "/Users/sunzhifan/Desktop/dipf_work/asag_augment/pt_asag"
    create_unified_meta(pt_asag_dir)
