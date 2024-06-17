from kanren import Relation, facts, run
import cv2
import numpy as np

def get_size(brain_mask_path, tumor_mask_path):
    
    b_mask = cv2.imread(brain_mask_path, cv2.IMREAD_GRAYSCALE)
    t_mask = cv2.imread(tumor_mask_path, cv2.IMREAD_GRAYSCALE)
    
    _, thrsh = cv2.threshold(b_mask, 127, 255, cv2.THRESH_BINARY)
    
    total_area = cv2.countNonZero(thrsh)
    
    # Calculate the area, width of tumor in pixels
    tumor_mask = cv2.resize(t_mask, (256,256))
    _, thrsh = cv2.threshold(tumor_mask, 127, 255, cv2.THRESH_BINARY)
            
    # Area
    area = cv2.countNonZero(thrsh)
    
    
    print(area)
    print(total_area)
    # First size
    if area <= ( 0.1 * total_area):
        size = "I"
    # Second size
    elif area <= ( 0.3 * total_area):
        size = "II"
    # Third size
    elif area <= ( 0.45 * total_area):
        size = "III"
    # Fourth size
    else:
        size = "IV"
        
    return size.lower()


def get_type(b_mask_path, t_mask_path, g, s, age, e, hist, w, exp, a, app, o, strs):
    
    size = get_size(b_mask_path, t_mask_path)
    
    if e != 'white':
        e = 'black'
    
    # Define relations for risk factors of malignant and benign
    potential_risk_factor_for_malignant = Relation()
    potential_risk_factor_for_benign = Relation()

    # Facts for potential risk factors
    #Facts(tumortype, gender, smoker, age, ethnicity, size, family history, weight, exposure, alcohol use, appearance, other_cancers, in stress?)
    facts(potential_risk_factor_for_benign, ('female',), ('not_smoker',), ('<=40',), ('white',), ('i',), ('ii',), ('tumor_family_history',),
        ('no_overweight',), ('toxins_or_radiation',), ('not_drinking',), ('didnt_appear_before',), ('no_other_cancer',), ('no_stress',))
    
    facts(potential_risk_factor_for_malignant, ('male',), ('smoker',), ('>40',), ('black',), ('iii',), ('iv',), ('cancer_family_history',),
        ('overweight',), ('carncinogens',), ('drinking',), ('appeared_before',), ('has_other_cancer',), ('stress',))

    # Function to check the correctness of an answer
    def is_malignant_risk_factor(factor):
        return potential_risk_factor_for_malignant(factor)

    def is_benign_risk_factor(factor):
        return potential_risk_factor_for_benign(factor)

    # Define a scoring function
    def update_score(score, correct):
        if correct:
            return score + 1
        else:
            return score

    # Example questions and answers
    risk_factors = [g, s, age, e, size, hist, w, exp, a, app, o, strs]

    # Initialize score
    score1 = 0
    score2 = 0

    # Simulate answering the questions
    for factor in risk_factors:
        if run(1, factor, is_malignant_risk_factor(factor)):
            score1 = update_score(score1, True)
        elif run(1, factor, is_benign_risk_factor(factor)):
            score2 = update_score(score2, True)
    
    if score1 > score2:
        t = 'm'
        tumor_type = 'You may have a malignant tumor.'
    elif score1 < score2:
        t = 'b'
        tumor_type = 'Your conditions indicate a benign tumor.'
    elif score1 == score2 and (score1 != 0 or score2 != 0):
        t = 'u'
        tumor_type = 'Your conditions indicate that you may have either a malignant or a benign tumor.'

    
    if t == 'u' and (size == 'i' or size == 'ii'):
        grade = "You may have a Grade-I or Grade-II tumor."
    elif t == 'u' and (size == 'iii' or size == 'iv'):
        grade = "You may have a Grade-III or Grade-IV tumor."
        
    if t == 'b' and (size == 'i' or size == 'ii'):
        grade = "You may have a Grade-I tumor."
    elif t == 'b' and (size == 'iii' or size == 'iv'):
        grade = "You may have a Grade-II tumor."

    if t == 'm' and (size == 'i' or size == 'ii'):
        grade = "You may have a Grade-III tumor."
    elif t == 'm' and (size == 'iii' or size == 'iv'):
        grade = "You may have a Grade-IV tumor."

    return tumor_type, grade