
def diagnose_disease(symptoms):
    # Rule 1: If symptoms include fever and cough, it might be the flu.
    if 'fever' in symptoms and 'cough' in symptoms:
        return 'Flu'

    #2: If symptoms include fever and headache, it might be a migraine.
    elif 'fever' in symptoms and 'headache' in symptoms:
        return 'Migraine'

    #3: If symptoms include cough and headache, it might be a common cold.
    elif 'cough' in symptoms and 'headache' in symptoms:
        return 'Common Cold'

    #4: If symptoms include only fever, it might be a fever due to infection.
    elif 'fever' in symptoms:
        return 'Infection'

    #5: If symptoms include only cough, it might be a respiratory infection.
    elif 'cough' in symptoms:
        return 'Respiratory Infection'

    #6: If symptoms include only headache, it might be a tension headache.
    elif 'headache' in symptoms:
        return 'Tension Headache'

    # Default: If no symptoms match, the diagnosis is uncertain.
    else:
        return 'Diagnosis Uncertain'

# Example usage
symptoms_input = input("Enter symptoms separated by commas (fever, cough, headache): ").lower().split(', ')
symptoms = [symptom.strip() for symptom in symptoms_input]

disease = diagnose_disease(symptoms)
print(f"Possible Disease: {disease}")