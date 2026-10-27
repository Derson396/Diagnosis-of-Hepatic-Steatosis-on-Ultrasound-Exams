import math
import random


def split_patient_validation(patient_ids, validation_fraction=0.2, random_state=42):
    patients = list(dict.fromkeys(patient_ids))
    if len(patients) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("A validação exige ao menos dois pacientes e uma fração entre 0 e 1.")

    random.Random(random_state).shuffle(patients)
    validation_count = min(len(patients) - 1, max(1, math.ceil(len(patients) * validation_fraction)))
    validation = set(patients[:validation_count])
    return set(patients[validation_count:]), validation
