from scipy.stats import t
import numpy as np


def loftus_masson(data, confidence=0.95):
    
    """
    Compute Loftus-Masson within-subject confidence intervals.
    Reference: "Using confidence intervals in within-subject designs, Loftus & Masson,1994)

    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            "data must be a 2D array with shape "
            "(n_subjects, n_conditions)"
        )

    if np.isnan(data).any():
        raise ValueError("data contains missing values")

    n_subjects, n_conditions = data.shape

    if n_subjects < 2:
        raise ValueError("At least two subjects are required")

    if n_conditions < 2:
        raise ValueError("At least two conditions are required")

    # Original condition means
    means = data.mean(axis=0)

    # Mean of each subject across conditions
    subject_means = data.mean(axis=1, keepdims=True)

    # Mean across all subjects and conditions
    grand_mean = data.mean()

    # Remove between-subject variability
    normalized_data = data - subject_means + grand_mean

    # Deviations around each condition mean after normalization
    residuals = normalized_data - normalized_data.mean(
        axis=0,
        keepdims=True,
    )

    # Subject × condition interaction sum of squares
    ss_subject_condition = np.sum(residuals**2)

    # Subject × condition degrees of freedom
    df_subject_condition = (
        (n_subjects - 1) * (n_conditions - 1)
    )

    # Subject × condition mean square
    ms_subject_condition = (
        ss_subject_condition / df_subject_condition
    )

    # Loftus-Masson within-subject SEM
    sem_within = np.sqrt(
        ms_subject_condition / n_subjects
    )

    # Critical t-value
    alpha = 1 - confidence
    t_critical = t.ppf(
        1 - alpha / 2,
        df_subject_condition,
    )

    ci = t_critical * sem_within

    return means, sem_within, ci