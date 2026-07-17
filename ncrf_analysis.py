from eelbrain import *
import os
import numpy as np

def create_ncrf_dataset(mod, path): 

    cases = [] 
    for i in range(1, 31): 

        if i > 9: 
            sessions = ["ImageNet01"] 
        else: 
            sessions = ["ImageNet03"] 
        subject = f"sub-{i:02d}" 
        for session in sessions: 

            morphed_file = f"{path}/M{subject}-{session}-ncrf.pickle" 
            if os.path.exists(morphed_file): 
                print(f"Loading {subject}-{session} from file.") 
                inanim, anim = load.unpickle(morphed_file) 

                if mod == "effect": 
                    cases.append([subject, session, "contrast", anim]) 
                    cases.append([subject, session, "general", inanim]) 
                    
                elif mod == "dummy": 
                    cases.append([subject, session, "inanimate", inanim]) 
                    cases.append([subject, session, "animate", anim]) 

                else: 
                   raise ValueError("mod must be either 'effect' or 'dummy'") 

            else: 
                print(f"File error: {morphed_file}")
    data = Dataset.from_caselist( ["subject", "session", "animacy", "ncrf"], cases ) 

    return data 

def ncrf_stats(
    data,
    comparison,
    mod,
    condition=None,
    tstart=80,
    tstop=600,
    samples=1000,
    tfce=True,
):
    """
    Run NCRF statistical tests.

    """

    if comparison == "condition":

        valid_conditions = {
            "dummy": ("animate", "inanimate"),
            "effect": ("general", "contrast"),
        }

        if mod not in valid_conditions:
            raise ValueError("mod must be 'dummy' or 'effect'.")

        if condition not in valid_conditions[mod]:
            raise ValueError(
                f"For mod='{mod}', condition must be one of {valid_conditions[mod]}."
            )

        condition_data = data.sub(f"animacy == '{condition}'")

        return testnd.Vector(
            "ncrf",
            match="subject",
            data=condition_data,
            tfce=tfce,
            tstart=tstart,
            tstop=tstop,
            samples=samples,
        )

    elif comparison == "paired":

        if mod != "dummy":
            raise ValueError("Paired test is only supported for mod='dummy'.")

        return testnd.VectorDifferenceRelated(
            "ncrf",
            "animacy",
            "inanimate",
            "animate",
            match="subject",
            data=data,
            tfce=tfce,
            tstart=tstart,
            tstop=tstop,
            samples=samples,
        )

    else:
        raise ValueError("comparison must be 'consition' or 'paired'.")

# Sparsity Analysis
def source_l1_timecourse(ndvar):
    """
    Compute the L1 norm across sources at each time point.
    """
    x = ndvar.x
    if x.ndim == 3:
        x = np.linalg.norm(x, axis=1)

    return np.asarray(ndvar.time.times), np.abs(x).sum(axis=0)


def get_contrast(data, subject, mod):
    """
    Return the contrast NDVar for one subject. 
    """
    dsub = data.sub(f"subject == '{subject}'")

    if mod == "effect":
        return dsub.sub("animacy == 'contrast'")["ncrf"][0]

    if mod == "dummy":
        anim = dsub.sub("animacy == 'animate'")["ncrf"][0]
        inanim = dsub.sub("animacy == 'inanimate'")["ncrf"][0]
        return anim - inanim

    raise ValueError("mod must be 'effect' or 'dummy'")


def compute_l1(data, mod):
    """
    Compute contrast L1 time courses for all subjects.
    """
    times = None
    values = {}

    for subject in sorted(set(data["subject"])):
        contrast = get_contrast(data, subject, mod)
        current_times, l1 = source_l1_timecourse(contrast)

        if times is None:
            times = current_times
        elif not np.allclose(times, current_times):
            raise ValueError(f"L1 time axis mismatch ")

        values[subject] = l1

    return times, values


def significant_source_timecourse(result):
    """
    Count sources with p < 0.05 at each time point.
    """
    return (np.asarray(result.p.time.times),np.sum(result.p.x < 0.05, axis=0))
    

