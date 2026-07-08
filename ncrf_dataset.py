from eelbrain import *
import os

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

