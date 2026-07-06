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