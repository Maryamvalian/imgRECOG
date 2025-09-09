import mne
import numpy as np
from eelbrain import NDVar
from eelbrain._data_obj import VolumeSourceSpace

def morph_nd(subject_from, subject_to, subjects_dir, ndvar, src):
    
   #Morph an NDVar from one subject space to another. Returns morphed ndvar
   
   
    
    src_sub = mne.read_source_spaces(f"{subjects_dir}/{subject_from}/bem/{subject_from}-{src}-src.fif")
    src_fs2 = mne.read_source_spaces(f"{subjects_dir}/{subject_to}/bem/{subject_to}-{src}-src.fif")
    
    
    morph = mne.compute_source_morph(
        src=src_sub, subject_from=subject_from, subject_to=subject_to,
        subjects_dir=subjects_dir, src_to=src_fs2, precompute=True, verbose=False
    )
    M = morph.vol_morph_mat
    
    
    Y = ndvar.get_data()
    
   
    if Y.ndim == 1 or Y.ndim == 2:
        
        Y_fs = M @ Y
        
    elif Y.ndim == 3:
        
        if Y.shape[1] == 3 and Y.shape[2] > Y.shape[1]:
            # Shape: (source, 3_vector_components, time)
            Y_fs = np.stack([M @ Y[:, k, :] for k in range(3)], axis=1)
            
        elif Y.shape[2] == 3 and Y.shape[1] > Y.shape[2]:
            # Shape: (source, time, 3_vector_components)
            Y_fs = np.stack([M @ Y[:, :, k] for k in range(3)], axis=2)
            
        else:
            raise ValueError(f"Unexpected 3D shape: {Y.shape}. Expected either "
                           f"(source, 3, time) or (source, time, 3)")
    else:
        raise ValueError(f"Unexpected NDVar dimensionality: {Y.ndim}. "
                        f"Expected 1, 2, or 3 dimensions")
    
    
    source_fs = VolumeSourceSpace.from_file(subjects_dir, subject_to, src, parc=None)
    
    
    
    ndvar_fs = NDVar(Y_fs, (source_fs,) + ndvar.dims[1:], name='Morphed_nd')
    
    return ndvar_fs