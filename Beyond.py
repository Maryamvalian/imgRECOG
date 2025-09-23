from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from eelbrain import plot
import mne
from eelbrain import NDVar,load
from eelbrain._data_obj import VolumeSourceSpace


def GlassBrainVideo(tmin, tmax, dt, nd, vname):
    """
    from Beyond import GlassBrainVideo
    GlassBrainVideo(0.1, 0.8, 0.1, anim, "v4")

    from IPython.display import Video
    video_path="videos/v4.mp4"
    Video(video_path, embed=True, width=1200)
    """
    
    mne.set_log_level("warning")
    
    out_dir = Path("videos/cache/")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tmin= max(tmin, float(nd.time.tmin))      
    tmax = min(float(nd.time.tmax),tmax)     
    dt= min(dt, 0.1)                               
    times =np.arange(tmin, tmax + 1e-12, dt)

    dpi = 100           #image quality
    fps = 5             #video frame rate
    video_path = f"videos/{vname}.mp4"

    stitched_paths = []

    for k, t in enumerate(times, 1):
        p = plot.Butterfly(nd.norm('space'), color='k', title=vname)
        p.add_vline(t, color='tab:blue', lw=2)
        f1 = out_dir / f"butter_{k:04d}.png"
        p.figure.savefig(f1, dpi=dpi, bbox_inches="tight")
        plt.close(p.figure) 

        gb = plot.GlassBrain(nd.sub(time=t), title=f" {t:.2f}s")
        f2 = out_dir / f"glass_{k:04d}.png"
        gb.figure.savefig(f2, dpi=dpi, bbox_inches="tight")
        plt.close(gb.figure)

        a = Image.open(f1); b = Image.open(f2)
        H= max(a.height, b.height)
        def resize_keep_h(img, H):
            W = int(round(img.width * (H / img.height)))
            return img.resize((W, H), Image.LANCZOS)
        a = resize_keep_h(a, H); b = resize_keep_h(b, H)
        W = a.width + b.width
        canvas =Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(a,(0, 0))
        canvas.paste(b ,(a.width, 0))

        frame_path = frames_dir / f"frame_{k:04d}.png"
        canvas.save(frame_path)
        stitched_paths.append(frame_path)

    frames = [imageio.imread(p) for p in stitched_paths]
    imageio.mimsave(video_path, frames, fps=fps, quality=8)
    print(f"Saved video to: {video_path}")


    #-------------
    
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


    #-----------------------------------

def morph_hemi(stc_vec, subject, subject_to="fsaverage2", *, subjects_dir, src_tag="vol-7"):
    
    #For all subjects and fsaverage2 vol-7-L-src , vol-7-R-src should exist in bem folder
    #STC should be on vol-7-lr-src where LH,RH are not merged by eelbrain
    #returns LH,RH,Both (NDvars) morphed to fsaverage2 
    
    src_L = mne.read_source_spaces(
        f"{subjects_dir}/{subject}/bem/{subject}-{src_tag}-L-src.fif", verbose=False)
    src_R = mne.read_source_spaces(
        f"{subjects_dir}/{subject}/bem/{subject}-{src_tag}-R-src.fif", verbose=False)

    # split STC by hemi 
    if len(stc_vec.vertices) != 2:
        raise ValueError("stc_vec must come from an LR source (two vertices arrays: LH & RH).")
    vL, vR = stc_vec.vertices
    nL = len(vL)
    X  = stc_vec.data  

    stc_L = mne.VolVectorSourceEstimate(X[:nL],   [vL], stc_vec.tmin, stc_vec.tstep, stc_vec.subject)
    stc_R = mne.VolVectorSourceEstimate(X[nL:],   [vR], stc_vec.tmin, stc_vec.tstep, stc_vec.subject)

    # subject_to
    src_to_L = mne.read_source_spaces(
        f"{subjects_dir}/{subject_to}/bem/{subject_to}-{src_tag}-L-src.fif", verbose=False)
    src_to_R = mne.read_source_spaces(
        f"{subjects_dir}/{subject_to}/bem/{subject_to}-{src_tag}-R-src.fif", verbose=False)

    #morph each hemi 
    morph_L = mne.compute_source_morph(
        src=src_L, subject_from=subject, subject_to=subject_to,
        subjects_dir=subjects_dir, src_to=src_to_L, precompute=True, verbose=False)
    stc_L_fs = morph_L.apply(stc_L)

    morph_R = mne.compute_source_morph(
        src=src_R, subject_from=subject, subject_to=subject_to,
        subjects_dir=subjects_dir, src_to=src_to_R, precompute=True, verbose=False)
    stc_R_fs = morph_R.apply(stc_R)

    # convert  NDVars 
    an_L_fs = load.mne.stc_ndvar(stc_L_fs, src=f"{src_tag}-L",
                                 subjects_dir=subjects_dir, subject=subject_to)
    an_R_fs = load.mne.stc_ndvar(stc_R_fs, src=f"{src_tag}-R",
                                 subjects_dir=subjects_dir, subject=subject_to)

    #  stitch LH+RH onto the merged whole-brain fsaverage2 
    src_wh = mne.read_source_spaces(
        f"{subjects_dir}/{subject_to}/bem/{subject_to}-{src_tag}-src.fif", verbose=False)
    v_wh = src_wh[0]['vertno']          # merged vertex IDs

    vL_to = stc_L_fs.vertices[0]
    vR_to = stc_R_fs.vertices[0]

    # consistency checks
    overlap = np.intersect1d(vL_to, vR_to)
    if overlap.size:
        raise RuntimeError(f"L/R vertex overlap on target grid: {overlap[:10]}")
    missL = np.setdiff1d(vL_to, v_wh)
    missR = np.setdiff1d(vR_to, v_wh)
    if missL.size or missR.size:
        raise RuntimeError("Some hemi vertices not present in merged fsaverage2 grid; "
                           "check that {src_tag} and pruning match on both sides.")

    # map vertex IDs → row indices in merged grid
    idx_wh = {v: i for i, v in enumerate(v_wh)}
    idxL = np.fromiter((idx_wh[v] for v in vL_to), dtype=int, count=len(vL_to))
    idxR = np.fromiter((idx_wh[v] for v in vR_to), dtype=int, count=len(vR_to))

    # fill whole-brain array
    T = stc_L_fs.data.shape[2]
    data_wh = np.zeros((len(v_wh), 3, T), dtype=stc_L_fs.data.dtype)
    data_wh[idxL] = stc_L_fs.data
    data_wh[idxR] = stc_R_fs.data

    stc_vec_fs_both = mne.VolVectorSourceEstimate(
        data_wh, [v_wh], stc_L_fs.tmin, stc_L_fs.tstep, subject=subject_to)

    # convert ndvar
    an_both = load.mne.stc_ndvar(stc_vec_fs_both, src=src_tag,
                                 subjects_dir=subjects_dir, subject=subject_to)

    return an_R_fs, an_L_fs, an_both
    
    