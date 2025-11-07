from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from eelbrain import plot
import mne
from eelbrain import NDVar,load
from eelbrain._data_obj import VolumeSourceSpace
#from scipy.sparse import block_diag, csr_matrix
from scipy.spatial import cKDTree
import pandas as pd




def ndvar_merged_to_stc_lr(ndvar, *, fwd, subject, subjects_dir, src_tag="vol-7"):
    
    tdim=next((d for d in ndvar.dims if getattr(d, 'type', None) == 'time'), None)
    tmin=float(getattr(tdim, 'tmin', 0.0))
    tstep=float(getattr(tdim, 'tstep', 10.0))                    #to ms

    # merged forward src 
    src_merged= fwd['src']; assert len(src_merged) == 1
    actv= src_merged[0]['vertno']              # merged active vertex IDs (1D int array)
    n_from= actv.size

    
    srcL = mne.read_source_spaces(
        f"{subjects_dir}/{subject}/bem/{subject}-{src_tag}-L-src.fif", verbose=False)
    srcR = mne.read_source_spaces(
        f"{subjects_dir}/{subject}/bem/{subject}-{src_tag}-R-src.fif", verbose=False)
    vL = srcL[0]['vertno']
    vR = srcR[0]['vertno']

    
    pos = {int(v): i for i, v in enumerate(actv)}
    try:
        idxL = np.fromiter((pos[int(v)] for v in vL), dtype=int, count=vL.size)
        idxR = np.fromiter((pos[int(v)] for v in vR), dtype=int, count=vR.size)
    except KeyError as e:
        missing = int(e.args[0])
        raise RuntimeError(
            f"Vertex id {missing} from split L/R is not present in merged vertno. "
            "This indicates your merged forward and split vol-7 src were generated with "
            "different pruning/spacing. Rebuild to match."
        )

    
    if np.intersect1d(idxL, idxR).size:
        raise RuntimeError("L/R index overlap via vertex IDs — split files are inconsistent.")
    if idxL.size + idxR.size != n_from:
        
        raise RuntimeError(
            f"Counts mismatch: merged={n_from}, L={idxL.size}, R={idxR.size}. "
            "Ensure split and merged vol-7 come from the same config."
        )

    # extract data in L-then-R order
    Y= ndvar.get_data()
    if Y.ndim == 2:
        # (src, T) scalar
        Y_lr = np.vstack([Y[idxL, :], Y[idxR, :]])
        stc = mne.VolSourceEstimate(Y_lr, [vL, vR], tmin, tstep, subject=subject)
    elif Y.ndim == 3:
        if Y.shape[1] == 3 and Y.shape[2] > 3:
            # (src, 3, T)
            YL = Y[idxL, :, :]; YR = Y[idxR, :, :]
            Y_lr = np.concatenate([YL, YR], axis=0)
            stc = mne.VolVectorSourceEstimate(Y_lr, [vL, vR], tmin, tstep, subject=subject)
        elif Y.shape[2] == 3 and Y.shape[1] > 3:
            # (src, T, 3) → transpose to (src, 3, T)
            Yt = np.transpose(Y, (0, 2, 1))
            YL = Yt[idxL, :, :]; YR = Yt[idxR, :, :]
            Y_lr = np.concatenate([YL, YR], axis=0)
            stc = mne.VolVectorSourceEstimate(Y_lr, [vL, vR], tmin, tstep, subject=subject)
        else:
            raise ValueError(f"Unexpected 3D shape {Y.shape}. Use (src, 3, T) or (src, T, 3).")
    else:
        raise ValueError(f"Unexpected NDVar ndim={Y.ndim}; expected 2 or 3.")
    return stc


#------------------------------------------------------------------------------

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
    #anim_fs= morph_nd(subject, 'fsaverage2', subjects_dir, anim, 'vol-7') 
    #When whole brain- NCRF morphing
   
    
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
        raise RuntimeError("Some hemi vertices not present in merged fsaverage grid ")

    # map vertex IDs → row indices in merged grid
    idx_wh ={v: i for i, v in enumerate(v_wh)}
    idxL= np.fromiter((idx_wh[v] for v in vL_to), dtype=int, count=len(vL_to))
    idxR= np.fromiter((idx_wh[v] for v in vR_to), dtype=int, count=len(vR_to))

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
    
 #------------------------------------------------------

def make_event_table(subject, session, run, *, root="/Users/maryamvalian/Data/ds005810",
                     stim_channel="UPPT001", stim_id=2,                # ID for 'stim_on' in RAW events
                     detailed_events_dir=None  # default: <root>/derivatives/detailed_events
                     ):
    
    root = Path(root)
    subject = str(subject)
    run_str = f"{int(run):02d}"
    bids_meg_dir = root / subject / f"ses-{session}" / "meg"
    events_tsv = bids_meg_dir / f"{subject}_ses-{session}_task-ImageNet_run-{run_str}_events.tsv"
    raw_fif    = bids_meg_dir / f"{subject}_ses-{session}_task-ImageNet_run-{run_str}_meg.fif"

    if detailed_events_dir is None:
        detailed_events_dir = root / "derivatives" / "detailed_events"
    detailed_csv = Path(detailed_events_dir) / f"{subject}_events.csv"

    # animate flags sre in detailed events CSV 
    
    meta = pd.read_csv(detailed_csv)
    meta_sub = meta[(meta["session"] == session) & (meta["run"] == int(run))].reset_index(drop=True)
    if "stim_is_animate" not in meta_sub.columns:
        raise RuntimeError("Missing 'stim_is_animate' in detailed events CSV.")
    anim_flags = meta_sub["stim_is_animate"].astype(str).str.lower().isin(["true", "1", "t", "yes"]).to_numpy()

    #  BIDS events.tsv for exact onsets 
    stim_times = None
    try:
        ev = pd.read_csv(events_tsv, sep="\t")
        if "onset" not in ev.columns:
            raise ValueError("events.tsv has no 'onset' column")

        
        if "trial_type" in ev.columns:
            mask = ev["trial_type"].astype(str).str.lower().isin(["stim_on", "stim"])
        

        stim_times = ev.loc[mask, "onset"].to_numpy(dtype=float)
    except Exception:
        
        print("event.tsv failed")
        raw = mne.io.read_raw_fif(str(raw_fif), preload=False, verbose="error")
        events = mne.find_events(raw, stim_channel=stim_channel, verbose="error")
        stim = events[events[:, 2] == stim_id]
        stim_times = stim[:, 0] / float(raw.info["sfreq"])

    
    n = min(len(stim_times), len(anim_flags))
    if len(stim_times) != len(anim_flags):
        print(f"WARNING [{subject} {session} run {run_str}]: "
              f"times={len(stim_times)} vs animate={len(anim_flags)} → truncating to {n}")

    event_table = pd.DataFrame({
        "time": np.asarray(stim_times[:n], dtype=float),
        "animate": np.asarray(anim_flags[:n], dtype=bool),
    }).reset_index(drop=True)

    return event_table

#----------------------------------_______________________________ 


def fisher_r_to_z(R_matrix):
    
    R_matrix = np.asarray(R_matrix)
    Z_matrix = 0.5 * np.log((1 + R_matrix) / (1 - R_matrix))
    return Z_matrix

