#Prepare anatomical models and MEG-MRI coregistration for NOD-MEG subjects.

from __future__ import annotations

import os
from pathlib import Path

import mne
from eelbrain.mne_fixes._source_space import (
    merge_volume_source_space,
    prune_volume_source_space,
)



# Configuration

DATASET_ROOT = Path("~/Data/ds005810").expanduser()
FREESURFER_DIR = DATASET_ROOT / "derivatives" / "freesurfer"
SUBJECTS_DIR = FREESURFER_DIR / "subjects"
RAW_DIR = DATASET_ROOT / "derivatives" / "preprocessed" / "raw"
TRANS_DIR = DATASET_ROOT / "derivatives" / "trans"

SUBJECT_NUMBERS = range(1, 31)

SOURCE_SPACING_MM = 7.0
SOURCE_NEIGHBOR_COUNT = 3
SOURCE_FILL_HOLES = 4
ICP_ITERATIONS = 50

# A single available MEG run is sufficient to obtain measurement info 
COREG_RUN = 1


def session_for_subject(subject_number: int) -> str:
    """Return the NOD-MEG session name for a subject."""
    return "ImageNet03" if subject_number <= 9 else "ImageNet01"


def raw_file_for_subject(subject: str, session: str) -> Path:
    """Return the preprocessed raw FIF file used to obtain MEG info."""
    return (
        RAW_DIR
        / f"{subject}_ses-{session}_task-ImageNet_run-{COREG_RUN:02d}_clean_meg.fif"
    )


def create_bem(subject: str) -> Path:
    """Create watershed surfaces and a single-layer MEG BEM solution."""
    print(f"\n[{subject}] Creating watershed BEM surfaces...")

    mne.bem.make_watershed_bem(
        subject=subject,
        subjects_dir=SUBJECTS_DIR,
        overwrite=True,
        verbose=False,
    )

    print(f"[{subject}] Creating single-layer BEM model and solution...")

    model = mne.make_bem_model(
        subject=subject,
        subjects_dir=SUBJECTS_DIR,
        ico=5,
        conductivity=(0.3,),
        verbose=False,
    )
    bem_solution = mne.make_bem_solution(model)

    output_file = SUBJECTS_DIR / subject / "bem" / f"{subject}-bem-sol.fif"
    mne.write_bem_solution(
        output_file,
        bem_solution,
        overwrite=True,
        verbose=False,
    )

    print(f"[{subject}] Saved BEM solution: {output_file}")
    return output_file


def create_coregistration(
    subject: str,
    session: str,
) -> Path:
    """Fit fiducials and ICP, then save the head-to-MRI transform."""
    raw_file = raw_file_for_subject(subject, session)

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Coregistration input file was not found:\n{raw_file}\n"
            "Update RAW_DIR or COREG_RUN in the Configuration section."
        )

    print(f"[{subject}] Reading MEG info from: {raw_file}")
    info = mne.io.read_info(raw_file, verbose=False)

    print(f"[{subject}] Fitting manually defined fiducials...")
    coreg = mne.coreg.Coregistration(
        subject=subject,
        info=info,
        subjects_dir=SUBJECTS_DIR,
    )
    coreg.fit_fiducials()

    print(f"[{subject}] Running ICP with {ICP_ITERATIONS} iterations...")
    coreg.fit_icp(n_iterations=ICP_ITERATIONS)

    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = TRANS_DIR / f"{subject}-{session}-trans.fif"

    mne.write_trans(
        output_file,
        coreg.trans,
        overwrite=True,
    )

    print(f"[{subject}] Saved transformation: {output_file}")
    return output_file


def create_source_space(
    subject: str,
    bem_file: Path,
) -> Path:
    """Create and save a pruned 7-mm cortical volume source space."""
    aseg_file = SUBJECTS_DIR / subject / "mri" / "aseg.mgz"

    if not aseg_file.exists():
        raise FileNotFoundError(
            f"FreeSurfer segmentation was not found:\n{aseg_file}"
        )

    cortex_labels = [
        "Left-Cerebral-Cortex",
        "Right-Cerebral-Cortex",
    ]

    print(
        f"[{subject}] Creating {SOURCE_SPACING_MM:g}-mm cortical "
        "volume source space..."
    )

    source_spaces_lr = mne.setup_volume_source_space(
        subject=subject,
        subjects_dir=SUBJECTS_DIR,
        mri=aseg_file,
        bem=bem_file,
        volume_label=cortex_labels,
        pos=SOURCE_SPACING_MM,
        verbose=False,
    )

    source_space_merged = merge_volume_source_space(
        source_spaces_lr,
        "cortex",
    )

    source_space = prune_volume_source_space(
        source_space_merged,
        SOURCE_SPACING_MM,
        SOURCE_NEIGHBOR_COUNT,
        remove_midline=True,
        fill_holes=SOURCE_FILL_HOLES,
    )

    output_file = (
        SUBJECTS_DIR
        / subject
        / "bem"
        / f"{subject}-vol-{int(SOURCE_SPACING_MM)}-src.fif"
    )

    mne.write_source_spaces(
        output_file,
        source_space,
        overwrite=True,
    )

    print(f"[{subject}] Saved source space: {output_file}")
    return output_file


def prepare_subject(subject_number: int) -> None:
    """Run all preparation steps for one subject."""
    subject = f"sub-{subject_number:02d}"
    session = session_for_subject(subject_number)

    print("\n" + "=" * 72)
    print(f"Preparing {subject} ({session})")
    print("=" * 72)

    bem_file = create_bem(subject)
    create_coregistration(subject, session)
    create_source_space(subject, bem_file)

    print(f"[{subject}] Preparation completed successfully.")


def main() -> None:
    """Prepare all configured subjects."""
    os.environ["SUBJECTS_DIR"] = str(SUBJECTS_DIR)

    if not SUBJECTS_DIR.exists():
        raise FileNotFoundError(
            f"FreeSurfer subjects directory was not found:\n{SUBJECTS_DIR}"
        )

    failed_subjects: list[tuple[str, str]] = []

    for subject_number in SUBJECT_NUMBERS:
        subject = f"sub-{subject_number:02d}"

        try:
            prepare_subject(subject_number)
        except Exception as error:
            failed_subjects.append((subject, str(error)))
            print(f"\n[{subject}] ERROR: {error}")
            print(f"[{subject}] Continuing with the next subject.")

    print("\n" + "=" * 72)

    if failed_subjects:
        print("Processing finished with errors for:")
        for subject, error in failed_subjects:
            print(f"- {subject}: {error}")
    else:
        print("All subjects were prepared successfully.")


if __name__ == "__main__":
    main()