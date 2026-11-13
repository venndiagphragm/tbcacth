"""Build a filtered solicited manifest from the local CODA audit outputs."""
from pathlib import Path
import argparse
import sys

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from src.coda_v5_features import read_physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PACKAGE.parent / 'data')
    args = parser.parse_args()
    root = args.project_root.resolve()
    audit = root / 'dataset-audit' / 'local-full'
    data_root = root / 'dataset' / 'dataset'
    required = [audit / 'audio_inventory.csv', audit / 'solicited_training_manifest_unfiltered.csv',
                audit / 'label_review_required.csv']
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Run audit_dataset.py first; missing {missing}')
    manifest_path = audit / 'solicited_cpu_manifest.csv'
    manifest_path.unlink(missing_ok=True)
    inventory = pd.read_csv(required[0])
    source = pd.read_csv(required[1])
    audio = inventory[inventory.subset == 'solicited'].copy()
    if source.empty or audio.empty:
        raise ValueError('No solicited participants or WAVs in audited input')
    if source.participant.isna().any() or source.tb_status.isna().any() or source.filename.isna().any():
        raise ValueError('Missing participant, filename, or TB label in source metadata')
    if not source.tb_status.isin([0, 1]).all():
        raise ValueError('Unexpected TB labels in source metadata')

    conflicts = []
    for filename, group in source.groupby('filename', dropna=False):
        if len(group) > 1 or group.participant.nunique() != 1 or group.tb_status.nunique() != 1:
            conflicts.extend({'filename': filename, 'participant': row.participant,
                              'reason': 'duplicate_filename_or_conflicting_mapping'}
                             for row in group.itertuples())
    duplicate_names = source.groupby('filename').filter(lambda group: len(group) > 1)
    duplicate_audio = audio.groupby('filename').filter(lambda group: len(group) > 1)
    bad_names = set(duplicate_names.participant)
    bad_audio_names = set(duplicate_audio.filename)
    bad_names.update(source.loc[source.filename.isin(bad_audio_names), 'participant'])
    conflicts.extend({'filename': row.filename, 'participant': row.participant,
                      'reason': 'duplicate_filename'} for row in duplicate_names.itertuples())
    conflicts.extend({'filename': filename, 'participant': participant,
                      'reason': 'duplicate_audio_filename'}
                     for filename in sorted(bad_audio_names)
                     for participant in source.loc[source.filename.eq(filename), 'participant'].unique())
    labels = source.groupby('participant').tb_status.nunique()
    bad_labels = set(labels[labels > 1].index)
    conflicts.extend({'filename': '', 'participant': participant, 'reason': 'conflicting_participant_labels'}
                     for participant in sorted(bad_labels))
    review = pd.read_csv(required[2])
    reference = review.Microbiologicreferencestandard.map({'Positive': 1, 'Negative': 0})
    disagreements = review.loc[reference.notna() & review.tb_status.ne(reference), 'participant']
    bad_reference = set(disagreements)
    conflicts.extend({'filename': '', 'participant': participant, 'reason': 'label_reference_disagreement'}
                     for participant in sorted(bad_reference))
    mapped = source[['participant', 'filename']].drop_duplicates().merge(
        audio[['filename', 'pcm_sha256']], on='filename', how='inner')
    mapped = mapped[mapped.pcm_sha256.notna() & mapped.pcm_sha256.ne('')]
    duplicate_pcm = mapped.groupby('pcm_sha256').participant.nunique()
    bad_pcm = set(duplicate_pcm[duplicate_pcm > 1].index)
    conflicts.extend({'filename': row.filename, 'participant': row.participant,
                      'reason': 'identical_pcm_across_participants'}
                     for row in mapped[mapped.pcm_sha256.isin(bad_pcm)].itertuples())
    conflict_frame = pd.DataFrame(conflicts, columns=['filename', 'participant', 'reason']).drop_duplicates()
    conflict_path = audit / 'solicited_cpu_conflicts_for_review.csv'
    if not conflict_frame.empty:
        conflict_frame.to_csv(conflict_path, index=False)
    else:
        conflict_path.unlink(missing_ok=True)
    excluded_participants = (bad_labels | bad_names | bad_reference
                             | set(mapped.loc[mapped.pcm_sha256.isin(bad_pcm), 'participant']))
    clean = source[~source.participant.isin(excluded_participants)].copy()
    combined = clean.merge(audio, on=['filename'], how='left', validate='one_to_one', suffixes=('', '_audio'))
    missing = combined[combined.path.isna()]
    invalid = [{'participant': row.participant, 'filename': row.filename,
                'reason': 'missing_audio_file'} for row in missing.itertuples()]
    combined = combined[combined.path.notna()].copy()
    if combined.empty:
        raise ValueError('No solicited WAVs are present on disk')
    if combined.participant.isna().any():
        raise ValueError('Audio metadata references participants missing from clinical metadata')
    if combined.tb_status.isna().any():
        raise ValueError('Audio metadata references participants missing TB labels')
    for row in combined.itertuples(index=False):
        reason = ''
        if (pd.notna(row.error) and str(row.error)) or bool(row.truncated):
            reason = 'decode_error_or_truncated'
        elif row.channels != 1 or row.sample_width != 2:
            reason = 'not_mono_pcm16'
        else:
            path = (data_root / row.path).resolve()
            if not path.is_relative_to(data_root.resolve()):
                reason = 'path_outside_dataset'
            else:
                try:
                    read_physical(path)
                except Exception as error:
                    reason = 'read_physical: ' + str(error)
        if reason:
            invalid.append({'participant': row.participant, 'filename': row.filename, 'reason': reason})
    if invalid:
        pd.DataFrame(invalid).to_csv(audit / 'solicited_cpu_manifest_exclusions.csv', index=False)
    else:
        (audit / 'solicited_cpu_manifest_exclusions.csv').unlink(missing_ok=True)
    valid_names = {row['filename'] for row in invalid}
    valid_source = combined[~combined.filename.isin(valid_names)]
    if valid_source.empty:
        raise ValueError('No solicited WAVs pass the physical-format audit')
    valid_source.to_csv(manifest_path, index=False)
    print(f'CPU manifest: {len(valid_source)} WAVs, {valid_source.participant.nunique()} participants')
    print(f'Excluded conflicting participants: {len(excluded_participants)}; invalid WAVs: {len(invalid)}')


if __name__ == '__main__':
    main()
