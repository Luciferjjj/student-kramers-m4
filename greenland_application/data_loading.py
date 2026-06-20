"""
Greenland data and result I/O facade.

The implementation remains in ``student_kramers.data_loading`` during this
structure-only migration. Keeping that source file unchanged preserves the
code fingerprint attached to completed long-running calculations.
"""
from student_kramers.data_loading import (  # noqa: F401
    build_partial_data,
    checkpoint_context,
    ensure_official_excel,
    load_model_fits,
    load_raw_excel,
    load_real_data,
    load_result,
    load_table,
    prepare_checkpoint,
    preprocess_ca2,
    result_path,
    save_model_fits,
    save_table,
)
