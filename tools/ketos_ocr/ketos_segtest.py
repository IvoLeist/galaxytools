"""Run Ketos with the Kraken 7.1.1 segtest report-indexing fix.

Pixel metrics contain only auxiliary and region classes, ordered by class ID.
The upstream report incorrectly indexes these compact arrays by model class ID
(which also includes baseline classes). Remove this shim when upgrading to a
Kraken release with the corrected report loop.
"""

import inspect


def corrected_source(source):
    """Apply only the known report fix, failing visibly if upstream changes."""
    replacements = {
        "for idx in pixel_idxs:": "for metric_idx, idx in enumerate(pixel_idxs):",
        "class_pixel_accuracy[idx]": "class_pixel_accuracy[metric_idx]",
        "class_iu[idx]": "class_iu[metric_idx]",
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise RuntimeError("Kraken segtest source changed; review the report compatibility fix")
        source = source.replace(original, replacement)
    return source


def main():
    from kraken.ketos import cli, segmentation

    # Compile only segtest into a private namespace; leave installed files and
    # other Ketos commands untouched. Preserve Click decorators and CLI defaults.
    source = corrected_source(inspect.getsource(segmentation.segtest.callback))
    namespace = vars(segmentation).copy()
    exec(compile(source, inspect.getfile(segmentation), "exec"), namespace)
    cli.add_command(namespace["segtest"])
    cli()


if __name__ == "__main__":
    main()
