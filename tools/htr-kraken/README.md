## How to test it with planemo serve

Run from this folder:

```bash
planemo serve --galaxy_root /home/ivo/projects/ufr/galaxy/galaxy-instance \
    --skip_client_build --docker \
    --docker_run_extra_arguments "-v /home/ivo/projects/ufr/galaxy/galaxy-tools-wrapping/kraken/ocr_models:/home/ivo/projects/ufr/galaxy/galaxytools/tools/htr-kraken/test-data/ml_models" 
```