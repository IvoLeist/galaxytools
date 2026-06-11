## How to test it with planemo serve
planemo serve --galaxy_root /Volumes/Dev/galaxy/galaxy-instance --skip_client_build tools/htr/kraken-ocr/kraken-ocr.xml --docker

planemo serve --galaxy_root /home/ivo/projects/ufr/galaxy/galaxy-instance --skip_client_build tools/htr/kraken-ocr/kraken-ocr.xml --docker --docker_run_extra_arguments "-e USER=galaxy"


planemo test --galaxy_root /home/ivo/projects/ufr/galaxy/galaxy-instance --docker --docker_run_extra_arguments "-e USER=galaxy"
