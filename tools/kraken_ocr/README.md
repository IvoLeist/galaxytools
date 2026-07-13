# Kraken-OCR Suite

## Configuration for Galaxy Administrators

### Enabling of GPU Processing for Segmentation and OCR

By default, Kraken performs segmentation and OCR on the CPU. 
GPU execution requires setting the `KRAKEN_DEVICE` environment variable.

Allowed options:
- `cuda:0`, `cuda:1`,... -> use a specific CUDA device
- `cpu` -> use the CPU

Below is an example `job_conf.xml`:

```xml
<job_conf>
  <plugins>
    <plugin 
      id="local" 
      type="runner"
      load="galaxy.jobs.runners.local:LocalJobRunner"/>
  </plugins>

  <destinations default="local">
    <destination id="local" runner="local"/>

    <destination id="gpu" runner="local">
      <env id="KRAKEN_DEVICE">cuda:0</env>
    </destination>
  </destinations>

  <tools>
    <tool id="kraken_segment" destination="gpu"/>
    <tool id="kraken_ocr" destination="gpu"/>
  </tools>
</job_conf>
```

Note, GPU processing is only supported for kraken `segment` and `ocr`.